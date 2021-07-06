"""
Code to test drive the phase variable for level-ground walking 
"""
import sys, time, os
from flexsea import flexsea as flex
from flexsea import fxEnums as fxe
#import matplotlib.pyplot as plt
import numpy as np

sys.path.append(r'/home/pi/OSL-master/locoAPI/') # Path to Loco module
#sys.path.append(r'/home/pi/.local/bin')
sys.path.append(r'/usr/share/python3-mscl/')     # Path of the MSCL - API for the IMU
import locoOSL as loco                           # Module from Locolab
import mscl as msl                               # Module from Microstrain

sys.path.append(r'/home/pi/prosthetic_phase_estimation/')
from EKF import *
from model_framework import *
from scipy.signal import butter, lfilter, lfilter_zi
import sender_test as sender   # for real-time plotting

# -----------------TODO: change these constants to match your setup ----------------------
ANK_PORT = r'/dev/ttyACM1'
KNE_PORT = r'/dev/ttyACM0'             # last number indicates the order for starting the actuators
IMU_PORT = r'/dev/ttyUSB0'
TIMEOUT  = 500                          #Timeout in (ms) to read the IMU
fxs = flex.FlexSEA()
# ------------------ INITIALIZATION ------------------------------------------------------
# Connect with knee and ankle actuator packs
kneID = fxs.open(port = KNE_PORT, baud_rate = 230400, log_level = 6)
ankID = fxs.open(port = ANK_PORT, baud_rate = 230400, log_level = 6)

# Initialize IMU - The sample rate (SR) of the IMU controls the SR of the OSL 
connection = msl.Connection.Serial(IMU_PORT, 921600)
IMU = msl.InertialNode(connection)
IMU.setToIdle()
packets = IMU.getDataPackets(TIMEOUT)   # Clean the internal circular buffer.

# Set streaming        
IMU.resume()
fxs.start_streaming(kneID, freq = 100, log_en = False)
fxs.start_streaming(ankID, freq = 100, log_en = False)
time.sleep(1)                           # Healthy pause before using ActPacks or IMU

# Soft start
G_K = {"kp": 40, "ki": 400, "K": 10, "B": 0, "FF": 128}  # Knee controller gains
G_A = {"kp": 40, "ki": 400, "K": 10, "B": 0, "FF": 128}  # Ankle controller gains
fxs.set_gains(ankID, G_A["kp"], G_A["ki"], 0, G_A["K"], G_A["B"], G_A["FF"])
fxs.set_gains(kneID, G_K["kp"], G_K["ki"], 0, G_K["K"], G_K["B"], G_K["FF"])
fxs.send_motor_command(ankID, fxe.FX_IMPEDANCE, fxs.read_device(ankID).mot_ang)
fxs.send_motor_command(kneID, fxe.FX_IMPEDANCE, fxs.read_device(kneID).mot_ang)
time.sleep(2/100)

# ------------------ MAIN LOOP -----------------------------------------------------------
try:
    # For gain details check https://dephy.com/wiki/flexsea/doku.php?id=controlgains
    G_K = {"kp": 40, "ki": 400, "K": 30, "B": 160, "FF": 128}  # Knee controller gains
    G_A = {"kp": 40, "ki": 400, "K": 30, "B": 160, "FF": 128}  # Ankle controller gains

    kneSta  = fxs.read_device(kneID)
    ankSta  = fxs.read_device(ankID)
    IMUPac  = IMU.getDataPackets(TIMEOUT)
    
    fxs.set_gains(ankID, G_A["kp"], G_A["ki"], 0, G_A["K"], G_A["B"], G_A["FF"])
    fxs.set_gains(kneID, G_K["kp"], G_K["ki"], 0, G_K["K"], G_K["B"], G_K["FF"])

    # Load trajectory
    refTrajectory  = loco.loadTrajectory(trajectory = 'walking')
    refAnk = refTrajectory ["ankl"]
    refKne = refTrajectory ["knee"]

    # Create encoder map
    kneSta  = fxs.read_device(kneID)
    ankSta  = fxs.read_device(ankID)
    IMUPac  = IMU.getDataPackets(TIMEOUT)
    encMap  = loco.read_enc_map( loco.read_OSL(kneSta, ankSta, IMUPac) ) 

    dataOSL = loco.read_OSL(kneSta, ankSta, IMUPac)
    cmd_log = {'refAnk': [0.0, 'deg'], 'refKnee': [0.0, 'deg']}
    ekf_log = {'phase': [0.0, '-'], 'phase_dot':[0.0, '1/s'], 'stride_length':[0.0, 'm'], 'ramp':[0.0, 'deg'],
               'thigh_angle_pred': [0.0, 'deg'], 'thigh_angle_vel_pred': [0.0, 'deg/s'], 'atan2_pred':[0.0, '-'],
               'thigh_angle_vel': [0.0, 'deg/s'], 'atan2':[0.0, '-']}
    logger = loco.ini_log({**dataOSL, **cmd_log, **ekf_log}, sensors = "all_sensors", trialName = "OSL_benchtop_test")

    ### Intitialize EKF
    sensors = [0, 6, 7]
    sensor_keys = ['global_thigh_angle', 'global_thigh_angle_vel', 'atan2']
    arctan2 = False
    if sensors[-1] == 7:
        arctan2 = True
    with open('R_s.pickle', 'rb') as file:
        R = pickle.load(file)

    m_model = model_loader('Measurement_model_' + str(len(sensors)) +'_sp.pickle')
    Psi = np.array([load_Psi('Generic')[key] for key in sensor_keys], dtype = object)
    saturation_range = [1, 0, 2, 0.8] 
    
    ## build the system
    sys = myStruct()
    sys.f = process_model
    sys.A = A
    sys.h = m_model
    sys.Q = np.diag([0, 1e-7, 1e-7, 0])
    # measurement noise covariance
    sys.R = R['Generic'][np.ix_(sensors, sensors)]
    U = np.diag([2, 2, 2])
    sys.R = U @ sys.R @ U.T

    # initialize the state
    init = myStruct()
    init.x = np.array([[0], [0.5], [1.1], [0]])
    init.Sigma = np.diag([1, 1, 1, 0])

    ekf = extended_kalman_filter(sys, init)

    ### Create filters 
    fs = 100          # sampling rate = 100Hz (actual: ~77Hz)
    nyq = 0.5 * fs    # Nyquist frequency = fs/2
    # configure low-pass filter (1-order)
    normal_cutoff = 1 / nyq   #cut-off frequency = 2Hz
    b_lp, a_lp = butter(1, normal_cutoff, btype = 'low', analog = False)
    z_lp_1 = lfilter_zi(b_lp,  a_lp)
    z_lp_2 = lfilter_zi(b_lp,  a_lp)
    
    # configure band-pass filter (2-order)
    normal_lowcut = 0.1 / nyq    #lower cut-off frequency = 0.5Hz
    normal_highcut = 1 / nyq     #upper cut-off frequency = 2Hz
    b_bp, a_bp = butter(2, [normal_lowcut, normal_highcut], btype = 'band', analog = False)
    z_bp = lfilter_zi(b_bp,  a_bp)

    if input('\n\nAbout to walk. Would you like to continue? (y/n): ').lower() == 'y':
        print("\n Let's walk!")
    else:
        sys.exit("User stopped the execution")

    ptr = 0
    t_0 = time.time()     # for EKF
    start_time = t_0      # for live plotting

    while True:
        # Read OSL
        kneSta  = fxs.read_device(kneID)
        ankSta  = fxs.read_device(ankID)
        IMUPac  = IMU.getDataPackets(TIMEOUT)  
        dataOSL = loco.read_OSL(kneSta, ankSta, IMUPac, logger['ini_time'], encMap)

        ### measurement data
        global_thigh_angle = dataOSL['ThighSagi'][0] * 180 / np.pi
        ankle_angle = dataOSL['ankJoiPos'][0] * 180 / np.pi
        knee_angle = dataOSL['kneJoiPos'][0] * 180 / np.pi
        
        # time
        t = time.time()
        dt = t - t_0
        t_0 = t

        ## Compute global thigh angle velocity
        if ptr == 0:
            global_thigh_angle_vel_lp = 0 
        else:
            global_thigh_angle_vel = (global_thigh_angle - global_thigh_angle_0) / dt
            # low-pass filtering
            global_thigh_angle_vel_lp, z_lp_1 = lfilter(b_lp, a_lp, [global_thigh_angle_vel], zi = z_lp_1)
            global_thigh_angle_vel_lp = global_thigh_angle_vel_lp[0]

        global_thigh_angle_0 = global_thigh_angle

        ## Compute atan2
        # band-pass filtering
        global_thigh_angle_bp, z_bp = lfilter(b_bp, a_bp, [global_thigh_angle], zi = z_bp) 
        global_thigh_angle_bp = global_thigh_angle_bp[0]
        if ptr == 0:
            global_thigh_angle_vel_blp = 0
        else:
            global_thigh_angle_vel_bp = (global_thigh_angle_bp - global_thigh_angle_bp_0) / dt
            # low-pass filtering
            global_thigh_angle_vel_blp, z_lp_2 = lfilter(b_lp, a_lp, [global_thigh_angle_vel_bp], zi = z_lp_2)
            global_thigh_angle_vel_blp = global_thigh_angle_vel_blp[0]

        global_thigh_angle_bp_0 = global_thigh_angle_bp

        Atan2 = np.arctan2(-global_thigh_angle_vel_blp / (2*np.pi*0.8), global_thigh_angle_bp)
        if Atan2 < 0:
            Atan2 = Atan2 + 2 * np.pi

        measurement = np.array([[global_thigh_angle], [global_thigh_angle_vel_lp], [Atan2]])
        measurement = np.squeeze(measurement)

        ### EKF implementation
        ekf.prediction(dt)
        ekf.state_saturation(saturation_range)
        ekf.correction(measurement, Psi, arctan2)
        ekf.state_saturation(saturation_range)
        
        ### Control commands: joints angles
        ## 1) Control commands generated by the trainned model
        """
        joint_angles = joints_control(ekf.x[0, 0], ekf.x[1, 0], ekf.x[2, 0], ekf.x[3, 0])
        knee_angle_model = joint_angles[0]
        ankle_angle_model = -joint_angles[1] # negative sign
        # saturate commands to actuators
        if knee_angle_model > -5: 
            knee_angle_model = -5
        if knee_angle_model < -50: 
            knee_angle_model = -50
        
        if ankle_angle_model > 18: 
            ankle_angle_model = 18
        if ankle_angle_model < -10: 
            ankle_angle_model = -10
        """
        
        ## 2) Control commands generated by the established trajectory
        pv = int(ekf.x[0, 0] * 998)  # phase variable conversion (scaling)
        ankle_angle_cmd = refAnk[pv]
        knee_angle_cmd = refKne[pv]

        ### Logging data
        # log joint control commands
        cmd_log['refAnk'][0] = ankle_angle_cmd
        cmd_log['refKnee'][0] = knee_angle_cmd
        # log ekf
        ekf_log['phase'][0] = ekf.x[0, 0]
        ekf_log['phase_dot'][0] = ekf.x[1, 0]
        ekf_log['stride_length'][0] = ekf.x[2, 0]
        ekf_log['ramp'][0] = ekf.x[3, 0]
        ekf_log['thigh_angle_pred'][0] = ekf.z_hat[0]
        ekf_log['thigh_angle_vel_pred'][0] = ekf.z_hat[1]
        ekf_log['atan2_pred'][0] = ekf.z_hat[2]
        ekf_log['thigh_angle_vel'][0] = global_thigh_angle_vel_lp
        ekf_log['atan2'][0] = Atan2
        loco.log_OSL({**dataOSL,**cmd_log, **ekf_log}, logger)
        
        ### Move the OSL
        ankMotCou, kneMotCou = loco.joi2motTic(encMap, knee_angle_cmd, ankle_angle_cmd)
        fxs.send_motor_command(ankID, fxe.FX_IMPEDANCE, ankMotCou)
        fxs.send_motor_command(kneID, fxe.FX_IMPEDANCE, kneMotCou)

        ### Live plotting
        elapsed_time = time.time() - start_time
        if ptr % 2 == 0:
            sender.graph(elapsed_time,
                         ekf.x[0, 0], ekf.x[0, 0], 'Phase', '--',
                         #ekf.x[0, 0], 'phase', '-',
                         #ekf.x[1, 0], 'phase_dot', '1/s',
                         #ekf.x[2, 0], 'step_length', 'm',
                         #ekf.x[3, 0], 'ramp_angle', 'deg'
                         global_thigh_angle, ekf.z_hat[0], 'Global Thigh Angle', 'deg',
                         #global_thigh_angle_vel_lp, ekf.z_hat[1], 'Global Thigh Angle Vel', 'deg/s',
                         # Atan2, ekf.z_hat[2], 'Atan2', '-',
                         knee_angle, knee_angle_cmd, 'Knee Angle', 'deg',
                         ankle_angle, ankle_angle_cmd, 'Ankle Angle', 'deg'
                         )
        print('Elapsed time:', elapsed_time, ptr)
        ptr+=1

except KeyboardInterrupt:
        print('\n*** OSL shutting down ***\n')

finally:        
    # Do anything but turn off the motors at the end of the program
    fxs.send_motor_command(ankID, fxe.FX_NONE, 0)
    fxs.send_motor_command(kneID, fxe.FX_NONE, 0)
    IMU.setToIdle()
    time.sleep(0.5)    
    fxs.close(ankID)
    fxs.close(kneID)  
    print('Communication with ActPacks closed and IMU set to idle')