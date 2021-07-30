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
from collections import deque
import sender_test as sender   # for real-time plotting

# -----------------TODO: change these constants to match your setup ------------------------------------------------
ANK_PORT = r'/dev/ttyACM1'
KNE_PORT = r'/dev/ttyACM0'             # last number indicates the order for starting the actuators
IMU_PORT = r'/dev/ttyUSB0'
TIMEOUT  = 500                          #Timeout in (ms) to read the IMU
fxs = flex.FlexSEA()
# --------------------- INITIALIZATION -----------------------------------------------------------------------------
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

try:
    ## Set controller gains =========================================================================================
    # For gain details check https://dephy.com/wiki/flexsea/doku.php?id=controlgains
    # 1) Medium gains (worked in the ankle tracking test)
    #G_K = {"kp": 40, "ki": 400, "K": 300, "B": 150, "FF": 60}  # Knee controller gains
    #G_A = {"kp": 40, "ki": 400, "K": 300, "B": 150, "FF": 60}  # Ankle controller gains

    # 2) Large  gains (worked in the ankle tracking test)
    #G_K = {"kp": 40, "ki": 400, "K": 300, "B": 350, "FF": 60}  # Knee controller gains
    G_A = {"kp": 40, "ki": 400, "K": 600, "B": 300, "FF": 128}  # Ankle controller gains

    # 3) Gains used in Ross's phaseVariableSiavash.py
    G_K = {"kp": 40, "ki": 400, "K": 500, "B": 1500, "FF": 128}  # Knee controller gains
    #G_A = {"kp": 40, "ki": 400, "K": 500, "B": 2500, "FF": 128}  # Ankle controller gains

    # 4) Gains used in Ross's phaseVariableStairAscent.py
    #G_K = {"kp": 40, "ki": 400, "K": 300, "B": 1600, "FF": 128}  # Knee controller gains
    #G_A = {"kp": 40, "ki": 400, "K": 300, "B": 1600, "FF": 128}  # Ankle controller gains

    #================================================================================================================

    kneSta  = fxs.read_device(kneID)
    ankSta  = fxs.read_device(ankID)
    IMUPac  = IMU.getDataPackets(TIMEOUT)
    
    fxs.set_gains(ankID, G_A["kp"], G_A["ki"], 0, G_A["K"], G_A["B"], G_A["FF"])
    fxs.set_gains(kneID, G_K["kp"], G_K["ki"], 0, G_K["K"], G_K["B"], G_K["FF"])

    # Load trajectory
    refTrajectory  = loco.loadTrajectory(trajectory = 'walking')
    refAnk = refTrajectory["ankl"]
    refKne = refTrajectory["knee"]
    #refHip = refTrajectory["hip_"]
    #maxHip = np.amax(refHip)
    #minHip = np.amin(refHip)

    # Create encoder map
    kneSta  = fxs.read_device(kneID)
    ankSta  = fxs.read_device(ankID)
    IMUPac  = IMU.getDataPackets(TIMEOUT)
    encMap  = loco.read_enc_map(loco.read_OSL(kneSta, ankSta, IMUPac)) 

    # Initialized logger
    dataOSL = loco.read_OSL(kneSta, ankSta, IMUPac)
    cmd_log = {'refAnk': 0.0, 'refKnee': 0.0}
    ekf_log = {'phase': 0.0, 'phase_dot': 0.0, 'stride_length': 0.0, 'ramp': 0.0,
               'thigh_angle_pred': 0.0, 'thigh_angle_vel_pred': 0.0, 'atan2_pred': 0.0,
               'thigh_angle_vel': 0.0, 'atan2': 0.0,
               'walking': 0, 'MD_movingAverage': 0.0, 'steady_state': 0}
    logger = loco.ini_log({**dataOSL, **cmd_log, **ekf_log}, sensors = "all_sensors", trialName = "OSL_parallelBar_test")

    ## Initialize buffers for joints angles =============================================================================
    knee_angle_buffer = []   # in rad
    ankle_angle_buffer = []
    loadCell_Fz_buffer = []
    ankle_moment_buffer = []
    for i in range(3):
        dataOSL = loco.read_OSL(kneSta, ankSta, IMUPac, logger['ini_time'], encMap)
        knee_angle_buffer.append(dataOSL['kneJoiPos'])
        ankle_angle_buffer.append(dataOSL['ankJoiPos'])
        loadCell_Fz_buffer.append(dataOSL['loadCelFz'])
        ankle_moment_buffer.append(dataOSL['ankJoiTor'])
    print("Initial knee position (deg): %.2f, %.2f, %.2f " % (knee_angle_buffer[0]*180/np.pi, knee_angle_buffer[1]*180/np.pi, knee_angle_buffer[2]*180/np.pi))
    print("Initial ankle position (deg): %.2f, %.2f, %.2f " % (ankle_angle_buffer[0]*180/np.pi, ankle_angle_buffer[1]*180/np.pi, ankle_angle_buffer[2]*180/np.pi))
    print("Initial loadCell Fz (N): %.2f, %.2f, %.2f " % (loadCell_Fz_buffer[0], loadCell_Fz_buffer[1], loadCell_Fz_buffer[2]))
    print("Initial ankle torque (N-m): %.2f, %.2f, %.2f " % (ankle_moment_buffer[0], ankle_moment_buffer[1], ankle_moment_buffer[2]))

    knee_angle_initial = np.median(knee_angle_buffer) * 180/np.pi   # deg
    ankle_angle_initial = np.median(ankle_angle_buffer) * 180/np.pi
    dataOSL['loadCelFz'] = np.median(loadCell_Fz_buffer)
    dataOSL['ankJoiTor'] = np.median(ankle_moment_buffer)
    #===================================================================================================================

    ## Saturation for joint angles =====================================================================================
    # Knee angle limits (deg)
    knee_max = -2
    knee_min = -65
    
    # Ankle angle limits (deg)
    ankle_max = 20
    ankle_min = -10
    #===================================================================================================================

    ## Initialize EKF ==================================================================================================
    # Dictionary of the sensors
    sensors_dict = {'global_thigh_angle': 0, 'force_z_ankle': 1, 'force_x_ankle': 2,
                    'moment_y_ankle': 3, 'global_thigh_angle_vel': 4, 'atan2': 5}

    # Determine which sensors to be used
    sensors = ['global_thigh_angle', 'global_thigh_angle_vel', 'atan2']
    sensor_id = [sensors_dict[key] for key in sensors]

    arctan2 = False
    if sensors[-1] == 'atan2':
        arctan2 = True

    with open('R.pickle', 'rb') as file:
        R = pickle.load(file)

    m_model = model_loader('Measurement_model_' + str(len(sensors)) +'.pickle')
    Psi = np.array([load_Psi('Generic')[key] for key in sensors], dtype = object)
    saturation_range = [2, 0, 2, 0.8] 
    
    ## build the system
    sys = myStruct()
    sys.f = process_model
    sys.A = A
    sys.h = m_model
    sys.Q = np.diag([0, 1e-5, 1e-5, 0])
    # measurement noise covariance
    sys.R = R['Generic'][np.ix_(sensor_id, sensor_id)]
    U = np.diag([2, 2, 2])
    sys.R = U @ sys.R @ U.T

    # initialize the state
    init = myStruct()
    init.x = np.array([[0], [0.8], [1.1], [0]])
    init.Sigma = np.diag([1, 1, 1, 0])

    ekf = extended_kalman_filter(sys, init)
    #==================================================================================================================

    ### Create filters ================================================================================================
    fs = 74          # sampling rate = 100Hz (actual: dt ~ 0.0135 sec; 74Hz) 
    nyq = 0.5 * fs    # Nyquist frequency = fs/2
    # configure low-pass filter (1-order)
    normal_cutoff = 2 / nyq   #cut-off frequency = 2Hz
    b_lp, a_lp = butter(1, normal_cutoff, btype = 'low', analog = False)
    z_lp_1 = lfilter_zi(b_lp,  a_lp)
    z_lp_2 = lfilter_zi(b_lp,  a_lp)
    
    # configure band-pass filter (2-order)
    normal_lowcut = 0.2 / nyq    #lower cut-off frequency = 0.5Hz
    normal_highcut = 2 / nyq     #upper cut-off frequency = 2Hz
    b_bp, a_bp = butter(2, [normal_lowcut, normal_highcut], btype = 'band', analog = False)
    z_bp = lfilter_zi(b_bp,  a_bp)
    #==================================================================================================================

    if input('\n\nAbout to walk. Would you like to continue? (y/n): ').lower() == 'y':
        print("\n Let's walk!")
    else:
        sys.exit("User stopped the execution")

    ptr = 0
    t_0 = time.time()     # for EKF
    start_time = t_0      # for live plotting
    
    fade_in_time = 2      # sec
    
    ### Walking Status Detector =====================================================================================
    t_s = start_time
    t_ns = start_time
    t_s_previous = start_time
    t_ns_previous = start_time
    steady_state = False
    steady_state_previous = False
    walking = False

    MD_hist= deque([])
    global_thigh_angle_hist = np.ones((int(fs*2), 1)) * dataOSL["ThighSagi"] * 180 / np.pi # ~ 2seconds window

    #MD_threshold = 5 # MD
    global_thigh_angle_max_threshold = 10    # global thigh angle range (deg)
    global_thigh_angle_min_threshold = 5    # global thigh angle range (deg)

    # ================================================= MAIN LOOP ====================================================
    
    while True:
        ## Read OSL ==================================================================================================
        kneSta  = fxs.read_device(kneID)
        ankSta  = fxs.read_device(ankID)
        IMUPac  = IMU.getDataPackets(TIMEOUT)  
        dataOSL = loco.read_OSL(kneSta, ankSta, IMUPac, logger['ini_time'], encMap)
        #==========================================================================================================

        ## Calculate knee and ankle angles ========================================================================
        # 1) Without buffers
        knee_angle = dataOSL['kneJoiPos'] * 180 / np.pi
        ankle_angle = dataOSL['ankJoiPos'] * 180 / np.pi

        # 2) Use the buffers
        """
        knee_angle_buffer = [dataOSL['kneJoiPos'], knee_angle_buffer[0], knee_angle_buffer[1]]
        ankle_angle_buffer = [dataOSL['ankJoiPos'], ankle_angle_buffer[0], ankle_angle_buffer[1]]

        dataOSL['kneJoiPos'] = np.median(knee_angle_buffer) # rad
        dataOSL['ankJoiPos'] = np.median(ankle_angle_buffer)

        knee_angle = dataOSL['kneJoiPos'] * 180 / np.pi # deg
        ankle_angle = dataOSL['ankJoiPos'] * 180 / np.pi
        """
        #==========================================================================================================
        
        ## Calculate loadCell Fz using the buffer =================================================================
        if dataOSL['loadCelFz'] > 50:
            loadCell_Fz_buffer = [loadCell_Fz_buffer[0], loadCell_Fz_buffer[0], loadCell_Fz_buffer[1]]
        else:
            loadCell_Fz_buffer = [dataOSL['loadCelFz'], loadCell_Fz_buffer[0], loadCell_Fz_buffer[1]]
        dataOSL['loadCelFz'] =  np.median(loadCell_Fz_buffer)
        #==========================================================================================================

        ## Calculate ankle moment using the buffer =================================================================
        ankle_moment_buffer = [dataOSL['ankJoiTor'], ankle_moment_buffer[0], ankle_moment_buffer[1]]
        dataOSL['ankJoiTor'] = np.median(ankle_moment_buffer)
        #==========================================================================================================

        ## Time and dt ============================================================================================
        t = time.time()
        dt = t - t_0
        t_0 = t
        #==========================================================================================================

        ## Calculate Measurements =================================================================================
        # 1) Global thigh angle (deg)
        global_thigh_angle = dataOSL['ThighSagi'] * 180 / np.pi

        # Walking detector
        global_thigh_angle_hist = np.roll(global_thigh_angle_hist, -1)
        global_thigh_angle_hist[-1] = global_thigh_angle
        if (min(global_thigh_angle_hist) < global_thigh_angle_min_threshold
            and max(global_thigh_angle_hist) > global_thigh_angle_max_threshold):
            walking = True
        else:
            walking = False

        # 2) Global thigh angle velocity (deg/s)
        if ptr == 0:
            global_thigh_angle_vel_lp = 0 
        else:
            global_thigh_angle_vel = (global_thigh_angle - global_thigh_angle_0) / dt
            # low-pass filtering
            global_thigh_angle_vel_lp, z_lp_1 = lfilter(b_lp, a_lp, [global_thigh_angle_vel], zi = z_lp_1)
            global_thigh_angle_vel_lp = global_thigh_angle_vel_lp[0]

        global_thigh_angle_0 = global_thigh_angle

        # 3) Atan2
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
        
        if walking == False:
            Atan2 = 0

        measurement = np.array([[global_thigh_angle], [global_thigh_angle_vel_lp], [Atan2]])
        measurement = np.squeeze(measurement)
        #==========================================================================================================

        ## EKF implementation  ====================================================================================
        ekf.prediction(dt)
        ekf.state_saturation(saturation_range)
        ekf.correction(measurement, Psi, arctan2)
        ekf.state_saturation(saturation_range)
        #==========================================================================================================
        
        ## Steady-State Walking Detection =========================================================================
        if len(MD_hist) < int(fs * 2.5):
            MD_hist.append(ekf.MD)
        else:
            MD_hist.append(ekf.MD)
            MD_hist.popleft()
        MD_movingAverage = np.mean(MD_hist)
        
        """
        if MD_movingAverage < MD_threshold:
            t_s = t
            t_ns = t_ns_previous
        else:
            t_ns = t
            t_s = t_s_previous
        
        if t_s - t_ns > 3:
            steady_state = True
        elif t_s - t_ns < -1:
            steady_state = False
        
        steady_state = steady_state and walking
        
        t_s_previous = t_s
        t_ns_previous = t_ns
        
        if steady_state == True and steady_state_previous == False:
            ekf.Q = np.diag([0, 1e-3, 1e-3, 0]) * dt 
            ekf.Sigma = np.diag([1e-3, 1e-3, 1e-3, 0])

        elif steady_state == False and steady_state_previous == True:
            ekf.Q = np.diag([0, 1e-3, 0, 0]) * dt 
            ekf.Sigma = np.diag([1e-3, 1e-3, 0, 0])
            ekf.x[2, 0] = 1.1
            ekf.x[3, 0] = 0
        
        steady_state_previous = steady_state
        """
        #==========================================================================================================

        ## Generate joint control commands ========================================================================
        # 1) Control commands generated by the trainned model
        """
        #joint_angles = joints_control(ekf.x[0, 0], ekf.x[1, 0], ekf.x[2, 0], ekf.x[3, 0])
        #knee_angle_model = joint_angles[0]
        #ankle_angle_model = joint_angles[1]
        """
        
        # 2) Control commands generated by Edgar's trajectory (look-up table)
        pv = int(ekf.x[0, 0] * 998)  # phase variable conversion (scaling)
        knee_angle_traj = refKne[pv]
        ankle_angle_traj = refAnk[pv]

        # cmd: traj or model
        knee_angle_cmd = knee_angle_traj
        ankle_angle_cmd = ankle_angle_traj
        #===========================================================================================================

        ## Set joint control commands: fade-in effect & saturation =================================================
        # Fade-in effect
        elapsed_time = t - start_time
        if (elapsed_time < fade_in_time):
            alpha = elapsed_time / fade_in_time 
            ankle_angle_cmd = ankle_angle_cmd * alpha + ankle_angle_initial * (1 - alpha)
            knee_angle_cmd = knee_angle_cmd * alpha + knee_angle_initial * (1 - alpha)

        # Saturate ankle command
        if ankle_angle_cmd > ankle_max:
            ankle_angle_cmd = ankle_max
        elif ankle_angle_cmd < ankle_min:
            ankle_angle_cmd = ankle_min
        
        # Saturate knee command
        if knee_angle_cmd > knee_max: 
            knee_angle_cmd = knee_max
        elif knee_angle_cmd < knee_min:
            knee_angle_cmd = knee_min
        #===========================================================================================================
        
        ## Move the OSL ============================================================================================
        ankMotCou, kneMotCou = loco.joi2motTic(encMap, knee_angle_cmd, ankle_angle_cmd)
        fxs.send_motor_command(ankID, fxe.FX_IMPEDANCE, ankMotCou)
        fxs.send_motor_command(kneID, fxe.FX_IMPEDANCE, kneMotCou)
        #===========================================================================================================

        ## Logging data ============================================================================================
        # log joint control commands
        cmd_log['refAnk'] = ankle_angle_cmd
        cmd_log['refKnee'] = knee_angle_cmd
        # log ekf
        ekf_log['phase'] = ekf.x[0, 0]
        ekf_log['phase_dot'] = ekf.x[1, 0]
        ekf_log['stride_length'] = ekf.x[2, 0]
        ekf_log['ramp'] = ekf.x[3, 0]
        ekf_log['thigh_angle_pred'] = ekf.z_hat[0][0]
        ekf_log['thigh_angle_vel_pred'] = ekf.z_hat[1][0]
        ekf_log['atan2_pred'] = ekf.z_hat[2][0]
        ekf_log['thigh_angle_vel'] = global_thigh_angle_vel_lp
        ekf_log['atan2'] = Atan2
        ekf_log['walking'] = int(walking)
        ekf_log['steady_state'] = int(steady_state)
        ekf_log['MD_movingAverage'] = MD_movingAverage
        loco.log_OSL({**dataOSL,**cmd_log, **ekf_log}, logger)
        #==========================================================================================================

        ## Live plotting ==========================================================================================
        #elapsed_time = t - start_time
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
        print("knee angle cmd: ", knee_angle_cmd, "; ankle angle cmd: ", ankle_angle_cmd, "; walking (T/F): ", walking)
        #==========================================================================================================
        
        ptr += 1

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