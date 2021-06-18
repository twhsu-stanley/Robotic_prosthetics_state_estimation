"""
Code to test drive the phase variable for level-ground walking 
"""
import sys, time, os
from flexsea import flexsea as flex
from flexsea import fxEnums as fxe
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(r'/home/pi/OSL-master/locoAPI/') # Path to Loco module
#sys.path.append(r'/home/pi/.local/bin')
sys.path.append(r'/usr/share/python3-mscl/')     # Path of the MSCL - API for the IMU
import locoOSL as loco                           # Module from Locolab
import mscl as msl                               # Module from Microstrain
import sender                                    # for real-time plotting

sys.path.append(r'/home/pi/prosthetic_phase_estimation/')
from EKF import *
from model_framework import *
from data_generators import *
from continuous_data import *
from model_fit import *
from scipy.signal import butter, lfilter, lfilter_zi

# Process model for the EKF
def A(dt):
    return np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

def process_model(x, dt):
    #dt = 0.01 # data sampling rate: 100 Hz
    return A(dt) @ x

# -----------------TODO: change these constants to match your setup ----------------------
ANK_PORT = r'/dev/ttyACM1'
KNE_PORT = r'/dev/ttyACM0'
IMU_PORT = r'/dev/ttyUSB0'
TIMEOUT  = 500                          #Timeout in (ms) to read the IMU
fxs = flex.FlexSEA()
# ------------------ INITIALIZATION ------------------------------------------------------
# Connect with knee and ankle actuator packs
kneID = fxs.open(port = KNE_PORT, baud_rate = 230400, log_level = 0)
ankID = fxs.open(port = ANK_PORT, baud_rate = 230400, log_level = 0)

# Initialize IMU - The sample rate (SR) of the IMU controls the SR of the OSL 
connection = msl.Connection.Serial(IMU_PORT, 921600)
IMU = msl.InertialNode(connection)
IMU.setToIdle()
packets = IMU.getDataPackets(TIMEOUT)   # Clean the internal circular buffer.

# Set streaming        
IMU.resume()
fxs.start_streaming(kneID, freq = 500, log_en = False)
fxs.start_streaming(ankID, freq = 500, log_en = False)
time.sleep(1)                           # Healthy pause before using ActPacks or IMU

# ------------------ MAIN LOOP -----------------------------------------------------------
try:
    # For gain details check https://dephy.com/wiki/flexsea/doku.php?id=controlgains
    # G_K = {"kp": 40, "ki": 400, "K": 600, "B": 300, "FF": 128}  # Knee controller gains
    G_K = {"kp": 40, "ki": 400, "K": 300, "B": 1600, "FF": 128}  # Knee controller gains
    # G_A = {"kp": 40, "ki": 400, "K": 600, "B": 300, "FF": 128}  # Ankle controller gains
    G_A = {"kp": 40, "ki": 400, "K": 300, "B": 1600, "FF": 128}  # Ankle controller gains

    kneSta  = fxs.read_device(kneID)
    ankSta  = fxs.read_device(ankID)
    IMUPac  = IMU.getDataPackets(TIMEOUT)
    
    fxs.set_gains(ankID, G_A["kp"], G_A["ki"], 0, G_A["K"], G_A["B"], G_A["FF"])
    fxs.set_gains(kneID, G_K["kp"], G_K["ki"], 0, G_K["K"], G_K["B"], G_K["FF"])

    # load trajectory and encoder map
    tra = loco.loadTrajectory(trajectory = 'walking')
    refAnk = tra["ankl"]
    refKne = tra["knee"]
    refHip = tra["hip_"]
    refPha = tra["phas"]
    # Create encoder map
    kneSta  = fxs.read_device(kneID)
    ankSta  = fxs.read_device(ankID)
    IMUPac  = IMU.getDataPackets(TIMEOUT)
    encMap  = loco.read_enc_map( loco.read_OSL(kneSta, ankSta, IMUPac) ) 

    dataOSL = loco.read_OSL(kneSta, ankSta, IMUPac)
    misclog = {'PV': [0.0, 'phase'], 'refAnk': [0.0,'deg'], 'refKnee': [0.0,'deg']}
    logger = loco.ini_log({**dataOSL, **misclog}, sensors="all_sensors", trialName="PV_TwoStates_1000Hz")

    ### Intitialize EKF
    sensors = [0, 6, 7] # [012456] w/ Q=[0, 3e-5, 1e-5, 1e-1] looks good
    arctan2 = False
    if sensors[-1] == 7:
        arctan2 = True
    with open('R_s.pickle', 'rb') as file:
        R = pickle.load(file)

    m_model = model_loader('Measurement_model_' + str(len(sensors)) +'_sp.pickle')
    Psi = load_Psi()[sensors]
    saturation_range = Conti_maxmin(subject, plot = False)

    ## build the system
    sys = myStruct()
    sys.f = process_model
    sys.A = A
    sys.h = m_model
    sys.Q = np.diag([0, 1e-5, 1e-5, 1e-1]) #[0, 6e-5, 1e-6, 1e-1] #process model noise covariance [0, 3e-5, 1e-5, 1e-1]=70%
    # measurement noise covariance
    sys.R = R[subject][np.ix_(sensors, sensors)]
    U = np.diag([2, 2, 2])
    sys.R = U @ sys.R @ U.T

    # initialize the state
    init = myStruct()
    init.x = np.array([[0], [0.8], [1.1], [0]])
    init.Sigma = np.diag([10, 10, 10, 100])

    ekf = extended_kalman_filter(sys, init)

    ### Create filters
    # configure low-pass filter (1-order)
    nyq = 0.5 * 100
    normal_cutoff = 2 / nyq
    order = 1
    b_lp, a_lp = butter(1, normal_cutoff, btype='low', analog=False)
    z_lp = lfilter_zi(b_lp,  a_lp)
    # configure band-pass filter (2-order)
    nyq = 0.5 * 100
    normal_lowcut = 0.5 / nyq
    normal_highcut = 2 / nyq
    b_bp, a_bp = butter(2, [normal_lowcut, normal_highcut], btype='band', analog=False)
    z_bp = lfilter_zi(b_bp,  a_bp)

    if input('\n\nAbout to walk. Would you like to continue? (y/n): ').lower() == 'y':
        print("\n Let's walk!")
    else:
        sys.exit("User stopped the execution")

    ptr = 0
    t_0 = time.time()
    start_time = time.time()
    while True:
        # Read OSL
        kneSta  = fxs.read_device(kneID)
        ankSta  = fxs.read_device(ankID)
        IMUPac  = IMU.getDataPackets(TIMEOUT)  
        dataOSL = loco.read_OSL(kneSta, ankSta, IMUPac, logger['ini_time'], encMap)

        ### measurement data
        ## global thigh angle
        global_thigh_angle = -dataOSL['ThighSagi'][0] * 180 / np.pi
        
        # time
        t = time.time()
        dt = t - t_0 
        t_0 = t

        ## Compute global thigh angle velocity
        if ptr == 0:
            global_thigh_angle_vel = 0 
        else:
            global_thigh_angle_vel = (global_thigh_angle - global_thigh_angle_0) / dt
            # low-pass filtering
            global_thigh_angle_vel_lp, z_lp = lfilter(b_lp, a_lp, global_thigh_angle_vel, zi = z_lp) # low-pass filtering

        global_thigh_angle_0 = global_thigh_angle

        ## Compute atan2
        # band-pass filtering
        global_thigh_angle_bp, z_bp = lfilter(b_bp, a_bp, global_thigh_angle, zi = z_bp) 
        Atan2 = np.arctan2(-global_thigh_angle_vel_lp / (2*np.pi*0.8), global_thigh_angle_bp)
        if Atan2 < 0:
            Atan2 = Atan2 + 2 * np.pi

        measurement = np.array([[global_thigh_angle], [global_thigh_angle_vel_lp], [Atan2]])

        ### EKF implementation
        ekf.prediction(dt)
        ekf.state_saturation(saturation_range)
        ekf.correction(measurement, Psi, arctan2)
        ekf.state_saturation(saturation_range)

        ### Control commands: joints angles
        joint_angles = joints_control(ekf.x[0], ekf.x[1], ekf.x[2], ekf.x[3])
        knee_angle_cmd = joint_angles[0]
        ankle_angle_cmd = joint_angles[1]

        # Estimate percentage within the gait cycle - Output from 0 to 998
        #pv = int(loco.getPhaseVariable_vTwoStates(dataOSL, FCThr = -5)*998) 
        misclog['PV'][0] = ekf.x[0] # gait phase
        misclog['refAnk'][0] = ankle_angle_cmd
        misclog['refKnee'][0] = knee_angle_cmd
        loco.log_OSL({**dataOSL,**misclog}, logger)
        
        ### Move the OSL
        #ankMotCou, kneMotCou = loco.joi2motTic(encMap, knee_angle_cmd, ankle_angle_cmd)
        #fxs.send_motor_command(ankID, fxe.FX_IMPEDANCE, ankMotCou)
        #fxs.send_motor_command(kneID, fxe.FX_IMPEDANCE, kneMotCou)


        elapsed_time = time.time() - start_time
        
        if ptr%10 == 0:
            sender.graph(elapsed_time, global_thigh_angle, 'Global Thigh Angle', 'deg',
                         ekf.z_hat[0], 'Global Thigh Angle Pred', 'deg',
                         global_thigh_angle_vel_lp, 'Global Thigh Angle Vel', 'deg/s',
                         ekf.z_hat[1], 'Global Thigh Angle Vel Pred', '-')
        
        print('Elapsed time:', elapsed_time, ptr)
        ptr+=1

        print("Thigh: {:>10.4f} [deg] || LoadCellFz: {:>10.4f} [N] || Ph. Va.: {:10.4f} "
            "|| Ref. Ankle: {:10.4f} [deg] || Ref. Knee: {:10.4f} [deg]".format(
                dataOSL['ThighSagi'][0]*180/np.pi, dataOSL['loadCelFz'][0], ekf.x[0], 
                ankle_angle_cmd, knee_angle_cmd) )
except KeyboardInterrupt:
        print('\n*** OSL shutting down ***\n')
finally:        
    # Do anything but turn off the motors at the end of the program
    fxs.send_motor_command(ankID, fxe.FX_NONE, 0)
    fxs.send_motor_command(kneID, fxe.FX_NONE, 0)
    IMU.setToIdle()
    fxs.close(ankID)
    fxs.close(kneID)  
    print('Communication with ActPacks closed and IMU set to idle')