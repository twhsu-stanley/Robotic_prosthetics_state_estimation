"""
Code to test drive the phase variable for level-ground walking 
"""
import sys, time, os
from flexsea import flexsea as flex
from flexsea import fxEnums as fxe
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
from basis_model_fitting import measurement_noise_covariance
#from collections import deque
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
fxs.start_streaming(kneID, freq = 500, log_en = False) # 500 or 100 Hz
fxs.start_streaming(ankID, freq = 500, log_en = False) # 500 or 100 Hz
time.sleep(1)                           # Healthy pause before using ActPacks or IMU

# Soft start
G_K = {"kp": 40, "ki": 400, "K": 10, "B": 0, "FF": 128}  # Knee controller gains
G_A = {"kp": 40, "ki": 400, "K": 10, "B": 0, "FF": 128}  # Ankle controller gains
fxs.set_gains(ankID, G_A["kp"], G_A["ki"], 0, G_A["K"], G_A["B"], G_A["FF"])
fxs.set_gains(kneID, G_K["kp"], G_K["ki"], 0, G_K["K"], G_K["B"], G_K["FF"])
fxs.send_motor_command(ankID, fxe.FX_IMPEDANCE, fxs.read_device(ankID).mot_ang)
fxs.send_motor_command(kneID, fxe.FX_IMPEDANCE, fxs.read_device(kneID).mot_ang)
time.sleep(0.1)

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

    # Create encoder map
    kneSta  = fxs.read_device(kneID)
    ankSta  = fxs.read_device(ankID)
    IMUPac  = IMU.getDataPackets(TIMEOUT)
    encMap  = loco.read_enc_map(loco.read_OSL(kneSta, ankSta, IMUPac)) 

    # Initialized logger
    dataOSL = loco.read_OSL(kneSta, ankSta, IMUPac)
    cmd_log = {'refAnk': 0.0, 'refKnee': 0.0}
    ekf_log = {'phase': 0.0, 'phase_dot': 0.0, 'stride_length': 0.0, 'ramp': 0.0,
               'global_thigh_angle_pred': 0.0, 'global_thigh_vel_pred': 0.0, 'atan2_pred': 0.0,
               'global_thigh_vel_lp': 0.0, 'global_thigh_angle_lp': 0.0, 'global_thigh_vel_lp_2': 0.0, 
               'global_thigh_angle_max': 0.0, 'global_thigh_angle_min': 0.0,
               'global_thigh_vel_max': 0.0, 'global_thigh_vel_min': 0.0,
               'phase_x': 0.0, 'phase_y': 0.0, 'radius': 0.0, 'atan2': 0.0}
    logger = loco.ini_log({**dataOSL, **cmd_log, **ekf_log}, sensors = "all_sensors", trialName = "OSL_benchtop_swing_test")

    ## Initialize buffers for joints angles =============================================================================
    knee_angle_buffer = []   # in rad
    ankle_angle_buffer = []
    #loadCell_Fz_buffer = []
    #ankle_moment_buffer = []
    for i in range(3):
        dataOSL = loco.read_OSL(kneSta, ankSta, IMUPac, logger['ini_time'], encMap)
        knee_angle_buffer.append(dataOSL['kneJoiPos'])
        ankle_angle_buffer.append(dataOSL['ankJoiPos'])
        #loadCell_Fz_buffer.append(dataOSL['loadCelFz'])
        #ankle_moment_buffer.append(dataOSL['ankJoiTor'])
    print("Initial knee position (deg): %.2f, %.2f, %.2f " % (knee_angle_buffer[0]*180/np.pi, knee_angle_buffer[1]*180/np.pi, knee_angle_buffer[2]*180/np.pi))
    print("Initial ankle position (deg): %.2f, %.2f, %.2f " % (ankle_angle_buffer[0]*180/np.pi, ankle_angle_buffer[1]*180/np.pi, ankle_angle_buffer[2]*180/np.pi))
    #print("Initial loadCell Fz (N): %.2f, %.2f, %.2f " % (loadCell_Fz_buffer[0], loadCell_Fz_buffer[1], loadCell_Fz_buffer[2]))
    #print("Initial ankle torque (N-m): %.2f, %.2f, %.2f " % (ankle_moment_buffer[0], ankle_moment_buffer[1], ankle_moment_buffer[2]))

    knee_angle_initial = np.median(knee_angle_buffer) * 180/np.pi   # deg
    ankle_angle_initial = np.median(ankle_angle_buffer) * 180/np.pi
    #dataOSL['loadCelFz'] = np.median(loadCell_Fz_buffer)
    #dataOSL['ankJoiTor'] = np.median(ankle_moment_buffer)
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
    sensors_dict = {'globalThighAngles':0, 'globalThighVelocities':1, 'atan2':2,
                    'globalFootAngles':3, 'ankleMoment':4, 'tibiaForce':5}

    # Specify which sensors to be used
    sensors = ['globalThighAngles', 'globalThighVelocities', 'atan2']
    using_atan2 = np.any(np.array(sensors) == 'atan2')

    sensor_id = [sensors_dict[key] for key in sensors]
    sensor_id_str = ""
    for i in range(len(sensor_id)):
        sensor_id_str += str(sensor_id[i])
    m_model = model_loader('Measurement_model_' + sensor_id_str +'_NSL.pickle')
    
    Psi = np.array([load_Psi('Generic')[key] for key in sensors], dtype = object)
    
    saturation_range = np.array([1.3, 0, 2, 0])

    ## build the system
    sys = myStruct()
    sys.f = process_model
    sys.A = A
    sys.h = m_model
    sys.Q = np.diag([0, 5e-3, 5e-3, 0])
    # measurement noise covariance
    U = np.diag([1, 1, 1])
    R = U @ measurement_noise_covariance(*sensors) @ U.T
    R_org = np.copy(R)
    sys.R = np.copy(R)

    # initialize the state
    init = myStruct()
    init.x = np.array([[0.3], [0], [0], [0]])
    init.Sigma = np.diag([1, 1, 1, 0])

    ekf = extended_kalman_filter(sys, init)
    #==================================================================================================================

    ### Create filters ================================================================================================
    fs = 80         # sampling rate = 100Hz (actual: dt ~ 0.0135 sec; 67Hz) 
    nyq = 0.5 * fs    # Nyquist frequency = fs/2
    normal_cutoff = 2 / nyq   #cut-off frequency = 2Hz
    # Configure 1st order low-pass filters for computing velocity 
    b_lp_1, a_lp_1 = butter(1, normal_cutoff, btype = 'low', analog = False)
    z_lp_1 = lfilter_zi(b_lp_1,  a_lp_1)
    
    # Configure 1st/2nd/3rd order low-pass filters for computing atan2
    b_lp_2, a_lp_2 = butter(1, normal_cutoff, btype = 'low', analog = False)
    z_lp_2 = lfilter_zi(b_lp_2,  a_lp_2)
    #==================================================================================================================

    if input('\n\nAbout to walk. Would you like to continue? (y/n): ').lower() == 'y':
        print("\n Let's walk!")
    else:
        sys.exit("User stopped the execution")

    ptr = 0
    t_0 = time.time()     # for EKF
    start_time = t_0      # for live plotting
    
    fade_in_time = 2      # sec

    idx_min_prev = 0
    idx_min = 0
    idx_max_prev = 0
    idx_max = 0
    global_thigh_angle_window = np.zeros(500) # time window/ pre-allocate 1 sec
    global_thigh_vel_window = np.zeros(500)

    # ================================================= MAIN LOOP ===================================================
    while True:
        ## Read OSL =================================================================================================
        kneSta  = fxs.read_device(kneID)
        ankSta  = fxs.read_device(ankID)
        IMUPac  = IMU.getDataPackets(TIMEOUT)  
        dataOSL = loco.read_OSL(kneSta, ankSta, IMUPac, logger['ini_time'], encMap)
        #==========================================================================================================

        ## Calculate knee and ankle angles ========================================================================
        # 1) Without buffers
        #knee_angle = dataOSL['kneJoiPos'] * 180 / np.pi
        #ankle_angle = dataOSL['ankJoiPos'] * 180 / np.pi

        # 2) Use the buffers
        knee_angle_buffer = [dataOSL['kneJoiPos'], knee_angle_buffer[0], knee_angle_buffer[1]]
        ankle_angle_buffer = [dataOSL['ankJoiPos'], ankle_angle_buffer[0], ankle_angle_buffer[1]]

        dataOSL['kneJoiPos'] = np.median(knee_angle_buffer) # rad
        dataOSL['ankJoiPos'] = np.median(ankle_angle_buffer)

        knee_angle = dataOSL['kneJoiPos'] * 180 / np.pi # deg
        ankle_angle = dataOSL['ankJoiPos'] * 180 / np.pi
        #==========================================================================================================
        
        ## Calculate loadCell Fz using the buffer =================================================================
        """
        if dataOSL['loadCelFz'] > 50:
            loadCell_Fz_buffer = [loadCell_Fz_buffer[0], loadCell_Fz_buffer[0], loadCell_Fz_buffer[1]]
        else:
            loadCell_Fz_buffer = [dataOSL['loadCelFz'], loadCell_Fz_buffer[0], loadCell_Fz_buffer[1]]
        dataOSL['loadCelFz'] =  np.median(loadCell_Fz_buffer)
        """
        #==========================================================================================================

        ## Calculate ankle moment using the buffer =================================================================
        """
        ankle_moment_buffer = [dataOSL['ankJoiTor'], ankle_moment_buffer[0], ankle_moment_buffer[1]]
        dataOSL['ankJoiTor'] = np.median(ankle_moment_buffer)
        """
        #==========================================================================================================

        ## Time and dt ============================================================================================
        t = time.time()
        dt = t - t_0
        t_0 = t
        #==========================================================================================================

        ## Calculate Measurements =================================================================================
        # 1) Global thigh angle (deg)
        global_thigh_angle = dataOSL['ThighSagi'] * 180 / np.pi

        # 2) Global thigh angle velocity (deg/s)
        if ptr == 0:
            global_thigh_vel_lp = 0 
        else:
            global_thigh_vel = (global_thigh_angle - global_thigh_angle_0) / dt
            # low-pass filtering
            global_thigh_vel_lp, z_lp_1 = lfilter(b_lp_1, a_lp_1, [global_thigh_vel], zi = z_lp_1)
            global_thigh_vel_lp = global_thigh_vel_lp[0]
        global_thigh_angle_0 = global_thigh_angle

        ## 3) Compute Atan2
        global_thigh_angle_lp, z_lp_2 = lfilter(b_lp_2, a_lp_2, [global_thigh_angle], zi = z_lp_2) 
        global_thigh_angle_lp = global_thigh_angle_lp[0]
        if ptr == 0:
            global_thigh_vel_lp_2 = 0
        else:
            global_thigh_vel_lp_2 = (global_thigh_angle_lp - global_thigh_angle_lp_0) / dt
        global_thigh_angle_lp_0 = global_thigh_angle_lp

        # allocte more space if ptr exceed the bounds
        if ptr - idx_max_prev >= np.shape(global_thigh_angle_window)[0]:
            global_thigh_angle_window = np.concatenate((global_thigh_angle_window, np.zeros(np.shape(global_thigh_angle_window)[0])))
            global_thigh_vel_window = np.concatenate((global_thigh_vel_window, np.zeros(np.shape(global_thigh_angle_window)[0])))

        global_thigh_angle_window[ptr - idx_max_prev] = global_thigh_angle_lp
        global_thigh_vel_window[ptr - idx_max_prev] = global_thigh_vel_lp_2
       
        if ptr > 0:
            global_thigh_angle_max = np.max(global_thigh_angle_window[idx_min_prev - idx_max_prev:ptr - idx_max_prev])
            global_thigh_angle_min = np.min(global_thigh_angle_window[idx_max_prev - idx_max_prev:ptr - idx_max_prev])
            global_thigh_vel_max = np.max(global_thigh_vel_window[idx_min_prev - idx_max_prev:ptr - idx_max_prev])
            global_thigh_vel_min = np.min(global_thigh_vel_window[idx_max_prev - idx_max_prev:ptr - idx_max_prev])
        
        # compute the indices
        if ptr > idx_max + 1:
            idx_min_temp = np.argmin(global_thigh_angle_window[idx_max - idx_max_prev:ptr - idx_max_prev]) + idx_max
            if ptr > idx_min_temp + 1 and global_thigh_angle_window[idx_min_temp - idx_max_prev] < -5:
                idx_min = idx_min_temp
                idx_max_temp = np.argmax(global_thigh_angle_window[idx_min_temp - idx_max_prev:ptr - idx_max_prev]) + idx_min_temp
                if ptr > idx_max_temp + 1 and global_thigh_angle_window[idx_max_temp - idx_max_prev] > 5: # new stride
                    # reform the time windows
                    global_thigh_angle_window = global_thigh_angle_window[idx_max - idx_max_prev:-1]
                    global_thigh_vel_window = global_thigh_vel_window[idx_max - idx_max_prev:-1]

                    # swap indices
                    idx_max_prev = idx_max
                    idx_max = idx_max_temp
                    idx_min_prev = idx_min
        
        # compute the scaled and shifted Atan2
        if idx_max_prev > 0 and idx_min_prev > 0:
            global_thigh_angle_shift = (global_thigh_angle_max + global_thigh_angle_min) / 2
            global_thigh_angle_scale = abs(global_thigh_vel_max - global_thigh_vel_min) / abs(global_thigh_angle_max - global_thigh_angle_min)
            global_thigh_vel_shift = (global_thigh_vel_max + global_thigh_vel_min) / 2

            phase_y = - (global_thigh_vel_lp_2 - global_thigh_vel_shift)
            phase_x = global_thigh_angle_scale * (global_thigh_angle_lp - global_thigh_angle_shift)
        else:
            phase_y = - global_thigh_vel_lp_2 #/ (2*np.pi*0.8)
            phase_x = global_thigh_angle_lp * (2*np.pi*0.8)
        
        Atan2 = np.arctan2(phase_y, phase_x)
        if Atan2 < 0:
            Atan2 = Atan2 + 2 * np.pi
        
        c = 45
        d = 45
        radius = (phase_x / c) ** 2 + (phase_y / d) ** 2
        if radius >= 1:
            ekf.R = np.copy(R_org)
        else:
            ekf.R[2, 2] = 1e30

        measurement = np.array([[global_thigh_angle], [global_thigh_vel_lp], [Atan2]])
        measurement = np.squeeze(measurement)
        #==========================================================================================================

        ## EKF implementation  ====================================================================================
        ekf.Q = np.diag([0, 5e-3, 5e-3, 0]) * dt 
        ekf.prediction(dt)
        ekf.state_saturation(saturation_range)
        ekf.correction(measurement, Psi, using_atan2)
        ekf.state_saturation(saturation_range)
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
        if elapsed_time < fade_in_time:
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
        ekf_log['global_thigh_angle_pred'] = ekf.z_hat[0][0]
        ekf_log['global_thigh_vel_pred'] = ekf.z_hat[1][0]
        ekf_log['atan2_pred'] = ekf.z_hat[2][0]
        
        ekf_log["global_thigh_vel_lp"] = global_thigh_vel_lp
        ekf_log["global_thigh_angle_lp"] = global_thigh_angle_lp
        ekf_log["global_thigh_vel_lp_2"] = global_thigh_vel_lp_2
        if ptr > 0:
            ekf_log["global_thigh_angle_max"] = global_thigh_angle_max
            ekf_log["global_thigh_angle_min"] = global_thigh_angle_min
            ekf_log["global_thigh_vel_max"] = global_thigh_vel_max
            ekf_log["global_thigh_vel_min"] = global_thigh_vel_min
        ekf_log["phase_x"] = phase_x
        ekf_log["phase_y"] = phase_y
        ekf_log['atan2'] = Atan2
        ekf_log['radius'] = radius
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
                         global_thigh_vel_lp, ekf.z_hat[1], 'Global Thigh Angle Vel', 'deg/s',
                         Atan2, ekf.z_hat[2], 'Atan2', '--',
                         #radius, radius, 'phase plane radius', '--',
                         #knee_angle, knee_angle_cmd, 'Knee Angle', 'deg',
                         #ankle_angle, ankle_angle_cmd, 'Ankle Angle', 'deg'
                         )
        print("knee angle cmd: ", knee_angle_cmd, "; ankle angle cmd: ", ankle_angle_cmd)
        #==========================================================================================================
        
        ptr += 1

except KeyboardInterrupt:
    print('\n*** OSL shutting down ***\n')

finally:        
    # Do anything but turn off the motors at the end of the program
    fxs.send_motor_command(ankID, fxe.FX_NONE, 0)
    fxs.send_motor_command(kneID, fxe.FX_NONE, 0)
    IMU.setToIdle()
    time.sleep(1)    
    fxs.close(ankID)
    fxs.close(kneID)  
    print('Communication with ActPacks closed and IMU set to idle')