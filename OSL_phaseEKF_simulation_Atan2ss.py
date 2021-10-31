"""
Code to simulate the phase-EKF and control commands using pre-revorded walking data
"""
import sys, time
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import csv

from EKF import *
from model_framework import *
from scipy.signal import butter, lfilter, lfilter_zi
from incline_experiment_utils import butter_lowpass_filter
from basis_model_fitting import measurement_noise_covariance
import sender_test as sender   # for real-time plotting

### A. Load Ross's pre-recorded walking data / EKF Tests 
"""
logFile = r"OSL_walking_data/210617_113644_PV_Siavash_walk_oscillations in phase.csv"
# 1) 210617_113644_PV_Siavash_walk_oscillations in phase
# 2) 210617_121732_PV_Siavash_walk_300_1600
# 3) 210617_122334_PV_Siavash_walk_500_2500

# 5) 210726_102901_OSL_parallelBar_test
# 6) 210730_140347_OSL_parallelBar_test
datatxt = np.genfromtxt(logFile , delimiter=',', names = True)
dataOSL = {
    "Time": datatxt["Time"],
    "ThighSagi": datatxt["ThighSagi"],
    #"PV": datatxt['PV'],
    'AnkleAngle': datatxt["ankJoiPos"],
    'AnkleAngleRef': datatxt["refAnk"],
    'KneeAngle': datatxt["kneJoiPos"],
    'KneeAngleRef': datatxt["refKnee"],
    
    # Kinetics data
    'AnkleTorque': datatxt["ankJoiTor"],
    #'AnkleMotorTorque': datatxt["ankMotTor"],
    'KneeTorque': datatxt["kneJoiTor"],
    'LoadCellFx':  datatxt['loadCelFx'],
    'LoadCellFy':  datatxt['loadCelFy'],
    'LoadCellFz':  datatxt['loadCelFz'],
    'LoadCellMx':  datatxt['loadCelMx'],
    'LoadCellMy':  datatxt['loadCelMy'],
    'LoadCellMz':  datatxt['loadCelMz']
}
"""

### B. Load Kevin's bypass-adapter walking data
"""
mat = scipy.io.loadmat('OSL_walking_data/Treadmill_speed1_incline0_file2.mat')
# Treadmill_speed1_incline0_file2
# Treadmill_speed1_incline0_file1
# Treadmill_speed1_incline5_file1
# Treadmill_speed1_incline5_file2
# Treadmill_speed1_incline10_file1
# Treadmill_speed1_inclineneg5_file1
# Treadmill_speed1_inclineneg10_file1

dataOSL = {
    "Time": np.cumsum(mat['ControllerOutputs'][0, 0]['dt']).reshape(-1),
    "ThighSagi": mat['ThighIMU'][0, 0]['ThetaX'].reshape(-1) / (180 / np.pi),
    "PV": mat['ControllerOutputs'][0, 0]['phaseEstimate'].reshape(-1) * 998,
    'AnkleAngle': -mat['AnkleEncoder'][0, 0]['FilteredJointAngle'].reshape(-1),
    'AnkleAngleRef': -mat['ControllerOutputs'][0, 0]['ankle_des'].reshape(-1),
    'KneeAngle': -mat['KneeEncoder'][0, 0]['FilteredJointAngle'].reshape(-1),
    'KneeAngleRef': -mat['ControllerOutputs'][0, 0]['knee_des'].reshape(-1),
    
    #'AnkleTorque': -mat['ControllerOutputs'][0, 0]['AnkleStanceTorque'].reshape(-1),
    #'AnkleTorqueSwing': -mat['ControllerOutputs'][0, 0]['AnkleSwingTorque'].reshape(-1),
    'AnkleTorque': -mat['ControllerOutputs'][0, 0]['AnkleTorqueCommand'].reshape(-1),
    'KneeTorque': -mat['ControllerOutputs'][0, 0]['KneeStanceTorque'].reshape(-1),
    'LoadCellFx': mat['LoadCell'][0, 0]['Fx'].reshape(-1),
    'LoadCellFy': mat['LoadCell'][0, 0]['Fy'].reshape(-1),
    'LoadCellFz': mat['LoadCell'][0, 0]['Fz'].reshape(-1),
    'LoadCellMx': mat['LoadCell'][0, 0]['Mx'].reshape(-1),
    'LoadCellMy': mat['LoadCell'][0, 0]['My'].reshape(-1),
    'LoadCellMz': mat['LoadCell'][0, 0]['Mz'].reshape(-1),
}
"""

### C. Load Benchtop Test Data
#"""
logFile = r"OSL_walking_data/211021_184343_OSL_benchtop_swing_test.csv"
# 211021_184718_OSL_benchtop_swing_test
# 211021_184343_OSL_benchtop_swing_test
# 211014_130906_OSL_benchtop_swing_test
# 211014_130313_OSL_benchtop_swing_test
# 211014_124556_OSL_benchtop_swing_test
# 210714_113523_OSL_benchtop_test
datatxt = np.genfromtxt(logFile , delimiter=',', names = True)
dataOSL = {
    "Time": datatxt["Time"],
    "ThighSagi": datatxt["ThighSagi"],
    'AnkleAngle': datatxt["ankJoiPos"],
    'AnkleAngleRef': datatxt["refAnk"],
    'KneeAngle': datatxt["kneJoiPos"],
    'KneeAngleRef': datatxt["refKnee"]
}
#"""

## From loco_OSL.py: Load referenced trajectories
def loadTrajectory(trajectory = 'walking'):
    # Create path to the reference csv trajectory
    if trajectory.lower() == 'walking':
        # walking data uses convention from D. A. Winter, “Biomechanical Motor Patterns in Normal Walking,”  
        # J. Mot. Behav., vol. 15, no. 4, pp. 302–330, Dec. 1983.
        pathFile = r'OSL_walking_data/walkingWinter_deg.csv'
        # Gains to scale angles to OSL convention
        ankGain = -1
        ankOffs = -0.15 # [deg] Small offset to take into accoun that ankle ROM is -10 deg< ankle < 19.65 deg
        kneGain = -1
        kneOffs = 0
        hipGain = 1
        hipOffs = 0
    else:
        raise ValueError('Please select a suported trajectory type')
    # Extract content from csv
    with open(pathFile, 'r') as f:
        datasetReader = csv.reader(f, quoting = csv.QUOTE_NONNUMERIC)
        data = np.transpose( np.array([row for row in datasetReader ]) )
    # Parse data to knee-ankle trajectories using OSL angle convention (+ ankle = plantarflexion. + knee = flexion)
    trajectory = dict(ankl = ankGain*data[0] + ankOffs)
    trajectory["ankd"] = ankGain*data[1]
    trajectory["andd"] = ankGain*data[2]
    trajectory["knee"] = kneGain*data[3] + kneOffs
    trajectory["kned"] = kneGain*data[4]
    trajectory["kndd"] = kneGain*data[5]
    trajectory["hip_"] = hipGain*data[6] + hipOffs
    trajectory["hipd"] = hipGain*data[7]
    trajectory["hidd"] = hipGain*data[8]
    trajectory["phas"] = data[9]
    trajectory["time"] = data[10]

    return trajectory

try:
    ### Load reference trajectory
    refTrajectory = loadTrajectory(trajectory = 'walking')
    refAnk = refTrajectory["ankl"]
    refKne = refTrajectory["knee"]

    ### Intitialize EKF
    # Dictionary of the sensors
    sensors_dict = {'globalThighAngles':0, 'globalThighVelocities':1, 'atan2':2,
                    'globalFootAngles':3, 'ankleMoment':4, 'tibiaForce':5}

    # Determine which sensors to be used
    # 1) measurements that use the basis model
    sensors = ['globalThighAngles', 'globalThighVelocities', 'atan2']

    sensor_id = [sensors_dict[key] for key in sensors]
    sensor_id_str = ""
    for i in range(len(sensor_id)):
        sensor_id_str += str(sensor_id[i])
    m_model = model_loader('Measurement_model_' + sensor_id_str +'_NSL.pickle')

    using_atan2 = np.any(np.array(sensors) == 'atan2')

    Psi = np.array([load_Psi('Generic')[key] for key in sensors], dtype = object)
    
    #saturation_range = saturation_bounds()
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
    init.x = np.array([[0.3], [0], [0], [0]]) # mid-stance
    init.Sigma = np.diag([1, 1, 1, 0])

    ekf = extended_kalman_filter(sys, init)

    ########## Create filters ################################################################
    fs = 1 / np.average(np.diff(dataOSL["Time"]))        # sampling rate = 100 Hz (actual: ~77 Hz)
    print("Average fs = %4.2f Hz" % fs)
    
    fc = 2 #cut-off frequency = 2Hz
    """
    nyq = 0.5 * fs    # Nyquist frequency = fs/2
    fc_normal = fc / nyq   
    # Configure 1st order low-pass filters for computing velocity 
    b_lp_1, a_lp_1 = butter(1, fc_normal, btype = 'low', analog = False)
    z_lp_1 = lfilter_zi(b_lp_1,  a_lp_1)
    
    # Configure 1st/2nd/3rd order low-pass filters for computing atan2
    b_lp_2, a_lp_2 = butter(1, fc_normal, btype = 'low', analog = False)
    z_lp_2 = lfilter_zi(b_lp_2,  a_lp_2)
    """

    global_thigh_angle_lp = 0

    ptr = 0    # for reading sensor data
    indx = 0   # for logging data
    null = 0   # number of null data points
    t_0 = dataOSL["Time"][ptr]   # for EKF
    start_time = t_0             # for live plotting
    fade_in_time = 2             # sec
    
    idx_min_prev = 0
    idx_min = 0
    idx_max_prev = 0
    idx_max = 0
    global_thigh_angle_window = np.zeros(100) # time window/ pre-allocate 1 sec
    global_thigh_vel_window = np.zeros(100)
    global_thigh_angle_shift = 0
    
    radius = 0
    radius_prev = 0
    walk = False
    walk_prev = False
    t_stop = 0
    t_walk = 0

    t_nwalk_ref = 0
    t_lwalk_ref = 0
    t_stop_ref = 0

    MD_prev = 0
    MD_threshold = 40 # 6.251(90%), 7.815(95%), 9.348(97.5%), 11.345(99%)
    lost = False
    t_lost = 0
    t_recover = 0

    knee_angle_initial = dataOSL['KneeAngle'][0]
    ankle_angle_initial = dataOSL['AnkleAngle'][0]
    knee_angle_model = 0
    ankle_angle_model = 0
    knee_angle_model_ref = 0
    ankle_angle_model_ref = 0

    simulation_log = {
        # state estimates
        "phase_est": np.zeros((len(dataOSL["Time"]), 1)),
        "phase_dot_est": np.zeros((len(dataOSL["Time"]), 1)),
        "step_length_est": np.zeros((len(dataOSL["Time"]), 1)),
        "ramp_est": np.zeros((len(dataOSL["Time"]), 1)),
        #"Sigma": np.zeros((len(dataOSL["Time"]), 4)),
        "MD": np.zeros((len(dataOSL["Time"]), 1)),
        "lost": np.zeros((len(dataOSL["Time"]), 1)),
        
        "idx_min_prev": np.zeros((len(dataOSL["Time"]), 1)),
        "idx_min": np.zeros((len(dataOSL["Time"]), 1)),
        "idx_max_prev": np.zeros((len(dataOSL["Time"]), 1)),
        "idx_max": np.zeros((len(dataOSL["Time"]), 1)),
        "global_thigh_angle_cline": np.zeros((len(dataOSL["Time"]), 1)),

        "global_thigh_angle_lp": np.zeros((len(dataOSL["Time"]), 1)),
        "global_thigh_vel_lp": np.zeros((len(dataOSL["Time"]), 1)),
        "global_thigh_angle_max": np.zeros((len(dataOSL["Time"]), 1)),
        "global_thigh_angle_min": np.zeros((len(dataOSL["Time"]), 1)),
        "global_thigh_vel_max": np.zeros((len(dataOSL["Time"]), 1)),
        "global_thigh_vel_min": np.zeros((len(dataOSL["Time"]), 1)),
        "phase_x": np.zeros((len(dataOSL["Time"]), 1)),
        "phase_y": np.zeros((len(dataOSL["Time"]), 1)),
        "radius": np.zeros((len(dataOSL["Time"]), 1)),
        "R": np.zeros((len(dataOSL["Time"]), 3)),
        "walk": np.zeros((len(dataOSL["Time"]), 1)),

        # EKF prediction of measurements/ derived measurements
        "global_thigh_angle_pred": np.zeros((len(dataOSL["Time"]), 1)),
        "global_thigh_angle": np.zeros((len(dataOSL["Time"]), 1)),
        "global_thigh_vel_pred": np.zeros((len(dataOSL["Time"]), 1)),
        "Atan2_pred": np.zeros((len(dataOSL["Time"]), 1)),
        "Atan2": np.zeros((len(dataOSL["Time"]), 1)),

        # Control commands
        "ankle_angle_model": np.zeros((len(dataOSL["Time"]), 1)),
        #"ankle_angle_cmd": np.zeros((len(dataOSL["Time"]), 1)),
        "knee_angle_model": np.zeros((len(dataOSL["Time"]), 1))
        #"knee_angle_cmd": np.zeros((len(dataOSL["Time"]), 1))
    }

    t_start = time.time()
    while True:
        ### Read OSL measurement data
        global_thigh_angle = dataOSL["ThighSagi"][ptr] * 180 / np.pi # deg # NO negative sign

        ## Calculate loadCell Fz using the buffer
        #if dataOSL['LoadCellFz'][ptr] > 50:
        #    loadCell_Fz_buffer = [loadCell_Fz_buffer[0], loadCell_Fz_buffer[0], loadCell_Fz_buffer[1]]
        #else:
        #    loadCell_Fz_buffer = [dataOSL['LoadCellFz'][ptr], loadCell_Fz_buffer[0], loadCell_Fz_buffer[1]]

        #dataOSL['LoadCellFz'][ptr] =  np.median(loadCell_Fz_buffer)
        #==========================================================================================================

        # time
        t = dataOSL["Time"][ptr]
        dt = t - t_0
        if ptr != 0 and dt < 0.0002:
            ptr += 1
            null += 1
            continue
        t_0 = t
        
        ## Compute measurements ###############################################################################################################
        alpha = 2 * np.pi * dt * fc / (2 * np.pi * dt * fc + 1) # for discrete-time 1st-order low-pass filter
        global_thigh_angle_lp = alpha * global_thigh_angle + (1 - alpha) * global_thigh_angle_lp
        if ptr == 0:
            global_thigh_vel_lp = 0
            global_thigh_angle_cline = global_thigh_angle_lp
        else:
            global_thigh_vel_lp = (global_thigh_angle_lp - global_thigh_angle_lp_0) / dt
            global_thigh_angle_cline = 0.05 * global_thigh_angle_lp + (1 - 0.05) * global_thigh_angle_cline
        global_thigh_angle_lp_0 = global_thigh_angle_lp
        
        # allocte more space if ptr exceed the bounds
        if ptr - idx_max_prev >= np.shape(global_thigh_angle_window)[0]: #np.shape(global_thigh_angle_window)[0]:
            #global_thigh_angle_window = np.concatenate((global_thigh_angle_window, np.zeros(1*len(global_thigh_angle_window))))
            #global_thigh_vel_window = np.concatenate((global_thigh_vel_window, np.zeros(1*len(global_thigh_angle_window))))
            global_thigh_angle_window = np.pad(global_thigh_angle_window, (0, np.shape(global_thigh_angle_window)[0]))
            global_thigh_vel_window = np.pad(global_thigh_vel_window, (0, np.shape(global_thigh_angle_window)[0]))
                
        global_thigh_angle_window[ptr - idx_max_prev] = global_thigh_angle_lp
        global_thigh_vel_window[ptr - idx_max_prev] = global_thigh_vel_lp
       
        if ptr > 0:
            global_thigh_angle_max = np.amax(global_thigh_angle_window[idx_min_prev - idx_max_prev:ptr - idx_max_prev])
            global_thigh_angle_min = np.amin(global_thigh_angle_window[idx_max_prev - idx_max_prev:ptr - idx_max_prev])
            global_thigh_vel_max = np.amax(global_thigh_vel_window[idx_min_prev - idx_max_prev:ptr - idx_max_prev])
            global_thigh_vel_min = np.amin(global_thigh_vel_window[idx_max_prev - idx_max_prev:ptr - idx_max_prev])

        # compute the indices
        if ptr > idx_max + 1:
            idx_min_temp = np.argmin(global_thigh_angle_window[idx_max - idx_max_prev:ptr - idx_max_prev]) + idx_max
            if ptr > idx_min_temp + 1 and global_thigh_angle_window[idx_min_temp - idx_max_prev] < global_thigh_angle_cline:
                idx_min = idx_min_temp
                idx_max_temp = np.argmax(global_thigh_angle_window[idx_min_temp - idx_max_prev:ptr - idx_max_prev]) + idx_min_temp
                if ptr > idx_max_temp + 1 and global_thigh_angle_window[idx_max_temp - idx_max_prev] > global_thigh_angle_cline: # new stride
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

            phase_y = - (global_thigh_vel_lp - global_thigh_vel_shift)
            phase_x = global_thigh_angle_scale * (global_thigh_angle_lp - global_thigh_angle_shift)
        else:
            phase_y = - global_thigh_vel_lp
            phase_x = global_thigh_angle_lp  * (2 * np.pi * 0.8)
        
        Atan2 = np.arctan2(phase_y, phase_x)
        if Atan2 < 0:
            Atan2 = Atan2 + 2 * np.pi
        
        c = 90
        d = 90
        radius = (phase_x / c) ** 2 + (phase_y / d) ** 2
        if radius >= 1:
            if radius_prev < 1:
                t_walk = t
            elif t - t_walk > 0.5:
                ekf.R = np.copy(R_org)
                walk = True
        elif radius < 1:
            if radius_prev >= 1:
                t_stop = t
            elif t - t_stop > 0.5:
                ekf.R[2, 2] = 1e20
                walk = False
        radius_prev = radius

        #######################################################################################################################################
        measurement = np.array([[global_thigh_angle], [global_thigh_vel_lp], [Atan2]])
        measurement = np.squeeze(measurement)

        ### EKF implementation
        ekf.Q = np.diag([0, 5e-3, 5e-3, 0]) * dt 
        ekf.prediction(dt)
        ekf.state_saturation(saturation_range)
        ekf.correction(measurement, Psi, using_atan2)
        ekf.state_saturation(saturation_range)

        ### Failure detector
        if ekf.MD > MD_threshold:
            if MD_prev <= MD_threshold:
                t_lost = t
            elif t - t_lost > 0.5:
                lost = True
        elif ekf.MD <= MD_threshold:
            if MD_prev > MD_threshold:
                t_recover = t
            elif t - t_recover > 0.75:
                lost = False
        MD_prev = ekf.MD
        
        ### Control commands: joints angles
        ## 1) Control commands generated by the trainned model
        if walk == False:
            # use the thigh_angle-phase look-up table
            pegleg_joint_angles = joints_control(0.3, 0, 1, 0)
            if walk_prev == True:
                t_stop_ref = t
                knee_angle_model_ref = knee_angle_model
                ankle_angle_model_ref = ankle_angle_model
            if t - t_stop_ref <= 1: # 1-sec transition to stop 
                trans_stop = t - t_stop_ref
                knee_angle_model = trans_stop * pegleg_joint_angles[0] + (1-trans_stop) * knee_angle_model_ref
                ankle_angle_model = trans_stop * pegleg_joint_angles[1] + (1-trans_stop) * ankle_angle_model_ref
            else:
                knee_angle_model = pegleg_joint_angles[0]
                ankle_angle_model = pegleg_joint_angles[1]

        elif walk == True and lost == False:
            joint_angles = joints_control(ekf.x[0, 0], ekf.x[1, 0], ekf.x[2, 0], ekf.x[3, 0])
            if walk_prev == False or (walk_prev == True and lost_prev == True):
                t_nwalk_ref = t
                knee_angle_model_ref = knee_angle_model
                ankle_angle_model_ref = ankle_angle_model
            if t - t_nwalk_ref <= 1: # 1-sec transition to normal walking 
                trans_nwalk = t - t_nwalk_ref
                knee_angle_model = trans_nwalk * joint_angles[0] + (1-trans_nwalk) * knee_angle_model_ref
                ankle_angle_model = trans_nwalk * joint_angles[1] + (1-trans_nwalk) * ankle_angle_model_ref
            else:
                knee_angle_model = joint_angles[0]
                ankle_angle_model = joint_angles[1]
        
        elif walk == True and lost == True:
            pegleg_joint_angles = joints_control(0.3, 0, 1, 0)
            if walk_prev == False or (walk_prev == True and lost_prev == False):
                t_lwalk_ref = t
                knee_angle_model_ref = knee_angle_model
                ankle_angle_model_ref = ankle_angle_model
            if t - t_lwalk_ref <= 1: # 1-sec transition to normal walking 
                trans_lwalk = t - t_lwalk_ref
                knee_angle_model = trans_lwalk * pegleg_joint_angles[0] + (1-trans_lwalk) * knee_angle_model_ref
                ankle_angle_model = trans_lwalk * pegleg_joint_angles[1] + (1-trans_lwalk) * ankle_angle_model_ref
            else:
                knee_angle_model = pegleg_joint_angles[0]
                ankle_angle_model = pegleg_joint_angles[1]
        lost_prev = lost
        walk_prev = walk

        ## 2) Control commands generated by the prescribed trajectory (lookup table) 
        #pv = int(ekf.x[0, 0] * 998)
        #pv = int(dataOSL['PV'][indx])
        #ankle_angle_cmd = refAnk[pv]
        #knee_angle_cmd = refKne[pv]

        # Fade-in effect at start
        elapsed_time = t - start_time
        if (elapsed_time < fade_in_time):
            fade_in = elapsed_time / fade_in_time 
            ankle_angle_model = ankle_angle_model * fade_in + ankle_angle_initial * (1 - fade_in)
            knee_angle_model = knee_angle_model * fade_in + knee_angle_initial * (1 - fade_in)
            #ankle_angle_cmd = ankle_angle_cmd * fade_in + ankle_angle_initial * (1 - fade_in)
            #knee_angle_cmd = knee_angle_cmd * fade_in + knee_angle_initial * (1 - fade_in)
        
        ## Loggging simulation results
        simulation_log['phase_est'][indx] = ekf.x[0, 0]
        simulation_log['phase_dot_est'][indx] = ekf.x[1, 0]
        simulation_log['step_length_est'][indx] = ekf.x[2, 0]
        simulation_log['ramp_est'][indx] = ekf.x[3, 0]
        #simulation_log['Sigma'][indx] = np.diag(ekf.Sigma)
        simulation_log['MD'][indx] = ekf.MD
        simulation_log['lost'][indx] = int(lost)
        simulation_log['walk'][indx] = int(walk)

        simulation_log["idx_min"][indx] = idx_min
        simulation_log["idx_min_prev"][indx] = idx_min_prev
        simulation_log["idx_max"][indx] = idx_max
        simulation_log["idx_max_prev"][indx] = idx_max_prev
        simulation_log["global_thigh_angle_cline"][indx] = global_thigh_angle_cline

        simulation_log["global_thigh_angle_lp"][indx] = global_thigh_angle_lp
        simulation_log["global_thigh_vel_lp"][indx] = global_thigh_vel_lp
   
        if indx > 0:
            simulation_log["global_thigh_angle_max"][indx] = global_thigh_angle_max
            simulation_log["global_thigh_angle_min"][indx] = global_thigh_angle_min
            simulation_log["global_thigh_vel_max"][indx] = global_thigh_vel_max
            simulation_log["global_thigh_vel_min"][indx] = global_thigh_vel_min
        simulation_log["phase_x"][indx] = phase_x
        simulation_log["phase_y"][indx] = phase_y
        simulation_log["radius"][indx] = radius
        simulation_log["R"][indx] = np.diag(ekf.R)
    
        simulation_log["global_thigh_angle_pred"][indx] = ekf.z_hat[0,0]
        simulation_log["global_thigh_angle"][indx] = global_thigh_angle
        simulation_log["global_thigh_vel_pred"][indx] = ekf.z_hat[1,0]
        simulation_log["Atan2_pred"][indx] = ekf.z_hat[2,0]
        simulation_log["Atan2"][indx] = Atan2

        simulation_log["ankle_angle_model"][indx] = ankle_angle_model
        #simulation_log["ankle_angle_cmd"][indx] = ankle_angle_cmd
        simulation_log["knee_angle_model"][indx] = knee_angle_model
        #simulation_log["knee_angle_cmd"][indx] = knee_angle_cmd

        ### Live plotting
        """
        elapsed_time = t - start_time
        if ptr % 2 == 0:
            sender.graph(elapsed_time, 
                         #dataOSL["PV"][ptr] / 998, ekf.x[0, 0], 'Phase', '-',
                         global_thigh_angle, ekf.z_hat[0], 'Global Thigh Angle', 'deg',
                         #ekf.z_hat[0], 'Global Thigh Angle Pred', 'deg',
                         global_thigh_vel_lp, global_thigh_vel_lp,'Global Thigh Angle Vel', 'deg/s',
                         #ekf.z_hat[1], 'Global Thigh Angle Vel Pred', 'deg/s'
                         #Atan2, 'atan2', '-',
                         #ekf.z_hat[2], 'atan2 Pred', '-'
                         #knee_angle, 'knee_angle', 'deg',
                         #dataOSL["KneeAngleRef"][ptr], knee_angle_cmd, 'Knee Angle', 'deg',
                         knee_angle_model, knee_angle_model, 'knee_angle_model', 'deg',
                         #dataOSL["AnkleAngleRef"][ptr], ankle_angle_cmd, 'Ankle Angle', 'deg',
                         ekf.x[1, 0],ekf.x[1, 0], 'phase_dot', '1/s',
                         #ekf.x[2, 0], 'step_length', 'm',
                         #ekf.x[3, 0], 'ramp_angle', 'deg'
                         )
        """
        ptr += 1
        indx += 1
        if (ptr >= len(dataOSL["Time"])-null-10): # 10
            break
        
except KeyboardInterrupt:
    print('\n*** OSL shutting down ***\n')

finally:
    print("Average time (ms) per iteration: ",  (time.time() - t_start) / ptr * 1000)

    ## Plot the results
    t_lower = dataOSL["Time"][0]
    t_upper = dataOSL["Time"][-1]
    plt.figure("Gait Phase")
    plt.subplot(411)
    plt.title("EKF Gait State Estimate")
    plt.plot(dataOSL["Time"], simulation_log['phase_est'], 'r-')
    #plt.plot(dataOSL["Time"], dataOSL['PV'] / 998, 'k-', alpha = 0.5)
    plt.ylabel("Phase")
    plt.xlim((t_lower, t_upper))
    plt.grid()
    plt.legend(('EKF phase', 'phase variable'))
    plt.subplot(412)
    plt.plot(dataOSL["Time"], simulation_log['phase_dot_est'], 'r-')
    plt.ylabel("Phase dot (1/s)")
    plt.xlim((t_lower, t_upper))
    plt.grid()
    plt.ylim((0, 2))
    plt.subplot(413)
    plt.plot(dataOSL["Time"], simulation_log['step_length_est'], 'r-')
    plt.ylabel("Stride Length (m)")
    plt.xlim((t_lower, t_upper))
    plt.grid()
    plt.ylim((0, 2))
    plt.subplot(414)
    plt.plot(dataOSL["Time"], simulation_log['ramp_est'], 'r-')
    plt.ylabel("Ramp (deg)")
    plt.xlim((t_lower, t_upper))
    plt.grid()

    plt.figure("R")
    plt.subplot(211)
    plt.semilogy(dataOSL["Time"], simulation_log["R"][:, 0])
    plt.semilogy(dataOSL["Time"], simulation_log["R"][:, 1])
    plt.semilogy(dataOSL["Time"], simulation_log["R"][:, 2])
    plt.ylabel("diag R")
    plt.xlim((t_lower, t_upper))
    plt.grid()
    plt.subplot(212)
    plt.plot(dataOSL["Time"], simulation_log["radius"], label = 'radius')
    plt.plot(dataOSL["Time"], simulation_log["walk"] * 10, 'k-', label = 'walk')
    plt.axhline(y=1, color='r', linestyle='-', label = '1')
    plt.ylabel("radius")
    plt.xlabel("time (s)")
    plt.xlim((t_lower, t_upper))
    plt.legend()
    plt.grid()
    #plt.figure("Sigma")
    #plt.plot(dataOSL["Time"], simulation_log["Sigma"][:, 0])
    #plt.plot(dataOSL["Time"], simulation_log["Sigma"][:, 1])
    #plt.plot(dataOSL["Time"], simulation_log["Sigma"][:, 2])
    #plt.plot(dataOSL["Time"], simulation_log["Sigma"][:, 3])

    plt.figure("Measurements")
    plt.subplot(311)
    plt.title("Measurements")
    plt.plot(dataOSL["Time"], simulation_log["global_thigh_angle"], 'k-')
    plt.plot(dataOSL["Time"], simulation_log["global_thigh_angle_pred"], 'r-')
    plt.legend(('actual', 'EKF predicted'))
    plt.ylabel("Global Thigh Angle (deg)")
    plt.xlim((t_lower, t_upper))
    plt.grid()
    plt.subplot(312)
    plt.plot(dataOSL["Time"], simulation_log["global_thigh_vel_lp"], 'k-')
    plt.plot(dataOSL["Time"], simulation_log["global_thigh_vel_pred"], 'r-')
    plt.legend(('actual', 'EKF predicted'))
    plt.ylabel("Global Thigh Angle Vel (deg/s)")
    plt.xlim((t_lower, t_upper))
    plt.grid()
    plt.subplot(313)
    plt.plot(dataOSL["Time"], simulation_log["Atan2"], 'k-', label = 'atan2')
    plt.plot(dataOSL["Time"], simulation_log["Atan2_pred"], 'r-', label = 'atan2_pred')
    plt.plot(dataOSL["Time"], simulation_log['phase_est'] * 2*np.pi, label = 'ekf phase')
    #plt.plot(dataOSL["Time"], dataOSL['PV'] / 998 * 2*np.pi, alpha = 0.5, label = 'pv')
    plt.legend(('actual', 'EKF predicted'))
    plt.ylabel("Atan2")
    plt.xlabel("Time (s)")
    plt.xlim((t_lower, t_upper))
    plt.legend()
    plt.grid()

    plt.figure("MD")
    plt.subplot(211)
    plt.plot(dataOSL["Time"], simulation_log['MD'])
    plt.ylabel("MD")
    plt.xlim((t_lower, t_upper))
    plt.grid()
    plt.subplot(212)
    plt.plot(dataOSL["Time"], simulation_log['lost'])
    plt.ylabel("lost (T/F)")
    plt.xlim((t_lower, t_upper))
    plt.grid()

    plt.figure("Atan2")
    plt.subplot(411)
    plt.plot(dataOSL["Time"], simulation_log["global_thigh_angle_lp"])
    plt.plot(dataOSL["Time"], simulation_log["global_thigh_angle_max"], 'r-')
    plt.plot(dataOSL["Time"], simulation_log["global_thigh_angle_min"], 'b-')
    plt.plot(dataOSL["Time"], simulation_log["global_thigh_angle_cline"], 'k-')
    plt.plot(dataOSL["Time"], simulation_log["global_thigh_angle"], 'm-', alpha = 0.5)
    #plt.plot(simulation_log["idx_min"], np.zeros(np.shape(simulation_log["idx_min"])[0]), 'bx', label = 'min')
    #plt.plot(simulation_log["idx_min_prev"], np.zeros(np.shape(simulation_log["idx_min_prev"])[0]),'g.', label = 'min_prev', alpha = 0.5)
    #plt.plot(simulation_log["idx_max"], np.zeros(np.shape(simulation_log["idx_max"])[0]),'rx', label = 'max')
    #plt.plot(simulation_log["idx_max_prev"], np.zeros(np.shape(simulation_log["idx_max_prev"])[0]), 'm.', label = 'max_prev', alpha = 0.5)
    plt.ylabel("low-passed $\\theta_{th}$ (deg)")
    plt.grid()
    plt.subplot(412)
    plt.plot(dataOSL["Time"], simulation_log["global_thigh_vel_lp"])
    plt.plot(dataOSL["Time"], simulation_log["global_thigh_vel_max"], 'r-')
    plt.plot(dataOSL["Time"], simulation_log["global_thigh_vel_min"], 'b-')
    plt.ylabel("low-passed $\dot{\\theta_{th}}$ (deg/s)")
    plt.grid()
    plt.subplot(413)
    plt.plot(dataOSL["Time"], simulation_log["Atan2"] / 2/np.pi, 'b-', label = 'Atan2/2pi')
    plt.plot(dataOSL["Time"], simulation_log['phase_est'], 'r-', label = 'phase_est')
    #plt.plot(dataOSL['PV'] / 998, 'k-', alpha = 0.5, label = 'PV')
    plt.ylabel("Atan2")
    plt.grid()
    plt.subplot(414)
    plt.plot(dataOSL["Time"], simulation_log["radius"])
    plt.axhline(y=1, color='r', linestyle='-')
    plt.xlabel("Time (s)")
    plt.ylabel("radius")
    plt.grid()

    plt.figure("Atan2 Phase Portrait")
    plt.subplot(211)
    t = np.linspace(0, 2*np.pi, 100)
    plt.plot(c*np.cos(t) , d*np.sin(t), 'r--')
    plt.plot(simulation_log["phase_x"], simulation_log["phase_y"])
    plt.xlabel("phase_x")
    plt.ylabel("phase_y")
    plt.axis('equal')
    plt.grid()
    plt.subplot(212)
    plt.plot(dataOSL["Time"], simulation_log["radius"])
    #plt.plot(dataOSL["Time"], simulation_log["radius_org"], label = 'original')
    plt.axhline(y=1, color='r', linestyle='-')
    #plt.plot(dataOSL["Time"], simulation_log["phase_x"], label = 'phase_x')
    #plt.plot(dataOSL["Time"], simulation_log["phase_y"], label = 'phase_y')
    plt.grid()
    #plt.legend()

    #plt.figure()
    #plt.plot(simulation_log["idx_min"], 'b-', label = 'min')
    #plt.plot(simulation_log["idx_min_prev"], 'g--', label = 'min_prev', alpha = 0.5)
    #plt.plot(simulation_log["idx_max"], 'r-', label = 'max')
    #plt.plot(simulation_log["idx_max_prev"], 'm--', label = 'max_prev', alpha = 0.5)
    #plt.legend()
    #plt.grid()

    plt.figure("Joints Angles")
    plt.subplot(411)
    plt.title("Joints Angles Commands")
    plt.plot(dataOSL["Time"], simulation_log['phase_est'], 'r-')
    #plt.plot(dataOSL["Time"], dataOSL['PV'] / 998, 'k-')
    plt.ylabel("Phase")
    plt.xlim((t_lower, t_upper))
    plt.legend(('EKF phase', 'phase variable'))
    plt.grid()
    plt.subplot(412)
    #plt.plot(dataOSL["Time"], dataOSL["AnkleAngleRef"], 'k-', label = 'recorded coomand')
    plt.plot(dataOSL["Time"], dataOSL["AnkleAngle"], 'k-', label = 'recorded actual')
    #plt.plot(dataOSL["Time"], simulation_log["ankle_angle_cmd"], 'r-', label = 'prescribed trajectories')
    plt.plot(dataOSL["Time"], simulation_log["ankle_angle_model"], 'm-', label = 'kinematics model')
    #plt.plot(dataOSL["Time"], dataOSL['AnkleAngle'], 'b-')
    plt.legend()
    plt.ylabel("Ankle angle command(deg)")
    plt.xlim((t_lower, t_upper))
    plt.ylim((-30, 30))
    plt.grid()
    plt.subplot(413)
    #plt.plot(dataOSL["Time"], dataOSL["KneeAngleRef"], 'k-', label = 'recorded coomand')
    plt.plot(dataOSL["Time"], dataOSL["KneeAngle"], 'k-', label = 'recorded')
    #plt.plot(dataOSL["Time"], simulation_log["knee_angle_cmd"], 'r-', label = 'prescribed trajectories')
    plt.plot(dataOSL["Time"], simulation_log["knee_angle_model"], 'm-', label = 'kinematics model')
    #plt.plot(dataOSL["Time"], dataOSL['KneeAngle'], 'b-')
    plt.legend()
    plt.ylabel("Knee angle command(deg)")
    plt.xlim((t_lower, t_upper))
    plt.ylim((-80, 5))
    plt.grid()
    plt.subplot(414)
    plt.plot(dataOSL["Time"], simulation_log["lost"], 'r--', label = 'lost')
    plt.plot(dataOSL["Time"], simulation_log["walk"], 'b-', label = 'walk')
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Indicators(T/F)")
    plt.xlim((t_lower, t_upper))
    plt.grid()

    """
    plt.figure("Steady-state Walking Detector")
    plt.subplot(311)
    plt.plot(dataOSL["Time"], simulation_log["MD"], 'r-')
    plt.plot(dataOSL["Time"], simulation_log["MD_movingAverage"], 'm-')
    plt.plot(dataOSL["Time"], MD_threshold * np.ones((len(dataOSL["Time"]), 1)), 'b--')
    plt.legend(('MD', 'Moving averagre of MD'))
    plt.ylabel("MD")
    plt.xlim((t_lower, t_upper))
    plt.subplot(312)
    plt.plot(dataOSL["Time"], simulation_log["global_thigh_angle_min"], 'g-')
    plt.plot(dataOSL["Time"], simulation_log["global_thigh_angle_max"], 'm-')
    plt.plot(dataOSL["Time"], global_thigh_angle_min_threshold * np.ones((len(dataOSL["Time"]), 1)), 'g--')
    plt.plot(dataOSL["Time"], global_thigh_angle_max_threshold * np.ones((len(dataOSL["Time"]), 1)), 'm--')
    plt.legend(('Min. thigh angle in the moving window', 'Max. thigh angle in the moving window', 
                'Threshold for Min. thigh angle', 'Threshold for Max. thigh angle'))
    plt.ylabel("Global Thigh Angle Range (deg)")
    plt.xlim((t_lower, t_upper))
    plt.subplot(313)
    plt.plot(dataOSL["Time"], simulation_log["steady-state"], 'r-')
    plt.plot(dataOSL["Time"], simulation_log["walking"], 'b--')
    plt.legend(('Steady-state walking', 'Walking'))
    plt.ylabel("Walking Status (T/F)")
    plt.xlim((t_lower, t_upper))
    plt.xlabel("Time (s)")
    #plt.subplot(313)
    #plt.plot(dataOSL["Time"], t_s)
    #plt.plot(dataOSL["Time"], t_ns)
    #plt.plot(dataOSL["Time"], t_s - t_ns)
    #plt.legend(('t_s', 't_ns', 'steady-state time'))
    #plt.ylabel("Steady-state time (s)")
    #plt.xlim((t_lower, t_upper))
    """

    """
    plt.figure("Kinetics")
    plt.subplot(211)
    plt.plot(dataOSL["Time"], dataOSL['LoadCellFz'])
    #plt.plot(datatxt["Time"], datatxt['loadCelFz'])
    plt.ylabel("Load Cell Force (N)")
    plt.xlabel("Time (s)")
    plt.xlim((t_lower, t_upper))
    #plt.ylim((-300, 20))
    plt.legend(('Median-filtered', 'Original'))
    
    plt.subplot(212)
    #plt.plot(dataOSL["Time"], dataOSL['LoadCellMx'])
    #plt.plot(dataOSL["Time"], dataOSL['LoadCellMy'])
    #plt.plot(dataOSL["Time"], dataOSL['LoadCellMz'])
    plt.plot(dataOSL["Time"], dataOSL['AnkleTorque'])
    #plt.plot(dataOSL["Time"], dataOSL['AnkleTorqueSwing'])
    #plt.plot(dataOSL["Time"], dataOSL['KneeTorque'])
    plt.ylabel("Ankle Moment (N-m)")
    plt.xlabel("Time (s)")
    plt.xlim((t_lower, t_upper))
    #plt.ylim((-2, 3))
    #plt.legend(('Load Cell Mx', 'Load Cell My', 'Ankle Torque', 'Knee Torque'))
    """
    plt.show()
