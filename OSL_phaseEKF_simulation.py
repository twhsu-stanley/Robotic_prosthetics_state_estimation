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
# 4) 210714_113523_OSL_benchtop_test

# 5) 210726_102901_OSL_parallelBar_test
# 6) 210730_140347_OSL_parallelBar_test
datatxt = np.genfromtxt(logFile , delimiter=',', names = True)
dataOSL = {
    "Time": datatxt["Time"],
    "ThighSagi": datatxt["ThighSagi"],
    "PV": datatxt['PV'],
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
#"""
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
    sensors = ['globalThighAngles', 'globalThighVelocities', 'atan2']#, 'ankleMoment', 'tibiaForce'

    sensor_id = [sensors_dict[key] for key in sensors]
    sensor_id_str = ""
    for i in range(len(sensor_id)):
        sensor_id_str += str(sensor_id[i])
    m_model = model_loader('Measurement_model_' + sensor_id_str +'_NSL.pickle')

    using_atan2 = np.any(np.array(sensors) == 'atan2')

    Psi = np.array([load_Psi('Generic')[key] for key in sensors], dtype = object)
    
    #saturation_range = saturation_bounds()
    saturation_range = np.array([1.8, 0, 2, 0])

    ## build the system
    sys = myStruct()
    sys.f = process_model
    sys.A = A
    sys.h = m_model
    sys.Q = np.diag([0, 5e-3, 5e-3, 0])
    # measurement noise covariance
    U = np.diag([1, 1, 1e10])
    R = U @ measurement_noise_covariance(*sensors) @ U.T
    R_org = np.copy(R)
    sys.R = R

    # initialize the state
    init = myStruct()
    init.x = np.array([[0.3], [0], [0], [0]])
    init.Sigma = np.diag([1, 1, 1, 0])

    ekf = extended_kalman_filter(sys, init)

    ########## Create filters ################################################################
    
    fs = 1 / np.average(np.diff(dataOSL["Time"]))        # sampling rate = 100 Hz (actual: ~77 Hz)
    print("Average fs = %4.2f Hz" % fs)
    nyq = 0.5 * fs    # Nyquist frequency = fs/2
    ## configure low-pass filter (1-order)
    fc_normal = 2 / nyq   #cut-off frequency = 2Hz
    b_lp, a_lp = butter(1, fc_normal, btype = 'low', analog = False)
    z_lp_1 = lfilter_zi(b_lp,  a_lp)
    z_lp_2 = lfilter_zi(b_lp,  a_lp)
    
    ## configure band-pass filter (2-order)
    normal_lowcut = 0.1 / nyq    #lower cut-off frequency = 0.5Hz 
    normal_highcut = 2 / nyq     #upper cut-off frequency = 2Hz
    b_bp, a_bp = butter(2, [normal_lowcut, normal_highcut], btype = 'band', analog = False)
    z_bp = lfilter_zi(b_bp,  a_bp)

    ### Kinetic data processing
    #dataOSL['AnkleTorque_lp'] = butter_lowpass_filter(dataOSL['AnkleTorque'], 10, fs, order = 1)
    #dataOSL['LoadCellFz_lp'] = butter_lowpass_filter(dataOSL['LoadCellFz'], 10, fs, order = 1)

    ptr = 0    # for reading sensor data
    indx = 0   # for logging data
    null = 0   # number of null data points
    t_0 = dataOSL["Time"][ptr]   # for EKF
    start_time = t_0             # for live plotting
    fade_in_time = 2             # sec

    t_s = np.zeros((len(dataOSL["Time"]), 1))  # for steady-state
    t_ns = np.zeros((len(dataOSL["Time"]), 1)) # for non steady-state
    t_s[0] = start_time
    t_ns[0] = start_time
    t_s_previous = start_time
    t_ns_previous = start_time
    #steady_state = False
    #steady_state_previous = False
    #walking = False

    #MD_hist= deque([])
    global_thigh_angle_hist = np.ones((int(fs*2.5), 1)) * dataOSL["ThighSagi"][0] * 180 / np.pi # ~ 2seconds window

    #MD_threshold = 5 # MD
    global_thigh_angle_max_threshold = 10    # global thigh angle range (deg)
    global_thigh_angle_min_threshold = -5     # global thigh angle range (deg)
    
    knee_angle_initial = dataOSL['KneeAngle'][0]
    ankle_angle_initial = dataOSL['AnkleAngle'][0]
    
    loadCell_Fz_buffer = []
    for i in range(3):
        loadCell_Fz_buffer.append(dataOSL['LoadCellFz'][ptr])
    dataOSL['LoadCellFz'][ptr] = np.median(loadCell_Fz_buffer)

    #subject_weight = 60

    simulation_log = {
        # state estimates
        "phase_est": np.zeros((len(dataOSL["Time"]), 1)),
        "phase_dot_est": np.zeros((len(dataOSL["Time"]), 1)),
        "step_length_est": np.zeros((len(dataOSL["Time"]), 1)),
        "ramp_est": np.zeros((len(dataOSL["Time"]), 1)),
        
        #"steady-state": np.zeros((len(dataOSL["Time"]), 1)),
        "walking": np.zeros((len(dataOSL["Time"]), 1)),
        "global_thigh_angle_min": np.zeros((len(dataOSL["Time"]), 1)),
        "global_thigh_angle_max": np.zeros((len(dataOSL["Time"]), 1)),
        "atan2_radius": np.zeros((len(dataOSL["Time"]), 1)),
        #"MD": np.zeros((len(dataOSL["Time"]), 1)),
        #"MD_movingAverage": np.zeros((len(dataOSL["Time"]), 1)),

        # EKF prediction of measurements/ derived measurements
        "global_thigh_angle_pred": np.zeros((len(dataOSL["Time"]), 1)),
        "global_thigh_angle": np.zeros((len(dataOSL["Time"]), 1)),
        "global_thigh_angle_bp": np.zeros((len(dataOSL["Time"]), 1)),
        "global_thigh_angle_vel_blp": np.zeros((len(dataOSL["Time"]), 1)),
        #"ankleMoment_pred": np.zeros((len(dataOSL["Time"]), 1)),
        #"ankleMoment": np.zeros((len(dataOSL["Time"]), 1)),
        "global_thigh_angle_vel_pred": np.zeros((len(dataOSL["Time"]), 1)),
        "global_thigh_angle_vel": np.zeros((len(dataOSL["Time"]), 1)),
        "Atan2_pred": np.zeros((len(dataOSL["Time"]), 1)),
        "Atan2": np.zeros((len(dataOSL["Time"]), 1)),

        # Control commands
        "ankle_angle_model": np.zeros((len(dataOSL["Time"]), 1)),
        "ankle_angle_cmd": np.zeros((len(dataOSL["Time"]), 1)),
        "knee_angle_model": np.zeros((len(dataOSL["Time"]), 1)),
        "knee_angle_cmd": np.zeros((len(dataOSL["Time"]), 1))
    }

    while True:
        ### Read OSL measurement data
        global_thigh_angle = dataOSL["ThighSagi"][ptr] * 180 / np.pi # deg # NO negative sign
        #ankle_angle = dataOSL['AnkleAngle'][ptr]
        #knee_angle = dataOSL['KneeAngle'][ptr]
        #ankleMoment = dataOSL['AnkleTorque'][ptr] / subject_weight

        ## Calculate loadCell Fz using the buffer
        if dataOSL['LoadCellFz'][ptr] > 50:
            loadCell_Fz_buffer = [loadCell_Fz_buffer[0], loadCell_Fz_buffer[0], loadCell_Fz_buffer[1]]
        else:
            loadCell_Fz_buffer = [dataOSL['LoadCellFz'][ptr], loadCell_Fz_buffer[0], loadCell_Fz_buffer[1]]

        dataOSL['LoadCellFz'][ptr] =  np.median(loadCell_Fz_buffer)
        #==========================================================================================================

        # time
        t = dataOSL["Time"][ptr]
        dt = t - t_0
        if ptr != 0 and dt < 0.0002:
            ptr += 1
            null += 1
            continue
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
        
        # neglect atan2 when the subejct is not waking
        global_thigh_angle_hist = np.roll(global_thigh_angle_hist, -1)
        global_thigh_angle_hist[-1] = global_thigh_angle
        
        #if (min(global_thigh_angle_hist) < global_thigh_angle_min_threshold
        #    and max(global_thigh_angle_hist) > global_thigh_angle_max_threshold):
        a = 5
        b = 5
        radius = (global_thigh_angle_bp / a)**2 + (global_thigh_angle_vel_blp / b)**2
        if radius >= 1:
            #walking = True
            ekf.R = np.copy(R_org)
        else:
            #walking = False
            ekf.R[2, 2] = 1e30

        measurement = np.array([[global_thigh_angle], [global_thigh_angle_vel_lp], [Atan2]])
        measurement = np.squeeze(measurement)

        ### EKF implementation
        ekf.Q = np.diag([0, 5e-2, 5e-2, 0]) * dt
        ekf.prediction(dt)
        ekf.state_saturation(saturation_range)
        ekf.correction(measurement, Psi, using_atan2) #, steady_state_walking = steady_state
        ekf.state_saturation(saturation_range)

        # Detect steady-state waling =========================================================================
        """
        if len(MD_hist) < int(fs * 2.5):
            MD_hist.append(ekf.MD)
        else:
            MD_hist.append(ekf.MD)
            MD_hist.popleft()
        MD_movingAverage = np.mean(MD_hist)
        
        if MD_movingAverage < MD_threshold:
            t_s[indx] = t
            t_ns[indx] = t_ns_previous
        else:
            t_ns[indx] = t
            t_s[indx] = t_s_previous
        
        if t_s[indx] - t_ns[indx] > 3:
            steady_state = True
        #else:
        elif t_s[indx] - t_ns[indx] < -1:
            steady_state = False
        
        steady_state = steady_state and walking
        
        t_s_previous = t_s[indx]
        t_ns_previous = t_ns[indx]
        
        if steady_state == True and steady_state_previous == False:
            ekf.Q = np.diag([0, 1e-5, 1e-5, 0])
            ekf.Sigma = np.diag([1, 1, 1, 0])

        elif steady_state == False and steady_state_previous == True:
            ekf.Q = np.diag([0, 1e-5, 0, 0]) 
            ekf.Sigma = np.diag([1, 1, 0, 0])
            ekf.x[2, 0] = 1.1
            ekf.x[3, 0] = 0
        
        steady_state_previous = steady_state
        """
        #=====================================================================================================

        ### Control commands: joints angles
        ## 1) Control commands generated by the trainned model
        #joint_angles = joints_control(ekf.x[0, 0], ekf.x[1, 0], ekf.x[2, 0], ekf.x[3, 0])
        joint_angles = joints_control(dataOSL['PV'][indx]/998, ekf.x[1, 0], ekf.x[2, 0], ekf.x[3, 0])
        knee_angle_model = joint_angles[0]
        ankle_angle_model = joint_angles[1] # negative sign
        
        # saturate commands to actuators
        """
        if knee_angle_model > -5: 
            knee_angle_model = -5
        if knee_angle_model < -60:
            knee_angle_model = -60
        
        if ankle_angle_model > 18: 
            ankle_angle_model = 18
        if ankle_angle_model < -10:
            ankle_angle_model = -10
        """
        
        ## 2) Control commands generated by the established trajectory
        #pv = int(ekf.x[0, 0] * 998)
        pv = int(dataOSL['PV'][indx])
        ankle_angle_cmd = refAnk[pv]
        knee_angle_cmd = refKne[pv]

        # Fade-in effect
        elapsed_time = t - start_time
        if (elapsed_time < fade_in_time):
            alpha = elapsed_time / fade_in_time 
            ankle_angle_cmd = ankle_angle_cmd * alpha + ankle_angle_initial * (1 - alpha)
            knee_angle_cmd = knee_angle_cmd * alpha + knee_angle_initial * (1 - alpha)
        
        ## Loggging simulation results
        simulation_log['phase_est'][indx] = ekf.x[0, 0]
        simulation_log['phase_dot_est'][indx] = ekf.x[1, 0]
        simulation_log['step_length_est'][indx] = ekf.x[2, 0]
        simulation_log['ramp_est'][indx] = ekf.x[3, 0]
        
        #simulation_log["steady-state"][indx] = steady_state
        #simulation_log["R"][indx] = np.diag(sys.R)
        simulation_log["global_thigh_angle_min"][indx] = min(global_thigh_angle_hist)
        simulation_log["global_thigh_angle_max"][indx] = max(global_thigh_angle_hist)
        simulation_log["atan2_radius"][indx] = radius
        #simulation_log["MD"][indx] = ekf.MD
        #simulation_log["MD_movingAverage"][indx] = MD_movingAverage

        simulation_log["global_thigh_angle_pred"][indx] = ekf.z_hat[0,0]
        simulation_log["global_thigh_angle"][indx] = global_thigh_angle
        simulation_log["global_thigh_angle_bp"][indx] = global_thigh_angle_bp
        simulation_log["global_thigh_angle_vel_blp"][indx] = global_thigh_angle_vel_blp
        #simulation_log["ankleMoment_pred"][indx] = ekf.z_hat[1,0]
        #simulation_log["ankleMoment"][indx] = ankleMoment
        simulation_log["global_thigh_angle_vel_pred"][indx] = ekf.z_hat[1,0]
        simulation_log["global_thigh_angle_vel"][indx] = global_thigh_angle_vel_lp
        simulation_log["Atan2_pred"][indx] = ekf.z_hat[2,0]
        simulation_log["Atan2"][indx] = Atan2

        simulation_log["ankle_angle_model"][indx] = ankle_angle_model
        simulation_log["ankle_angle_cmd"][indx] = ankle_angle_cmd
        simulation_log["knee_angle_model"][indx] = knee_angle_model
        simulation_log["knee_angle_cmd"][indx] = knee_angle_cmd
        
        ### Live plotting
        """
        elapsed_time = t - start_time
        if ptr % 2 == 0:
            sender.graph(elapsed_time, 
                         dataOSL["PV"][ptr] / 998, ekf.x[0, 0], 'Phase', '-',
                         global_thigh_angle, ekf.z_hat[0], 'Global Thigh Angle', 'deg',
                         #ekf.z_hat[0], 'Global Thigh Angle Pred', 'deg',
                         #global_thigh_angle_vel_lp, 'Global Thigh Angle Vel', 'deg/s',
                         #ekf.z_hat[1], 'Global Thigh Angle Vel Pred', 'deg/s'
                         #Atan2, 'atan2', '-',
                         #ekf.z_hat[2], 'atan2 Pred', '-'
                         #knee_angle, 'knee_angle', 'deg',
                         dataOSL["KneeAngleRef"][ptr], knee_angle_cmd, 'Knee Angle', 'deg',
                         #knee_angle_model, 'knee_angle_model', 'deg',
                         dataOSL["AnkleAngleRef"][ptr], ankle_angle_cmd, 'Ankle Angle', 'deg',
                         #ekf.x[1, 0], 'phase_dot', '1/s',
                         #ekf.x[2, 0], 'step_length', 'm',
                         #ekf.x[3, 0], 'ramp_angle', 'deg'
                         )
        """
        ptr += 1
        indx += 1
        if (ptr >= len(dataOSL["Time"])-null-10):
            break
        
except KeyboardInterrupt:
    print('\n*** OSL shutting down ***\n')

finally:
    ## Plot the results
    t_lower = dataOSL["Time"][0]
    t_upper = dataOSL["Time"][-1]
    plt.figure("Gait Phase")
    plt.subplot(411)
    plt.title("EKF Gait State Estimate")
    plt.plot(dataOSL["Time"], simulation_log['phase_est'], 'r-')
    plt.plot(dataOSL["Time"], dataOSL['PV'] / 998, 'k-', alpha = 0.5)
    plt.ylabel("Phase")
    plt.xlim((t_lower, t_upper))
    plt.legend(('EKF phase', 'phase variable'))
    plt.subplot(412)
    plt.plot(dataOSL["Time"], simulation_log['phase_dot_est'], 'r-')
    plt.ylabel("Phase dot (1/s)")
    plt.xlim((t_lower, t_upper))
    plt.ylim((0, 2))
    plt.subplot(413)
    plt.plot(dataOSL["Time"], simulation_log['step_length_est'], 'r-')
    plt.ylabel("Stride Length (m)")
    plt.xlim((t_lower, t_upper))
    plt.ylim((0, 2))
    plt.subplot(414)
    plt.plot(dataOSL["Time"], simulation_log['ramp_est'], 'r-')
    plt.ylabel("Ramp (deg)")
    plt.xlim((t_lower, t_upper))

    plt.figure("Measurements")
    plt.subplot(311)
    plt.title("Measurements")
    plt.plot(dataOSL["Time"], simulation_log["global_thigh_angle"], 'k-')
    plt.plot(dataOSL["Time"], simulation_log["global_thigh_angle_pred"], 'r-')
    #plt.plot(dataOSL["Time"], simulation_log["global_thigh_angle_bp"], 'm--')
    plt.legend(('actual', 'EKF predicted'))
    plt.ylabel("Global Thigh Angle (deg)")

    plt.xlim((t_lower, t_upper))
    #plt.subplot(412)
    #plt.plot(dataOSL["Time"], simulation_log["global_thigh_angle_min"], 'g-')
    #plt.plot(dataOSL["Time"], simulation_log["global_thigh_angle_max"], 'm-')
    #plt.plot(dataOSL["Time"], global_thigh_angle_min_threshold * np.ones((len(dataOSL["Time"]), 1)), 'g--')
    #plt.plot(dataOSL["Time"], global_thigh_angle_max_threshold * np.ones((len(dataOSL["Time"]), 1)), 'm--')
    #plt.legend(('Min. thigh angle in the moving window', 'Max. thigh angle in the moving window', 
    #            'Threshold for Min. thigh angle', 'Threshold for Max. thigh angle'))
    #plt.ylabel("Global Thigh Angle Range (deg)")
    #plt.xlim((t_lower, t_upper))
    plt.subplot(312)
    plt.plot(dataOSL["Time"], simulation_log["global_thigh_angle_vel"], 'k-')
    plt.plot(dataOSL["Time"], simulation_log["global_thigh_angle_vel_pred"], 'r-')
    plt.legend(('actual', 'EKF predicted'))
    plt.ylabel("Global Thigh Angle Vel (deg/s)")
    plt.xlim((t_lower, t_upper))
    plt.subplot(313)
    plt.plot(dataOSL["Time"], simulation_log["Atan2"], 'k-', label = 'atan2')
    plt.plot(dataOSL["Time"], simulation_log["Atan2_pred"], 'r-', label = 'atan2_pred')
    plt.plot(dataOSL["Time"], simulation_log['phase_est'] * 2*np.pi, label = 'ekf phase')
    plt.plot(dataOSL["Time"], dataOSL['PV'] / 998 * 2*np.pi, alpha = 0.5, label = 'pv')
    plt.legend(('actual', 'EKF predicted'))
    plt.ylabel("Atan2")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.xlim((t_lower, t_upper))

    plt.figure("Atan2")
    plt.subplot(311)
    plt.plot(dataOSL["Time"], simulation_log["global_thigh_angle"], 'k-')
    plt.plot(dataOSL["Time"], simulation_log["global_thigh_angle_bp"])
    plt.ylabel("Band-passed $\\theta_{th}$ (deg)")
    plt.legend(('actual', 'band-passed'))
    plt.grid()
    plt.subplot(312)
    plt.plot(dataOSL["Time"], -simulation_log["global_thigh_angle_vel_blp"])
    plt.ylabel("Band-passed & Scaled $\dot{\\theta_{th}}$ (deg/s)")
    plt.grid()
    plt.subplot(313)
    plt.plot(dataOSL["Time"], simulation_log["Atan2"], 'k-')
    plt.ylabel("Atan2")
    plt.xlabel("Time (s)")
    plt.xlim((t_lower, t_upper))
    plt.grid()

    plt.figure("Atan2 Phase Portrait")
    t = np.linspace(0, 2*np.pi, 100)
    plt.plot( a*np.cos(t) , b*np.sin(t), 'r--')
    plt.plot(simulation_log["global_thigh_angle_bp"], -simulation_log["global_thigh_angle_vel_blp"])
    plt.xlabel("Band-passed Global Thigh Angle (deg)")
    plt.ylabel("Band-passed & Scaled Global Thigh Angle Vel (deg/s)")
    plt.grid()

    plt.figure()
    plt.plot(dataOSL["Time"], simulation_log["atan2_radius"])
    plt.axhline(y = 1, color='r', linestyle='-')
    plt.grid()

    plt.figure("Joints Angles")
    plt.subplot(311)
    plt.title("Joints Angles Commands")
    plt.plot(dataOSL["Time"], simulation_log['phase_est'], 'r-')
    #plt.plot(dataOSL["Time"], dataOSL['PV'] / 998, 'k-')
    plt.ylabel("Phase")
    plt.xlim((t_lower, t_upper))
    plt.legend(('EKF phase', 'phase variable'))
    plt.subplot(312)
    plt.plot(dataOSL["Time"], dataOSL["AnkleAngleRef"], 'k-')
    plt.plot(dataOSL["Time"], simulation_log["ankle_angle_cmd"], 'r-')
    plt.plot(dataOSL["Time"], simulation_log["ankle_angle_model"], 'm-')
    #plt.plot(dataOSL["Time"], dataOSL['AnkleAngle'], 'b-')
    plt.legend(('recorded', 'prescribed trajectories', 'kinematic model'))
    plt.ylabel("Ankle angle command(deg)")
    plt.xlim((t_lower, t_upper))
    plt.ylim((-30, 30))
    plt.subplot(313)
    plt.plot(dataOSL["Time"], dataOSL["KneeAngleRef"], 'k-')
    plt.plot(dataOSL["Time"], simulation_log["knee_angle_cmd"], 'r-')
    plt.plot(dataOSL["Time"], simulation_log["knee_angle_model"], 'm-')
    #plt.plot(dataOSL["Time"], dataOSL['KneeAngle'], 'b-')
    plt.legend(('recorded', 'prescribed trajectories', 'kinematic model'))
    plt.ylabel("Knee angle command(deg)")
    plt.xlabel("Time (s)")
    plt.xlim((t_lower, t_upper))
    plt.ylim((-80, 5))

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
