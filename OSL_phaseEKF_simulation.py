"""
Code to simulate the phase-EKF and control commands using pre-revorded walking data
"""
import sys, time
import matplotlib.pyplot as plt
import numpy as np
import csv

from EKF import *
from model_framework import *
from scipy.signal import butter, lfilter, lfilter_zi
import scipy.io
import sender_test as sender   # for real-time plotting

### A. Load Ross's pre-recorded walking data
"""
logFile = r"OSL_walking_data/210617_113644_PV_Siavash_walk_oscillations in phase.csv"
# 1) 210617_113644_PV_Siavash_walk_oscillations in phase
# 2) 210617_121732_PV_Siavash_walk_300_1600
# 3) 210617_122334_PV_Siavash_walk_500_2500
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
    'AnkleTorque': datatxt["ankMotTor"],
    'KneeTorque': datatxt["kneMotTor"],
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
mat = scipy.io.loadmat('OSL_walking_data/Treadmill_speed1_incline0_file1.mat')
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
    sys.Q = np.diag([0, 1e-7, 1e-7, 0])
    # measurement noise covariance
    sys.R = R['Generic'][np.ix_(sensor_id, sensor_id)]
    U = np.diag([1, 1, 1])
    sys.R = U @ sys.R @ U.T

    # initialize the state
    init = myStruct()
    init.x = np.array([[0], [0.4], [1.1], [0]])
    init.Sigma = np.diag([1e-3, 1e-3, 1e-3, 0])

    ekf = extended_kalman_filter(sys, init)

    ########## Create filters ################################################################
    fs = 1 / (dataOSL["Time"][1] - dataOSL["Time"][0])        # sampling rate = 100 Hz (actual: ~77 Hz)
    nyq = 0.5 * fs    # Nyquist frequency = fs/2
    ## configure low-pass filter (1-order)
    normal_cutoff = 2 / nyq   #cut-off frequency = 2Hz
    b_lp, a_lp = butter(1, normal_cutoff, btype = 'low', analog = False)
    z_lp_1 = lfilter_zi(b_lp,  a_lp)
    z_lp_2 = lfilter_zi(b_lp,  a_lp)
    
    ## configure band-pass filter (2-order)
    normal_lowcut = 0.5 / nyq    #lower cut-off frequency = 0.5Hz 
    normal_highcut = 2 / nyq     #upper cut-off frequency = 2Hz
    b_bp, a_bp = butter(2, [normal_lowcut, normal_highcut], btype = 'band', analog = False)
    z_bp = lfilter_zi(b_bp,  a_bp)

    ptr = 0    # for reading sensor data
    indx = 0   # for logging data
    null = 0   # number of null data points
    t_0 = dataOSL["Time"][ptr]   # for EKF
    start_time = t_0             # for live plotting
    fade_in_time = 3             # sec
    
    stride_peroid = np.array([0, 0, 0])
    steady_state = False
    heel_strike_time = []
    heel_strike_previous = start_time
    
    knee_angle_initial = dataOSL['KneeAngle'][0]
    ankle_angle_initial = dataOSL['AnkleAngle'][0]
    
    simulation_log = {
        # state estimates
        "phase_est": np.zeros((len(dataOSL["Time"]), 1)),
        "phase_dot_est": np.zeros((len(dataOSL["Time"]), 1)),
        "step_length_est": np.zeros((len(dataOSL["Time"]), 1)),
        "ramp_est": np.zeros((len(dataOSL["Time"]), 1)),
        
        "steady-state walking": np.zeros((len(dataOSL["Time"]), 1)),
        "MD": np.zeros((len(dataOSL["Time"]), 1)),

        # EKF prediction of measurements/ derived measurements
        "global_thigh_angle_pred": np.zeros((len(dataOSL["Time"]), 1)),
        "global_thigh_angle": np.zeros((len(dataOSL["Time"]), 1)),
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
        ankle_angle = dataOSL['AnkleAngle'][ptr]
        knee_angle = dataOSL['KneeAngle'][ptr]
        
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
        
        measurement = np.array([[global_thigh_angle], [global_thigh_angle_vel_lp], [Atan2]])
        measurement = np.squeeze(measurement)
        #measurement = measurement[sensors]

        ### EKF implementation
        ekf.prediction(dt)
        ekf.state_saturation(saturation_range)
        ekf.correction(measurement, Psi, arctan2, steady_state_walking = steady_state)
        ekf.state_saturation(saturation_range)

        # Detect steady-state waling
        phase_current = ekf.x[0, 0]
        if indx > 0:
            phase_previous = simulation_log['phase_est'][indx - 1]
        else:
            phase_previous = 0

        if phase_current < 0.05 and phase_previous > 0.95: # heel strike
            heel_strike_time.append(t)
            stride_peroid = np.array([t - heel_strike_previous, stride_peroid[0], stride_peroid[1]])
            heel_strike_previous = t
            steady_state = np.all(np.logical_and(stride_peroid > 0.5, stride_peroid < 2.5))
            
        if steady_state == True:
            ekf.Q = sys.Q
        else:
            ekf.Q[2, 2] = 0
            ekf.Q[3, 3] = 0
            ekf.x[2, 0] = 1.2
            ekf.x[3, 0] = 0

        ### Control commands: joints angles
        ## 1) Control commands generated by the trainned model
        joint_angles = joints_control(ekf.x[0, 0], ekf.x[1, 0], ekf.x[2, 0], ekf.x[3, 0])
        if steady_state == True:
            knee_angle_model = joint_angles[0]
            ankle_angle_model = joint_angles[1] # negative sign
        else:
            pv = int(dataOSL['PV'][indx])
            knee_angle_model = refKne[pv]
            ankle_angle_model = refAnk[pv]

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
        if steady_state == True:
            pv = int(ekf.x[0, 0] * 998)  # phase variable conversion (scaling)
        else:
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
        
        simulation_log["steady-state walking"][indx] = steady_state
        simulation_log["MD"][indx] = ekf.MD

        simulation_log["global_thigh_angle_pred"][indx] = ekf.z_hat[0,0]
        simulation_log["global_thigh_angle"][indx] = global_thigh_angle
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
    t_lower = 0
    t_upper = 35
    plt.figure("Gait Phase")
    plt.subplot(511)
    plt.title("EKF Gait State Estimate")
    plt.plot(dataOSL["Time"], simulation_log['phase_est'], 'r-')
    plt.plot(dataOSL["Time"], dataOSL['PV'] / 998, 'k-')
    plt.ylabel("Phase")
    plt.xlim((t_lower, t_upper))
    plt.legend(('EKF phase', 'phase variable'))

    plt.subplot(512)
    plt.plot(dataOSL["Time"], simulation_log["steady-state walking"], 'r-')
    plt.plot(np.array(heel_strike_time).reshape(-1), np.zeros((len(heel_strike_time), 1)), 'rx')
    plt.xlim((t_lower, t_upper))

    plt.subplot(513)
    plt.plot(dataOSL["Time"], simulation_log['phase_dot_est'], 'r-')
    plt.ylabel("Phase dot (1/s)")
    plt.xlim((t_lower, t_upper))
    plt.subplot(514)
    plt.plot(dataOSL["Time"], simulation_log['step_length_est'], 'r-')
    plt.ylabel("Stride Length (m)")
    plt.xlim((t_lower, t_upper))
    plt.subplot(515)
    plt.plot(dataOSL["Time"], simulation_log['ramp_est'], 'r-')
    plt.xlabel("Time (s)")
    plt.ylabel("Ramp (deg)")
    plt.xlim((t_lower, t_upper))

    plt.figure("Measurements")
    plt.subplot(311)
    plt.title("Measurements")
    plt.plot(dataOSL["Time"], simulation_log["global_thigh_angle"], 'k-')
    plt.plot(dataOSL["Time"], simulation_log["global_thigh_angle_pred"], 'r-')
    plt.legend(('actual', 'EKF predicted'))
    plt.ylabel("Global Thigh Angle (deg)")
    plt.xlim((t_lower, t_upper))
    plt.subplot(312)
    plt.plot(dataOSL["Time"], simulation_log["global_thigh_angle_vel"], 'k-')
    plt.plot(dataOSL["Time"], simulation_log["global_thigh_angle_vel_pred"], 'r-')
    plt.legend(('actual', 'EKF predicted'))
    plt.ylabel("Global Thigh Angle Vel (deg/s)")
    plt.xlim((t_lower, t_upper))
    plt.subplot(313)
    plt.plot(dataOSL["Time"], simulation_log["Atan2"], 'k-')
    plt.plot(dataOSL["Time"], simulation_log["Atan2_pred"], 'r-')
    plt.legend(('actual', 'EKF predicted'))
    plt.ylabel("Atan2")
    plt.xlabel("Time (s)")
    plt.xlim((t_lower, t_upper))

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
    plt.legend(('recorded', 'Edgar\'s trajectories', 'kinematic model'))
    plt.ylabel("Ankle angle command(deg)")
    plt.xlim((t_lower, t_upper))
    plt.subplot(313)
    plt.plot(dataOSL["Time"], dataOSL["KneeAngleRef"], 'k-')
    plt.plot(dataOSL["Time"], simulation_log["knee_angle_cmd"], 'r-')
    plt.plot(dataOSL["Time"], simulation_log["knee_angle_model"], 'm-')
    #plt.plot(dataOSL["Time"], dataOSL['KneeAngle'], 'b-')
    plt.legend(('recorded', 'Edgar\'s trajectories', 'kinematic model'))
    plt.ylabel("Knee angle command(deg)")
    plt.xlabel("Time (s)")
    plt.xlim((t_lower, t_upper))

    plt.figure("MD")
    plt.plot(dataOSL["Time"], simulation_log["MD"], 'r-')
    plt.ylabel("MD")
    plt.xlabel("Time (s)")
    plt.xlim((t_lower, t_upper))

    """
    plt.figure("Kinetics")
    plt.subplot(211)
    plt.plot(dataOSL["Time"], dataOSL['LoadCellFx'])
    plt.plot(dataOSL["Time"], dataOSL['LoadCellFy'])
    plt.plot(dataOSL["Time"], dataOSL['LoadCellFz'])
    plt.ylabel("Load Cell Force (?)")
    plt.xlabel("Time (s)")
    plt.legend(('Fx', 'Fy', 'Fz'))
    
    plt.subplot(212)
    plt.plot(dataOSL["Time"], dataOSL['LoadCellMx'])
    #plt.plot(dataOSL["Time"], dataOSL['LoadCellMy'])
    #plt.plot(dataOSL["Time"], dataOSL['LoadCellMz'])
    plt.plot(dataOSL["Time"], -dataOSL['AnkleTorque']*9.81)
    plt.plot(dataOSL["Time"], dataOSL['KneeTorque'])
    plt.ylabel("Load Cell Moment (?)")
    plt.xlabel("Time (s)")
    plt.legend(('Mx', 'Ankle Torque', 'Knee Torque'))
    """
    plt.show()
