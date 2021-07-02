"""
Code to simulate the phase-EKF and control commands using pre-revorded walking data
"""
import sys, time
import matplotlib.pyplot as plt
import numpy as np
import csv

from ekf import *
from model_framework import *
from model_fit import load_Psi
from scipy.signal import butter, lfilter, lfilter_zi
import sender_test as sender   # for real-time plotting

### Load pre-recorded walking data 
logFile = r"OSL_walking_data/210617_122334_PV_Siavash_walk_500_2500.csv"
# 210617_113644_PV_Siavash_walk_oscillations in phase
# 210617_121732_PV_Siavash_walk_300_1600
# 210617_122334_PV_Siavash_walk_500_2500
datatxt = np.genfromtxt(logFile , delimiter=',', names = True)
dataOSL = {
    "Time": datatxt["Time"],
    "ThighSagi": datatxt["ThighSagi"],
    "PV": datatxt['PV'],
    'AnkleAngle': datatxt["ankJoiPos"],
    'AnkleAngleRef': datatxt["refAnk"],
    'KneeAngle': datatxt["kneJoiPos"],
    'KneeAngleRef': datatxt["refKnee"],
    'AnkleTorque': datatxt["ankMotTor"],
}
#print(len(dataOSL["Time"]))

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

# Process model for the EKF
def A(dt):
    return np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

def process_model(x, dt):
    #dt = 0.01 # data sampling rate: 100 Hz
    return A(dt) @ x

try:
    ### Load reference trajectory
    refTrajectory = loadTrajectory(trajectory = 'walking')
    refAnk = refTrajectory["ankl"]
    refKne = refTrajectory["knee"]

    ### Intitialize EKF
    sensors = [0, 6, 7]
    arctan2 = False
    if sensors[-1] == 7:
        arctan2 = True
    with open('R_s.pickle', 'rb') as file:
        R = pickle.load(file)

    m_model = model_loader('Measurement_model_' + str(len(sensors)) +'_sp.pickle')
    Psi = load_Psi('Generic')[sensors]
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
    init.x = np.array([[0], [0.4], [1.1], [0]])
    init.Sigma = np.diag([1e-3, 1e-3, 1e-3, 0])

    ekf = extended_kalman_filter(sys, init)

    ########## Create filters ################################################################ 
    fs = 100          # sampling rate = 100 Hz (actual: ~77 Hz)
    nyq = 0.5 * fs    # Nyquist frequency = fs/2
    ## configure low-pass filter (1-order)
    normal_cutoff = 1 / nyq   #cut-off frequency = 2Hz
    b_lp, a_lp = butter(1, normal_cutoff, btype = 'low', analog = False)
    z_lp_1 = lfilter_zi(b_lp,  a_lp)
    z_lp_2 = lfilter_zi(b_lp,  a_lp)
    
    ## configure band-pass filter (2-order)
    normal_lowcut = 0.1 / nyq    #lower cut-off frequency = 0.5Hz """"
    normal_highcut = 1 / nyq     #upper cut-off frequency = 2Hz
    b_bp, a_bp = butter(2, [normal_lowcut, normal_highcut], btype = 'band', analog = False)
    z_bp = lfilter_zi(b_bp,  a_bp)

    if input('\n\nAbout to walk. Would you like to continue? (y/n): ').lower() == 'y':
        print("\n Let's walk!")
    else:
        sys.exit("User stopped the execution")
    
    ptr = 0
    t_0 = dataOSL["Time"][0]     # for EKF
    start_time = t_0             # for live plotting
    
    simulation_log = {
        # state estimates
        "phase_est": np.zeros((len(dataOSL["Time"]), 1)),
        "phase_dot_est": np.zeros((len(dataOSL["Time"]), 1)),
        "step_length_est": np.zeros((len(dataOSL["Time"]), 1)),
        "ramp_est": np.zeros((len(dataOSL["Time"]), 1)),

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
        
        measurement = np.array([[global_thigh_angle], [global_thigh_angle_vel_lp], [Atan2]]) #
        measurement = np.squeeze(measurement)
        #measurement = measurement[sensors]

        ### EKF implementation
        ekf.prediction(dt)
        ekf.state_saturation(saturation_range)
        ekf.correction(measurement, Psi, arctan2)
        ekf.state_saturation(saturation_range)
        
        ### Control commands: joints angles
        ## 1) Control commands generated by the trainned model
        joint_angles = joints_control(ekf.x[0, 0], ekf.x[1, 0], ekf.x[2, 0], ekf.x[3, 0])
        knee_angle_model = joint_angles[0]
        ankle_angle_model = -joint_angles[1] # negative sign
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
        pv = int(ekf.x[0, 0] * 998)  # phase variable conversion (scaling)
        ankle_angle_cmd = refAnk[pv]
        knee_angle_cmd = refKne[pv]
        
        ## Loggging simulation results
        simulation_log['phase_est'][ptr] = ekf.x[0, 0]
        simulation_log['phase_dot_est'][ptr] = ekf.x[1, 0]
        simulation_log['step_length_est'][ptr] = ekf.x[2, 0]
        simulation_log['ramp_est'][ptr] = ekf.x[3, 0]

        simulation_log["global_thigh_angle_pred"][ptr] = ekf.z_hat[0]
        simulation_log["global_thigh_angle"][ptr] = global_thigh_angle
        simulation_log["global_thigh_angle_vel_pred"][ptr] = ekf.z_hat[1]
        simulation_log["global_thigh_angle_vel"][ptr] = global_thigh_angle_vel_lp
        simulation_log["Atan2_pred"][ptr] = ekf.z_hat[2]
        simulation_log["Atan2"][ptr] = Atan2

        simulation_log["ankle_angle_model"][ptr] = ankle_angle_model
        simulation_log["ankle_angle_cmd"][ptr] = ankle_angle_cmd
        simulation_log["knee_angle_model"][ptr] = knee_angle_model
        simulation_log["knee_angle_cmd"][ptr] = knee_angle_cmd

        ### Live plotting
        
        elapsed_time = time.time() - start_time
        if ptr % 2 == 0:
            sender.graph(elapsed_time, 
                         global_thigh_angle, ekf.z_hat[0], 'Global Thigh Angle', 'deg',
                         #ekf.z_hat[0], 'Global Thigh Angle Pred', 'deg',
                         #global_thigh_angle_vel_lp, 'Global Thigh Angle Vel', 'deg/s',
                         #ekf.z_hat[1], 'Global Thigh Angle Vel Pred', 'deg/s'
                         #Atan2, 'atan2', '-',
                         #ekf.z_hat[2], 'atan2 Pred', '-'
                         #knee_angle, 'knee_angle', 'deg',
                         knee_angle_cmd, dataOSL["KneeAngleRef"][ptr], 'knee_angle', 'deg',
                         #knee_angle_model, 'knee_angle_model', 'deg',
                         ankle_angle_cmd, dataOSL["AnkleAngleRef"][ptr], 'ankle_angle', 'deg',
                         ekf.x[0, 0], dataOSL["PV"][ptr] / 998, 'phase', '-',
                         #ekf.x[1, 0], 'phase_dot', '1/s',
                         #ekf.x[2, 0], 'step_length', 'm',
                         #ekf.x[3, 0], 'ramp_angle', 'deg'
                         )
        
        ptr += 1
        if (ptr == len(dataOSL["Time"])):
            break
        
except KeyboardInterrupt:
    print('\n*** OSL shutting down ***\n')

finally:
    ## Plot the results
    plt.figure("Gait Phase")
    plt.subplot(411)
    plt.title("Gait State Estimate")
    plt.plot(dataOSL["Time"], simulation_log['phase_est'], 'r-')
    plt.plot(dataOSL["Time"], dataOSL['PV'] / 998, 'k-')
    plt.ylabel("Phase")
    plt.legend(('EKF phase', 'PV/998'))
    plt.subplot(412)
    plt.plot(dataOSL["Time"], simulation_log['phase_dot_est'], 'r-')
    plt.ylabel("Phase dot (1/s)")
    plt.subplot(413)
    plt.plot(dataOSL["Time"], simulation_log['step_length_est'], 'r-')
    plt.ylabel("Stride Length (m)")
    plt.subplot(414)
    plt.plot(dataOSL["Time"], simulation_log['ramp_est'], 'r-')
    plt.xlabel("Time (s)")
    plt.ylabel("Ramp (deg)")

    plt.figure("Measurements")
    plt.subplot(311)
    plt.title("Measurements")
    plt.plot(dataOSL["Time"], simulation_log["global_thigh_angle"], 'k-')
    plt.plot(dataOSL["Time"], simulation_log["global_thigh_angle_pred"], 'r-')
    plt.legend(('global thigh', 'global thigh pred'))
    plt.ylabel("Global Thigh Angle (deg)")
    plt.subplot(312)
    plt.plot(dataOSL["Time"], simulation_log["global_thigh_angle_vel"], 'k-')
    plt.plot(dataOSL["Time"], simulation_log["global_thigh_angle_vel_pred"], 'r-')
    plt.legend(('global thigh vel', 'global thigh vel pred'))
    plt.ylabel("Global Thigh Angle Vel (deg/s)")
    plt.subplot(313)
    plt.plot(dataOSL["Time"], simulation_log["Atan2"], 'k-')
    plt.plot(dataOSL["Time"], simulation_log["Atan2_pred"], 'r-')
    plt.legend(('Atan2', 'Atan2 pred'))
    plt.ylabel("Atan2")
    plt.xlabel("Time")

    plt.figure("Joints Angles")
    plt.subplot(211)
    plt.title("Joints Angles Commands")
    #plt.plot(dataOSL["Time"], dataOSL['AnkleAngle'], 'r--')
    plt.plot(dataOSL["Time"], dataOSL["AnkleAngleRef"], 'k-')
    plt.plot(dataOSL["Time"], simulation_log["ankle_angle_cmd"], 'r-')
    plt.plot(dataOSL["Time"], simulation_log["ankle_angle_model"], 'm--')
    #plt.plot(dataOSL["Time"], dataOSL["AnkleAngle"], 'g-')
    plt.legend(('recorded', 'simulated', 'kinematic_model'))
    plt.ylabel("Ankle angle command(deg)")
    plt.subplot(212)
    #plt.plot(dataOSL["Time"], dataOSL['KneeAngle'], 'r--')
    plt.plot(dataOSL["Time"], dataOSL["KneeAngleRef"], 'k-')
    plt.plot(dataOSL["Time"], simulation_log["knee_angle_cmd"], 'r-')
    plt.plot(dataOSL["Time"], simulation_log["knee_angle_model"], 'm--')
    #plt.plot(dataOSL["Time"], dataOSL["KneeAngle"], 'g-')
    plt.legend(('recorded', 'simulated', 'kinematic_model'))
    plt.ylabel("Knee angle command(deg)")
    plt.xlabel("Time")
    #plt.figure("Ankle Torque")
    #plt.plot(dataOSL["Time"], dataOSL["AnkleTorque"])

    plt.show()