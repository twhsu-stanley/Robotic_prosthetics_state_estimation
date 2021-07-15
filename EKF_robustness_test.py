from pickle import FALSE
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#import time
from EKF import *
from model_framework import *
from data_generators import *
from continuous_data import *
import csv

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

def ekf_test(subject, trial, side, kidnap = False, plot = False):
    dt = 1/100
    # load ground truth
    phases, phase_dots, step_lengths, ramps = Conti_state_vars(subject, trial, side)
    # load measurements
    global_thigh_angle_Y, force_z_ankle, force_x_ankle, moment_y_ankle, global_thigh_angVel_2hz, atan2\
                                        = load_Conti_measurement_data(subject, trial, side)

    #### Joint Control ############################################################
    knee_angle, ankle_angle = load_Conti_joints_angles(subject, trial, side)
    ### Load reference trajectory
    refTrajectory = loadTrajectory(trajectory = 'walking')
    refAnk = refTrajectory["ankl"]
    refKne = refTrajectory["knee"]
    ################################################################################

    z = np.array([[global_thigh_angle_Y],
                  [force_z_ankle],
                  [force_x_ankle],
                  [moment_y_ankle],
                  [global_thigh_angVel_2hz],
                  [atan2]])
    z = np.squeeze(z)
    z = z[sensor_id, :]

    Psi = np.array([load_Psi('Generic')[key] for key in sensors], dtype = object)
    saturation_range = Conti_maxmin(subject, plot = False)

    # build the system
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
    init.x = np.array([[phases[0]], [phase_dots[0]], [step_lengths[0]], [ramps[0]]])
    init.Sigma = np.diag([10, 10, 10, 0])

    ekf = extended_kalman_filter(sys, init)
    
    heel_strike_index = Conti_heel_strikes(subject, trial, side) - Conti_heel_strikes(subject, trial, side)[0]
    kidnap_index = np.random.randint(heel_strike_index[3, 0], heel_strike_index[4, 0]) # step at which kidnapping occurs
    print("kidnap_index(%) = ", (kidnap_index - heel_strike_index[3, 0])/(heel_strike_index[4, 0]- heel_strike_index[3, 0])*100)
    
    phase_kidnap =  np.random.uniform(0, 1)
    phase_dot_kidnap = np.random.uniform(0, 5) #(0.5, 1)#
    step_length_kidnap = np.random.uniform(0, 2) #(0.9, 1.5)#
    ramp_kidnap = np.random.uniform(-45, 45) #(-10, 10)#
    state_kidnap = np.array([[phase_kidnap], [phase_dot_kidnap], [step_length_kidnap], [ramp_kidnap]])
    print("state_kidnap = ", state_kidnap)

    total_step =  int(heel_strike_index[25, 0]) + 1 #np.shape(z)[1] #
    phases = phases[0 : total_step]
    phase_dots = phase_dots[0 : total_step]
    step_lengths = step_lengths[0 : total_step]
    ramps = ramps[0 : total_step]

    x = np.zeros((total_step, 4))  # state estimate
    z_pred = np.zeros((total_step, len(sensors)))
    Sigma_norm = np.zeros((total_step, 1))
    Sigma_diag = np.zeros((total_step, 4))
    Mahal_dist = np.zeros((total_step, 1))
    knee_angle_kmd = np.zeros((total_step, 1))
    ankle_angle_kmd = np.zeros((total_step, 1))
    knee_angle_cmd = np.zeros((total_step, 1))
    ankle_angle_cmd = np.zeros((total_step, 1))
    #t_step_max = 0
    for i in range(total_step):
        Sigma_norm[i] = np.linalg.norm(ekf.Sigma)
        # kidnap
        if kidnap != False and i == kidnap_index:
            ekf.x[kidnap] = state_kidnap[kidnap]
        
        ekf.prediction(dt)
        ekf.state_saturation(saturation_range)

        ekf.correction(z[:, i], Psi, arctan2)
        ekf.state_saturation(saturation_range)
        

        x[i,:] = ekf.x.T
        z_pred[i,:] = ekf.z_hat.T
        #Sigma_diag[i,:] = np.array([ekf.Sigma[k,k] for k in range(4)])
        Sigma_diag[i,:] = np.diag(ekf.Sigma)
        Mahal_dist[i] = ekf.MD

        ## Joints control commands 
        # 1) generated by the kinematic model
        joint_angles = joints_control(x[i,0], x[i,1], x[i,2], x[i,3])
        knee_angle_kmd[i] = joint_angles[0]
        ankle_angle_kmd[i] = joint_angles[1]
        # 2) generated by Edgar's prescribed trajectories
        pv = int(ekf.x[0, 0] * 998)  # phase variable conversion (scaling)
        ankle_angle_cmd[i] = refAnk[pv]
        knee_angle_cmd[i] = refKne[pv]

    # evaluate robustness
    # compare x and ground truth:
    track = True
    track_tol = 0.15
    start_check = kidnap_index + 1/0.8/dt*0.5 #5
    se = 0
    for i in range(total_step):
        error_phase = phase_error(x[i, 0], phases[i])
        se += error_phase ** 2
        if i >= start_check: #int(heel_strike_index[start_check]):
            track = track and (error_phase < track_tol)
            #if error_phase > track_tol:
            #    print(str(i/100)+": " + str(error_phase))
    
    RMSE_phase = np.sqrt(se / total_step)
    #track = track or (RMSE_phase < 0.1)
    print("RMSE phase = ", RMSE_phase)
    print("Final Sigma = ", Sigma_diag[-1, :])

    if kidnap != False:
        #phase_dot_akn = x[kidnap_index, 1]
        #phase_dot_b4kn = x[kidnap_index - 1, 1]
        #kidnap_step = kidnap_index / (heel_strike_index[1, 0] - heel_strike_index[0, 0]) * 100 # kidmap step % of stride
        #print("kidnapping step (%_stride) = ", kidnap_step)
        #print("phase_dot right after kidnap = ", phase_dot_akn)
        #print("phase_dot right before kidnap = ", phase_dot_b4kn)
        print("recover from kidnap? ", track)
        result = (track, RMSE_phase, step_lengths[0], ramps[0])
    elif kidnap == False:
        print("track without kidnapping? ", track)
        result = (track, RMSE_phase)
    
    if plot == True:
        th = heel_strike_index[0:25, 0].astype(int) # time step of heel strikes
        #nu = np.sqrt(13.277)
        # plot results
        tt = dt * np.arange(len(phases))
        plt.figure("State Estimate")
        plt.subplot(411)
        plt.title('EKF Robustness Test')
        plt.plot(tt, phases, 'k-')
        plt.plot(tt, x[:, 0], 'r--')
        #plt.plot(tt, phases + Sigma_diag[:, 0] * nu, 'b-')
        #plt.plot(tt, phases - Sigma_diag[:, 0] * nu, 'g-')
        #plt.plot(th* dt, np.zeros((len(th), 1)), "rx")
        plt.legend(('ground truth', 'estimate'))
        #plt.legend(('ground truth', 'estimate'), bbox_to_anchor=(1, 1.05))
        plt.ylabel('$\phi$')
        plt.ylim([0, 2])
        plt.xlim([0, tt[-1]+0.1])
        plt.subplot(412)
        plt.plot(tt, phase_dots, 'k-')
        plt.plot(tt, x[:, 1], 'r--')
        #plt.plot(tt, phase_dots + Sigma_diag[:, 1]*nu, 'b-')
        #plt.plot(tt, phase_dots - Sigma_diag[:, 1]*nu, 'g-')
        plt.ylabel('$\dot{\phi}~(s^{-1})$')
        plt.xlim([0, tt[-1]+0.1])
        #plt.ylim([0.5, 1.2])
        plt.subplot(413)
        plt.plot(tt, step_lengths, 'k-')
        plt.plot(tt, x[:, 2], 'r--')
        #plt.plot(tt, step_lengths + Sigma_diag[:, 2]*nu, 'b-')
        #plt.plot(tt, step_lengths - Sigma_diag[:, 2]*nu, 'g-')
        plt.ylabel('$l~(m)$')
        plt.xlim([0, tt[-1]+0.1])
        #plt.ylim([0, 2])
        plt.subplot(414)
        plt.plot(tt, ramps, 'k-')
        plt.plot(tt, x[:, 3], 'r--')
        #plt.plot(tt, ramps + Sigma_diag[:, 3]*nu, 'b-')
        #plt.plot(tt, ramps - Sigma_diag[:, 3]*nu, 'g-')
        plt.ylabel('$\\alpha~(deg)$')
        plt.xlabel('time (s)')
        plt.xlim([0, tt[-1]+0.1])
        #plt.ylim([-10, 10])

        plt.figure("Sigma")
        plt.title("Sigma Norms")
        plt.plot(Sigma_norm[th])
        plt.ylabel('$\Sigma$')
        plt.xlabel('heel strikes')

        plt.figure("Mahalanobis Distance")
        plt.title("Mahalanobis Distance")
        plt.plot(tt, Mahal_dist)
        plt.ylabel('MD')
        plt.xlabel('time (s)')

        plt.figure("Control Commands: Joint Angles")
        plt.title("Control Commands: Joint Angles")
        plt.subplot(211)
        plt.plot(tt, knee_angle[0:total_step], 'k-')
        plt.plot(tt, knee_angle_cmd, 'r-')
        plt.plot(tt, knee_angle_kmd, 'm-')
        plt.legend(('actual', 'Edgar\'s trajectory', 'kinematic model'))
        plt.ylabel('knee angle (deg)')
        plt.subplot(212)
        plt.plot(tt, ankle_angle[0:total_step], 'k-')
        plt.plot(tt, ankle_angle_cmd, 'r-')
        plt.plot(tt, ankle_angle_kmd, 'm-')
        plt.legend(('actual', 'Edgar\'s trajectory', 'kinematic model'))
        plt.ylabel('ankle angle (deg)')
        plt.xlabel('time (s)')

        
        plt.figure("Measurements")
        plt.subplot(411)
        plt.title("Measurements")
        plt.plot(tt, z[0, 0:total_step], 'k-')
        plt.plot(tt, z_pred[:, 0], 'r--')
        plt.legend(('actual', 'predicted'))
        plt.ylabel('$\\theta_Y$')
        #plt.ylim([-10, 50])
        plt.xlim([0, tt[-1]+0.1])
        plt.subplot(412)
        plt.plot(tt, z[1, 0:total_step], 'k-')
        plt.plot(tt, z_pred[:, 1], 'r--')
        #plt.ylabel('$f_Z$')
        #plt.ylim([0, 1500])
        plt.xlim([0, tt[-1]+0.1])
        plt.subplot(413)
        plt.plot(tt, z[2, 0:total_step], 'k-')
        plt.plot(tt, z_pred[:, 2], 'r--')
        #plt.ylabel('$f_X$')
        #plt.ylim([-500, 200])
        plt.xlim([0, tt[-1]+0.1])
        """
        plt.subplot(414)
        plt.plot(tt, z[3, 0:total_step], 'k-')
        plt.plot(tt, z_pred[:, 3], 'r--')
        plt.ylabel('$m_Y$')
        #plt.ylim([0, 2000])
        plt.xlim([0, tt[-1]+0.1])
        plt.xlabel("time (s)")
        
        plt.figure("Auxiliary Measurements")
        plt.subplot(411)
        plt.title("Original Measurements")
        plt.plot(tt, z[4, 0:total_step], 'k-')
        plt.plot(tt, z_pred[:, 4], 'r--')
        plt.legend(('actual', 'predicted'))
        plt.ylabel('$\dot{\\theta}_{Y_{5Hz}} ~(deg/s)$')
        #plt.ylim([-150, 150])
        plt.xlim([0, tt[-1]+0.1])
        plt.subplot(412)
        plt.plot(tt, z[5, 0:total_step], 'k-')
        plt.plot(tt, z_pred[:, 5], 'r--')
        plt.ylabel('$\dot{\\theta}_{Y_{2.5Hz}} ~(deg/s)$')
        #plt.ylim([-150, 150])
        plt.xlim([0, tt[-1]+0.1])
        
        plt.subplot(211)
        plt.plot(tt, z[4, 0:total_step], 'k-')
        plt.plot(tt, z_pred[:, 4], 'r--')
        plt.ylabel('$\dot{\\theta}_{Y_{2Hz}} ~(deg/s)$')
        #plt.ylim([-150, 150])
        plt.xlim([0, tt[-1]+0.1])
        plt.subplot(212)
        plt.plot(tt, z[5, 0:total_step], 'k-')
        plt.plot(tt, z_pred[:, 5], 'r--')
        plt.ylabel('$atan2$')
        plt.ylim([0, 10])
        plt.xlim([0, tt[-1]+0.1])
        plt.xlabel("time (s)")
        """
        plt.show()
    return result

def ekf_bank_test(subject, trial, side, N = 30, kidnap = [0,1,2,3], plot = True):
    # N: number of EKFs in the EKF-bank
    dt = 1/100
    # load ground truth
    phases, phase_dots, step_lengths, ramps = Conti_state_vars(subject, trial, side)
    # load measurements
    global_thigh_angle_Y, force_z_ankle, force_x_ankle, moment_y_ankle, global_thigh_angVel_2hz, atan2\
                                        = load_Conti_measurement_data(subject, trial, side)

    z = np.array([[global_thigh_angle_Y],
                  [force_z_ankle],
                  [force_x_ankle],
                  [moment_y_ankle],
                  [global_thigh_angVel_2hz],
                  [atan2]])
    z = np.squeeze(z)
    z = z[sensor_id, :]

    Psi = np.array([load_Psi('Generic')[key] for key in sensors], dtype = object)
    saturation_range = Conti_maxmin(subject, plot = False)

    # build the system
    sys = myStruct()
    sys.f = process_model
    sys.A = A
    sys.h = m_model
    sys.Q = np.diag([0, 1e-5, 1e-5, 1e-1])
    # measurement noise covariance
    sys.R = R['Generic'][np.ix_(sensor_id, sensor_id)]
    U = np.diag([2, 2, 2])
    sys.R = U @ sys.R @ U.T
    
    init = myStruct()

    heel_strike_index = Conti_heel_strikes(subject, trial, side) - Conti_heel_strikes(subject, trial, side)[0]
    total_step =  int(heel_strike_index[10, 0])+1 #np.shape(z)[1]
    
    # ground truth states
    phases = phases[0 : total_step]
    phase_dots = phase_dots[0 : total_step]
    step_lengths = step_lengths[0 : total_step]
    ramps = ramps[0 : total_step]
    
    #kidnap_index = 30 # step at which kidnapping occurs
    x = np.zeros((N, total_step, 4))  # state estimate
    phase_dot_ROC = np.zeros(N)
    phase_rakn = np.zeros(N)
    phase_dot_rakn = np.zeros(N)
    step_length_rakn = np.zeros(N)
    ramp_rakn = np.zeros(N)
    tf = np.zeros(N)
    M = 0
    for n in range(N):
        # initialize the state
        init.x = np.array([[phases[0]], [phase_dots[0]], [step_lengths[0]], [ramps[0]]])
        init.Sigma = np.diag([10, 10, 10, 100])
        # build EKF
        ekf = extended_kalman_filter(sys, init)
        
        kidnap_index = np.random.randint(heel_strike_index[3, 0], heel_strike_index[4, 0]) # step at which kidnapping occurs
        phase_kidnap = np.random.uniform(0, 1)
        phase_dot_kidnap = np.random.uniform(0, 5)
        step_length_kidnap = np.random.uniform(0, 2)#np.random.uniform(0, 2)
        ramp_kidnap = np.random.uniform(-45, 45)#np.random.uniform(-45, 45)
        state_kidnap = np.array([[phase_kidnap], [phase_dot_kidnap], [step_length_kidnap], [ramp_kidnap]])
    
        for i in range(total_step):
            # kidnap
            if i == kidnap_index:
                ekf.x[kidnap] = state_kidnap[kidnap]
            ekf.prediction(dt)
            ekf.state_saturation(saturation_range)
            ekf.correction(z[:, i], Psi, arctan2)
            ekf.state_saturation(saturation_range)
            x[n, i,:] = ekf.x.T
        
        phase_rakn[n] = state_kidnap[0]#x[n, kidnap_index, 0] #- phases[kidnap_index]
        phase_dot_rakn[n] = state_kidnap[1]#x[n, kidnap_index, 1] #- phase_dots[kidnap_index]
        step_length_rakn[n] = state_kidnap[2]#x[n, kidnap_index, 2] #- step_lengths[kidnap_index]
        ramp_rakn[n] = state_kidnap[3]#x[n, kidnap_index, 3] #- ramps[kidnap_index]
        phase_dot_ROC[n] = x[n, -1, 1]

        # evaluate robustness
        track = True
        track_tol = 0.15
        start_check = kidnap_index + 1/0.8/dt*0.5 #5
        se = 0
        for i in range(total_step):
            error_phase = phase_error(x[n, i, 0], phases[i])
            se += error_phase ** 2
            if i >= start_check: #int(heel_strike_index[start_check]):
                track = track and (error_phase < track_tol)
        RMSE_phase = np.sqrt(se / total_step)
        #track = track or (RMSE_phase < 0.1)
        tf[n] = track
        print(track)
        if track:
            M += 1

    robustness = M / N * 100
    print("Robustness (%) = ", robustness)

    if plot == True:
        # plot results
        tt = dt * np.arange(len(phases))
        plt.figure("phase")
        plt.plot(tt, phases, 'k-', linewidth=3)
        plt.plot(tt,  x[:, :, 0].T, 'b--', linewidth=1)
        plt.ylabel('$\phi$')
        plt.legend(('ground truth', 'estimate'))
        #plt.legend(('ground truth', 'estimate'), bbox_to_anchor=(1, 1.05))
        plt.ylim([0, 1.4])
        plt.xlim([0, tt[-1]+0.1])
        plt.xlabel('time (s)')

        plt.figure("states")
        plt.subplot(411)
        plt.title('EKFs-Bank Test')
        plt.plot(tt, phases, 'k--', linewidth=2)
        plt.plot(tt,  x[:, :, 0].T, '--')#,alpha = 0.35
        plt.ylabel('$\phi$')
        plt.legend(('ground truth', 'estimate'))
        #plt.legend(('ground truth', 'estimate'), bbox_to_anchor=(1, 1.05))
        plt.ylim([0, 2])
        plt.xlim([0, tt[-1]+0.1])
        plt.subplot(412)
        plt.plot(tt, phase_dots, 'k-', linewidth=2)
        plt.plot(tt, x[:, :, 1].T, '--')
        plt.ylabel('$\dot{\phi}~(s^{-1})$')
        plt.xlim([0, tt[-1]+0.1])
        plt.subplot(413)
        plt.plot(tt, step_lengths, 'k-', linewidth=2)
        plt.plot(tt, x[:, :, 2].T, '--')
        plt.ylabel('$l~(m)$')
        plt.xlim([0, tt[-1]+0.1])
        plt.subplot(414)
        plt.plot(tt, ramps, 'k-', linewidth=2)
        plt.plot(tt, x[:, :, 3].T, '--')
        plt.ylabel('$\\alpha~(deg)$')
        plt.xlim([0, tt[-1]+0.1])
        plt.xlabel('time (s)')

        plt.figure("phase_dot cluster")
        plt.hist(phase_dot_ROC)
        plt.xlabel('phase_dot in the end')
        plt.ylabel('counts')

        plt.figure("Region of attraction_1")
        ax = plt.axes(projection='3d')
        for n in range(N):
            #if phase_dot_ROC[n] < phase_dots[-1] + 0.2 and phase_dot_ROC[n] > phase_dots[-1] - 0.2:
            if tf[n] == True:
                ax.scatter3D(phase_rakn[n], phase_dot_rakn[n], step_length_rakn[n], c = 'r')
            #elif phase_dot_ROC[n] < 0.2 and phase_dot_ROC[n] > - 0.3:
            else:
                ax.scatter3D(phase_rakn[n], phase_dot_rakn[n], step_length_rakn[n], c = 'b')
            #else:
            #    ax.scatter3D(phase_rakn[n], phase_dot_rakn[n], step_length_rakn[n], c = 'k')
        ax.set_xlabel('phase right after kidnapping')
        ax.set_ylabel('phase_dot right after kidnapping')
        ax.set_zlabel('step_length right after kidnapping')

        plt.figure("Region of attraction_2")
        ax = plt.axes(projection='3d')
        for n in range(N):
            #if phase_dot_ROC[n] < phase_dots[-1] + 0.2 and phase_dot_ROC[n] > phase_dots[-1] - 0.2:
            if tf[n] == True:
                ax.scatter3D(phase_rakn[n], phase_dot_rakn[n], ramp_rakn[n], c = 'r')
            #elif phase_dot_ROC[n] < 0.2 and phase_dot_ROC[n] > - 0.3:
            else:
                ax.scatter3D(phase_rakn[n], phase_dot_rakn[n], ramp_rakn[n], c = 'b')
            #else:
            #    ax.scatter3D(phase_rakn[n], phase_dot_rakn[n], ramp_rakn[n], c = 'k')
        ax.set_xlabel('phase right after kidnapping')
        ax.set_ylabel('phase_dot right after kidnapping')
        ax.set_zlabel('ramp right after kidnapping')

        plt.figure("Region of attraction_3")
        ax = plt.axes(projection='3d')
        for n in range(N):
            #if phase_dot_ROC[n] < phase_dots[-1] + 0.2 and phase_dot_ROC[n] > phase_dots[-1] - 0.2:
            if tf[n] == True:
                ax.scatter3D(phase_rakn[n], ramp_rakn[n], step_length_rakn[n], c = 'r')
            #elif phase_dot_ROC[n] < 0.2 and phase_dot_ROC[n] > - 0.3:
            else:
                ax.scatter3D(phase_rakn[n], ramp_rakn[n], step_length_rakn[n], c = 'b')
            #else:
            #    ax.scatter3D(phase_rakn[n], ramp_rakn[n], step_length_rakn[n], c = 'k')
        ax.set_xlabel('phase right after kidnapping')
        ax.set_ylabel('ramp right after kidnapping')
        ax.set_zlabel('step_length right after kidnapping')

        plt.figure("Region of attraction_4")
        ax = plt.axes(projection='3d')
        for n in range(N):
            #if phase_dot_ROC[n] < phase_dots[-1] + 0.2 and phase_dot_ROC[n] > phase_dots[-1] - 0.2:
            if tf[n] == True:
                ax.scatter3D(ramp_rakn[n], phase_dot_rakn[n], step_length_rakn[n], c = 'r')
            #elif phase_dot_ROC[n] < 0.2 and phase_dot_ROC[n] > - 0.3:
            else:
                ax.scatter3D(ramp_rakn[n], phase_dot_rakn[n], step_length_rakn[n], c = 'b')
            #else:
            #    ax.scatter3D(ramp_rakn[n], phase_dot_rakn[n], step_length_rakn[n], c = 'k')
        ax.set_xlabel('ramp right after kidnapping')
        ax.set_ylabel('phase_dot right after kidnapping')
        ax.set_zlabel('step_length right after kidnapping')

        plt.show()
    
    return robustness

def ekf_robustness(kidnap = True):
    track_count = 0
    total_trials = 0
    RMSerror_phase = []

    robustness = 0

    with open('Continuous_data/GlobalThighAngles_with_Nan.pickle', 'rb') as file:
        nan_dict = pickle.load(file)

    #for subject in Conti_subject_names():
    for subject in ['AB06']: # , 'AB02', 'AB03', 'AB08', 'AB09', 'AB10'
        print("subject: ", subject)
        for trial in Conti_trial_names(subject):
        #for trial in ['s1x2d2x5', 's1x2d7x5', 's0x8i10', 's0x8i5']:
            if trial == 'subjectdetails':
                continue
            print("trial: ", trial)
            for side in ['left']:
                if nan_dict[subject][trial][side] == False:
                    print(subject + "/"+ trial + "/"+ side+ ": Trial skipped!")
                    continue
                #print("side: ", side)
                total_trials = total_trials + 1
                
                if kidnap == True:
                    #track, RMSE_phase, step_length, ramp = ekf_test(subject, trial, side, kidnap, plot = False)
                    #RMSerror_phase.append([RMSE_phase, step_length, ramp])
                    robustness += ekf_bank_test(subject, trial, side, N = 30, plot = False)
                else:
                    track, RMSE_phase = ekf_test(subject, trial, side, kidnap, plot = False)
                    RMSerror_phase.append([RMSE_phase])
                
                #if  track == True:
                #    track_count = track_count + 1

    #robustness = track_count / total_trials * 100
    robustness = robustness / total_trials
    print("==========================================")
    print("Overall Average Robustness (%) = ", robustness)

    """
    if kidnap == True:
        RMSerror_phase = np.array(RMSerror_phase).reshape(-1, 3)
        RMSerror_phase_df = pd.DataFrame(RMSerror_phase, columns = ['RMSE', 'x', 'y'])
        sns.heatmap(RMSerror_phase_df.pivot('y', 'x', 'RMSE'))
        plt.title("RMSE of phase")
        plt.xlabel("step_length")
        plt.ylabel("ramp")
        plt.ylim((-10, 10))
        plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
        plt.show()
    else:
        pass
        # heatmap for normal test
    """
    return robustness

if __name__ == '__main__':
    subject = 'AB02'
    trial = 's1i0'
    side = 'left'

    ekf_test(subject, trial, side, kidnap = False, plot = True)
    #ekf_bank_test(subject, trial, side, N = 20, kidnap = [0, 1, 2, 3], plot = True)
    #ekf_robustness(kidnap = True)
    #print(np.diag(R[subject]))
