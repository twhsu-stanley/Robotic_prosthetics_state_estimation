from pickle import FALSE
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#import time
from EKF import *
from model_framework import *
from continuous_data import *
from basis_model_fitting import measurement_noise_covariance, saturation_bounds
import csv

# Dictionary of the sensors
sensors_dict = {'globalThighAngles': 0, 'ankleMoment': 1, 'globalThighVelocities': 2, 'atan2': 3}

# Determine which sensors to be used
sensors = ['globalThighAngles', 'ankleMoment', 'globalThighVelocities', 'atan2']
sensor_id = [sensors_dict[key] for key in sensors]

sensor_id_str = ""
for i in range(len(sensor_id)):
    sensor_id_str += str(sensor_id[i])
m_model = model_loader('Measurement_model_' + sensor_id_str +'.pickle')

using_atan2 = False
if sensors[-1] == 'atan2':
    using_atan2 = True

tibiaForce_threshold = -1

Psi = np.array([load_Psi('Generic')[key] for key in sensors], dtype = object)

dt = 1/100
Q = np.diag([0, 1e-3, 1e-3, 4]) * dt
U = np.diag([2, 2, 2, 1])
R = U @ measurement_noise_covariance(*sensors) @ U.T
saturation_range = saturation_bounds()

using_ankleMoment = False
if sensors[1] == 'ankleMoment':
    using_ankleMoment = True
    sensors_swing = []
    for i in range(len(sensors)):
        if sensors[i] == 'ankleMoment':
            continue
        else:
            sensors_swing.append(sensors[i])
    sensor_swing_id = [sensors_dict[key] for key in sensors_swing]
    sensor_swing_id_str = ""
    for i in range(len(sensor_swing_id)):
        sensor_swing_id_str += str(sensor_swing_id[i])
    Psi_swing = np.array([load_Psi('Generic')[key] for key in sensors_swing], dtype = object)
    m_model_swing = model_loader('Measurement_model_' + sensor_swing_id_str +'.pickle')
    U_swing = np.diag(np.diag(U)[sensor_swing_id])
    R_swing = U_swing @ measurement_noise_covariance(*sensors_swing) @ U_swing.T

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
    globalThighAngle, ankleMoment, tibiaForce, globalThighVelocity, atan2 = load_Conti_measurement_data(subject, trial, side)

    #### Joint Control ############################################################
    knee_angle, ankle_angle = load_Conti_joints_angles(subject, trial, side)
    ### Load reference trajectory
    refTrajectory = loadTrajectory(trajectory = 'walking')
    refAnk = refTrajectory["ankl"]
    refKne = refTrajectory["knee"]
    ################################################################################

    z_full = np.array([[globalThighAngle], [ankleMoment], [globalThighVelocity], [atan2]])
    z_full = np.squeeze(z_full)
    z = z_full[sensor_id, :]

    # build the system
    sys = myStruct()
    sys.f = process_model
    sys.A = A
    sys.h = m_model
    sys.Q = Q
    sys.R = R

    # initialize the state
    init = myStruct()
    init.x = np.array([[phases[0]], [phase_dots[0]], [step_lengths[0]], [ramps[0]]])
    #init.x = np.array([[0.5], [0], [0], [45]])
    init.Sigma = np.diag([1e-3, 1e-3, 1e-3, 1e-3])

    ekf = extended_kalman_filter(sys, init)
    
    heel_strike_index = Conti_heel_strikes(subject, trial, side) - Conti_heel_strikes(subject, trial, side)[0]
    
    if kidnap != False:
        kidnap_index = np.random.randint(heel_strike_index[3, 0], heel_strike_index[4, 0]) # step at which kidnapping occurs
        #print("kidnap_index(%) = ", (kidnap_index - heel_strike_index[3, 0])/(heel_strike_index[4, 0]- heel_strike_index[3, 0])*100)
        phase_kidnap =  np.random.uniform(0, 1)
        phase_dot_kidnap = np.random.uniform(0, 5)
        step_length_kidnap = np.random.uniform(0, 2)
        ramp_kidnap = np.random.uniform(-45, 45)
        state_kidnap = np.array([[phase_kidnap], [phase_dot_kidnap], [step_length_kidnap], [ramp_kidnap]])
        print("state_kidnap = [%4.2f, %4.2f, %4.2f, %5.2f]" % (state_kidnap[0], state_kidnap[1], state_kidnap[2], state_kidnap[3]))

    total_step =  int(heel_strike_index[20, 0]) + 1
    phases = phases[0 : total_step]
    phase_dots = phase_dots[0 : total_step]
    step_lengths = step_lengths[0 : total_step]
    ramps = ramps[0 : total_step]

    x = np.zeros((total_step, 4))  # state estimate
    z_pred = np.zeros((total_step, len(sensors)))
    
    Q_diag = np.zeros((total_step, 4))
    Sigma_diag = np.zeros((total_step, 4))
    
    estimate_error = np.zeros((total_step, 4))
    #MD_residual = np.zeros((total_step, 1))
    #MD_estimate = np.zeros((total_step, 1))
    
    knee_angle_kmd = np.zeros((total_step, 1))
    ankle_angle_kmd = np.zeros((total_step, 1))
    knee_angle_cmd = np.zeros((total_step, 1))
    ankle_angle_cmd = np.zeros((total_step, 1))
    #t_step_max = 0
    for i in range(total_step):
        if kidnap != False:
            if i == kidnap_index:
                ekf.x[kidnap] = state_kidnap[kidnap]
        
        ekf.prediction(dt)
        ekf.state_saturation(saturation_range)

        if using_ankleMoment:
            if tibiaForce[i] <= tibiaForce_threshold:
                ekf.h = m_model
                ekf.R = R
                ekf.correction(z[:, i], Psi, using_atan2, steady_state_walking = True)
                z_pred[i,:] = ekf.z_hat.T
            else:
                ekf.h = m_model_swing
                ekf.R = R_swing
                ekf.correction(z_full[sensor_swing_id, i], Psi_swing, using_atan2, steady_state_walking = True)
                z_pred[i,sensor_swing_id] = ekf.z_hat.T
        else:
            ekf.correction(z[:, i], Psi, using_atan2, steady_state_walking = True)
            z_pred[i,:] = ekf.z_hat.T
        ekf.state_saturation(saturation_range)

        x[i,:] = ekf.x.T

        Q_diag[i,:] = np.diag(ekf.Q)
        Sigma_diag[i, :] = np.diag(ekf.Sigma)
        #MD_residual[i] = ekf.MD_residual

        estimate_error[i, :] = (ekf.x - np.array([[phases[i]], [phase_dots[i]], [step_lengths[i]], [ramps[i]]])).reshape(-1)
        estimate_error[i, 0] = phase_error(ekf.x[0, 0], phases[i])
        #MD_estimate[i] = np.sqrt(estimate_error[i, :].T @ np.linalg.inv(ekf.Sigma) @ estimate_error[i, :])

        ## Joints control commands 
        # 1) generated by the kinematic model
        joint_angles = joints_control(x[i,0], x[i,1], x[i,2], x[i,3])
        knee_angle_kmd[i] = joint_angles[0]
        ankle_angle_kmd[i] = joint_angles[1]
        # 2) generated by Edgar's prescribed trajectories
        pv = int(ekf.x[0, 0] * 998)  # phase variable conversion (scaling)
        ankle_angle_cmd[i] = refAnk[pv]
        knee_angle_cmd[i] = refKne[pv]

    if kidnap != False:
        phase_recover = np.all(abs(estimate_error[int(kidnap_index + 1.5/np.average(phase_dots)/dt):, 0]) < 0.2)
        step_length_recover = np.all(abs(estimate_error[int(kidnap_index + 3/np.average(phase_dots)/dt):, 2]) < 0.3)
        ramp_recover = np.all(abs(estimate_error[int(kidnap_index + 3/np.average(phase_dots)/dt):, 3]) < 4)
        print("phase recover:", phase_recover, "; step length recover:", step_length_recover, "; ramp recover:", ramp_recover)

        RMSE_start_idx = int(kidnap_index + 3/np.average(phase_dots)/dt)
        RMSE_end_idx = int(kidnap_index + 13/np.average(phase_dots)/dt)
        RMSE_phase = np.sqrt((estimate_error[RMSE_start_idx:RMSE_end_idx, 0] ** 2).mean())
        RMSE_phase_dot = np.sqrt((estimate_error[RMSE_start_idx:RMSE_end_idx, 1] ** 2).mean())
        RMSE_step_length = np.sqrt((estimate_error[RMSE_start_idx:RMSE_end_idx, 2] ** 2).mean())
        RMSE_ramp = np.sqrt((estimate_error[RMSE_start_idx:RMSE_end_idx, 3] ** 2).mean())
        print("RMSE 3-13 strides after kidnapping ===============")
        print("RMSE phase = %5.3f" % RMSE_phase)
        print("RMSE phase_dot = %5.3f" % RMSE_phase_dot)
        print("RMSE step_length = %5.3f" % RMSE_step_length)
        print("RMSE ramp = %5.3f" % RMSE_ramp)

        track = (phase_recover and step_length_recover and ramp_recover and
                 RMSE_phase < 0.05 and RMSE_step_length < 0.2 and RMSE_ramp < 2)
        
        print("Recover from kidnapping?", track)

        result = (track, RMSE_phase)
    else:
        start_check_idx = int(3/np.average(phase_dots)/dt)
        RMSE_phase = np.sqrt((estimate_error[start_check_idx:, 0] ** 2).mean()) 
        RMSE_phase_dot = np.sqrt((estimate_error[start_check_idx:, 1] ** 2).mean()) 
        RMSE_step_length = np.sqrt((estimate_error[start_check_idx:, 2] ** 2).mean()) 
        RMSE_ramp = np.sqrt((estimate_error[start_check_idx:, 3] ** 2).mean()) 
        print("RMSE phase = %5.3f" % RMSE_phase)
        print("RMSE phase_dot = %5.3f" % RMSE_phase_dot)
        print("RMSE step_length = %5.3f" % RMSE_step_length)
        print("RMSE ramp = %5.3f" % RMSE_ramp)
        
        result = RMSE_phase

    if plot == True:
        #th = heel_strike_index[0:25, 0].astype(int) # time step of heel strikes
        nu = np.sqrt(18.5)
        # plot results
        tt = dt * np.arange(len(phases))
        plt.figure("State Estimate")
        plt.subplot(411)
        plt.title('EKF Robustness Test')
        plt.plot(tt, phases, 'k-')
        plt.plot(tt, x[:, 0], 'r--')
        #plt.plot(tt, x[:, 0] + Sigma_diag[:, 0]*nu, 'b-')
        #plt.plot(tt, x[:, 0] - Sigma_diag[:, 0]*nu, 'g-')
        #plt.plot(th* dt, np.zeros((len(th), 1)), "rx")
        plt.legend(('ground truth', 'estimate'))
        #plt.legend(('ground truth', 'estimate'), bbox_to_anchor=(1, 1.05))
        plt.ylabel('$\phi$')
        plt.ylim([0, 2])
        plt.xlim([0, tt[-1]+0.1])
        plt.subplot(412)
        plt.plot(tt, phase_dots, 'k-')
        plt.plot(tt, x[:, 1], 'r--')
        #plt.plot(tt, x[:, 1] + Sigma_diag[:, 1]*nu, 'b-')
        #plt.plot(tt, x[:, 1] - Sigma_diag[:, 1]*nu, 'g-')
        plt.ylabel('$\dot{\phi}~(s^{-1})$')
        plt.xlim([0, tt[-1]+0.1])
        plt.ylim([0.5, 1.5])
        plt.subplot(413)
        plt.plot(tt, step_lengths, 'k-')
        plt.plot(tt, x[:, 2], 'r--')
        #plt.plot(tt, x[:, 2] + Sigma_diag[:, 2]*nu, 'b-')
        #plt.plot(tt, x[:, 2] - Sigma_diag[:, 2]*nu, 'g-')
        plt.ylabel('$l~(m)$')
        plt.xlim([0, tt[-1]+0.1])
        plt.ylim([0.5, 1.6])
        plt.subplot(414)
        plt.plot(tt, ramps, 'k-')
        plt.plot(tt, x[:, 3], 'r--')
        #plt.plot(tt, x[:, 3] + Sigma_diag[:, 3]*nu, 'b-')
        #plt.plot(tt, x[:, 3] - Sigma_diag[:, 3]*nu, 'g-')
        plt.ylabel('$\\alpha~(deg)$')
        plt.xlabel('time (s)')
        plt.xlim([0, tt[-1]+0.1])
        plt.ylim([-15, 15])

        plt.figure("Estimation Errors")
        plt.subplot(411)
        plt.title('Estimation Errors')
        plt.plot(tt, abs(estimate_error[:, 0].T))
        plt.ylabel('$\phi$ error')
        plt.ylim([0, 0.5])
        plt.xlim([0, tt[-1]+0.1])
        plt.subplot(412)
        plt.plot(tt, abs(estimate_error[:, 1].T))
        plt.ylabel('$\dot{\phi}$ error (1/s)')
        plt.xlim([0, tt[-1]+0.1])
        plt.subplot(413)
        plt.plot(tt, abs(estimate_error[:, 2].T))
        plt.ylabel('$l$ error (m)')
        plt.xlim([0, tt[-1]+0.1])
        plt.ylim([0, 1])
        plt.subplot(414)
        plt.plot(tt, abs(estimate_error[:, 3].T))
        plt.ylabel('$\\alpha$ error (deg)')
        plt.xlim([0, tt[-1]+0.1])
        #plt.ylim([5, 0.5])
        plt.xlabel('time (s)')

        """
        plt.figure("Process Noise Covariance")
        plt.subplot(411)
        plt.title("diag Q")
        plt.plot(tt, Q_diag[:, 0])
        plt.ylabel('$Q(1,1)$')
        plt.subplot(412)
        plt.plot(tt, Q_diag[:, 1])
        plt.ylabel('$Q(2,2)$')
        plt.subplot(413)
        plt.plot(tt, Q_diag[:, 2])
        plt.ylabel('$Q(3,3)$')
        plt.subplot(414)
        plt.plot(tt, Q_diag[:, 3])
        plt.ylabel('$Q(4,4)$')
        plt.xlabel('time (s)')
        
        plt.figure("Mahalanobis Distance of Residuals")
        plt.title("Mahalanobis Distance of Residuals")
        plt.plot(tt, MD_residual)
        plt.ylabel('MD of Residuals')
        plt.xlabel('time (s)')
        
        plt.figure("Mahalanobis Distance of State Estimate")
        plt.title("Mahalanobis Distance of State Estimate")
        plt.plot(tt, MD_estimate)
        plt.plot(tt, nu * np.ones((len(tt), 1)))
        plt.ylabel('MD of State Estimate')
        plt.xlabel('time (s)')
        """

        plt.figure("Control Commands: Joint Angles")
        plt.title("Control Commands: Joint Angles")
        plt.subplot(211)
        plt.plot(tt, knee_angle[0:total_step], 'k-')
        #plt.plot(tt, knee_angle_cmd, 'r-')
        plt.plot(tt, knee_angle_kmd, 'm-')
        plt.legend(('actual', 'kinematic model', 'Edgar\'s trajectory'))
        plt.ylabel('knee angle (deg)')
        plt.subplot(212)
        plt.plot(tt, ankle_angle[0:total_step], 'k-')
        #plt.plot(tt, ankle_angle_cmd, 'r-')
        plt.plot(tt, ankle_angle_kmd, 'm-')
        plt.legend(('actual', 'kinematic model', 'Edgar\'s trajectory'))
        plt.ylabel('ankle angle (deg)')
        plt.xlabel('time (s)')

        plt.figure("Measurements")
        plt.subplot(411)
        plt.title("Measurements")
        plt.plot(tt, z[0, 0:total_step], 'k-')
        plt.plot(tt, z_pred[:, 0], 'r--')
        plt.legend(('actual', 'predicted'))
        plt.xlim([0, tt[-1]+0.1])
        plt.subplot(412)
        plt.plot(tt, z[1, 0:total_step], 'k-')
        plt.plot(tt, z_pred[:, 1], 'r--')
        plt.xlim([0, tt[-1]+0.1])
        plt.subplot(413)
        plt.plot(tt, z[2, 0:total_step], 'k-')
        plt.plot(tt, z_pred[:, 2], 'r--')
        plt.xlim([0, tt[-1]+0.1])
        plt.subplot(414)
        plt.plot(tt, z[3, 0:total_step], 'k-')
        plt.plot(tt, z_pred[:, 3], 'r--')
        plt.xlim([0, tt[-1]+0.1])
        plt.xlabel("time (s)")
        
        plt.show()
    return result

def ekf_bank_test(subject, trial, side, N = 30, kidnap = [0, 1, 2, 3], plot = True):
    # N: number of EKFs in the EKF-bank
    print("Monte-Carlo Test: ", subject, "/", trial, '/', side)

    # load ground truth
    phases, phase_dots, step_lengths, ramps = Conti_state_vars(subject, trial, side)
    # load measurements
    globalThighAngle, ankleMoment, tibiaForce, globalThighVelocity, atan2 = load_Conti_measurement_data(subject, trial, side)

    z_full = np.array([[globalThighAngle], [ankleMoment], [globalThighVelocity], [atan2]])
    z_full = np.squeeze(z_full)
    z = z_full[sensor_id, :]

    # build the system
    sys = myStruct()
    sys.f = process_model
    sys.A = A
    sys.h = m_model
    sys.Q = Q
    sys.R = R
    
    init = myStruct()

    heel_strike_index = Conti_heel_strikes(subject, trial, side) - Conti_heel_strikes(subject, trial, side)[0]
    total_step =  int(heel_strike_index[15, 0]) + 1
    
    # ground truth states
    phases = phases[0 : total_step]
    phase_dots = phase_dots[0 : total_step]
    step_lengths = step_lengths[0 : total_step]
    ramps = ramps[0 : total_step]
    
    #kidnap_index = 30 # step at which kidnapping occurs
    x = np.zeros((N, total_step, 4))  # state estimate
    estimate_error = np.zeros((N, total_step, 4))
    M = 0
    for n in range(N):
        # initialize the state
        init.x = np.array([[phases[0]], [phase_dots[0]], [step_lengths[0]], [ramps[0]]])
        init.Sigma = np.diag([1e-3, 1e-3, 1e-3, 1e-3])
        # build EKF
        ekf = extended_kalman_filter(sys, init)
        
        kidnap_index = np.random.randint(heel_strike_index[3, 0], heel_strike_index[4, 0]) # step at which kidnapping occurs
        phase_kidnap = np.random.uniform(0, 1)
        phase_dot_kidnap = np.random.uniform(0, 5)
        step_length_kidnap = np.random.uniform(0, 2)
        ramp_kidnap = np.random.uniform(-45, 45)
        state_kidnap = np.array([[phase_kidnap], [phase_dot_kidnap], [step_length_kidnap], [ramp_kidnap]])

        for i in range(total_step):
            # kidnap
            if i == kidnap_index:
                ekf.x[kidnap] = state_kidnap[kidnap]
            
            ekf.prediction(dt)
            ekf.state_saturation(saturation_range)

            if using_ankleMoment:
                if tibiaForce[i] <= tibiaForce_threshold:
                    ekf.h = m_model
                    ekf.R = R
                    ekf.correction(z[:, i], Psi, using_atan2, steady_state_walking = True)
                else:
                    ekf.h = m_model_swing
                    ekf.R = R_swing
                    ekf.correction(z_full[sensor_swing_id, i], Psi_swing, using_atan2, steady_state_walking = True)
            else:
                ekf.correction(z[:, i], Psi, using_atan2, steady_state_walking = True)
            ekf.state_saturation(saturation_range)

            x[n, i,:] = ekf.x.T

            estimate_error[n, i, :] = (ekf.x - np.array([[phases[i]], [phase_dots[i]], [step_lengths[i]], [ramps[i]]])).reshape(-1)
            estimate_error[n, i, 0] = phase_error(ekf.x[0, 0], phases[i])

        phase_recover = np.all(abs(estimate_error[n, int(kidnap_index + 1.5/np.average(phase_dots)/dt):, 0]) < 0.2)
        step_length_recover = np.all(abs(estimate_error[n, int(kidnap_index + 3/np.average(phase_dots)/dt):, 2]) < 0.3)
        ramp_recover = np.all(abs(estimate_error[n, int(kidnap_index + 3/np.average(phase_dots)/dt):, 3]) < 4)

        RMSE_start_idx = int(kidnap_index + 3/np.average(phase_dots)/dt)
        RMSE_end_idx = int(kidnap_index + 13/np.average(phase_dots)/dt)
        RMSE_phase = np.sqrt((estimate_error[n, RMSE_start_idx:RMSE_end_idx, 0] ** 2).mean())
        RMSE_step_length = np.sqrt((estimate_error[n, RMSE_start_idx:RMSE_end_idx, 2] ** 2).mean())
        RMSE_ramp = np.sqrt((estimate_error[n, RMSE_start_idx:RMSE_end_idx, 3] ** 2).mean())

        track = (phase_recover and step_length_recover and ramp_recover and
                 RMSE_phase < 0.05 and RMSE_step_length < 0.2 and RMSE_ramp < 2)
        if track:
            M += 1
        print(track)

    robustness = M / N * 100
    print("Robustness (%) = ", robustness)

    if plot == True:
        # plot results
        tt = dt * np.arange(len(phases))

        plt.figure("State Estimate")
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
        plt.ylim([-10, 10])
        plt.xlabel('time (s)')

        plt.figure("Estimation Errors")
        plt.subplot(411)
        plt.title('Estimation Errors')
        plt.plot(tt, abs(estimate_error[:, :, 0].T))
        plt.ylabel('$\phi$ error')
        plt.ylim([0, 0.5])
        plt.xlim([0, tt[-1]+0.1])
        plt.subplot(412)
        plt.plot(tt, abs(estimate_error[:, :, 1].T))
        plt.ylabel('$\dot{\phi}$ error (1/s)')
        plt.xlim([0, tt[-1]+0.1])
        plt.subplot(413)
        plt.plot(tt, abs(estimate_error[:, :, 2].T))
        plt.ylabel('$l$ error (m)')
        plt.xlim([0, tt[-1]+0.1])
        #plt.ylim([0, 0.5])
        plt.subplot(414)
        plt.plot(tt, abs(estimate_error[:, :, 3].T))
        plt.ylabel('$\\alpha$ error (deg)')
        plt.xlim([0, tt[-1]+0.1])
        #plt.ylim([5, 0.5])
        plt.xlabel('time (s)')
        
        plt.show()
    
    return robustness

def ekf_robustness(kidnap = True):
    total_trials = 0
    RMSerror_phase = []

    robustness = 0

    with open('Continuous_data/GlobalThighAngles_with_Nan.pickle', 'rb') as file:
        nan_dict = pickle.load(file)

    #for subject in Conti_subject_names():
    for subject in ['AB09', 'AB10']: # , 'AB02', 'AB03', 'AB08', 'AB09', 'AB10'
        #print("subject: ", subject)
        for trial in Conti_trial_names(subject):
            if trial == 'subjectdetails':
                continue
            #print("trial: ", trial)
            for side in ['left']:
                if nan_dict[subject][trial][side] == False:
                    print(subject + "/"+ trial + "/"+ side+ ": Trial skipped!")
                    continue
                #print("side: ", side)
                total_trials = total_trials + 1
                
                if kidnap == True:
                    robustness += ekf_bank_test(subject, trial, side, N = 5, plot = False)
                else:
                    RMSE_phase = ekf_test(subject, trial, side, kidnap, plot = False)
                    RMSerror_phase.append([RMSE_phase])

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
    subject = 'AB10'
    trial = 's0x8i0'
    side = 'left'

    #ekf_test(subject, trial, side, kidnap = [0, 1, 2, 3], plot = True)
    ekf_bank_test(subject, trial, side, N = 10, kidnap = [0, 1, 2, 3], plot = True)
    #ekf_robustness(kidnap = True)
    #ekf_robustness(kidnap = False)
