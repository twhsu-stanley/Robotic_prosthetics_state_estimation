import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from EKF import *
from UKF import *
from model_framework import *
from continuous_data_incExp import *
from streaming_data_R01 import *
from basis_model_fitting import measurement_noise_covariance, heteroscedastic_measurement_noise_covariance
import csv

# Skip trials with problematic measurements
with open('Continuous_data/Measurements_with_Nan.pickle', 'rb') as file:
    nan_dict = pickle.load(file)

# Dictionary of all sensors
# All measurements use the basis model, except for directRamp
sensors_dict = {'globalThighAngles':0, 'globalThighVelocities':1, 'atan2':2,
                'globalFootAngles':3, 'ankleMoment':4, 'tibiaForce':5}

# Determine what sensors to be used
# 1) measurements that use the basis model
sensors = ['globalThighAngles', 'globalThighVelocities', 'atan2']

sensor_id = [sensors_dict[key] for key in sensors]
sensor_id_str = ""
for i in range(len(sensor_id)):
    sensor_id_str += str(sensor_id[i])
m_model = model_loader('Measurement_model_' + sensor_id_str +'_NSL.pickle')
    
using_atan2 = np.any(np.array(sensors) == 'atan2')

print("Using sensors:", sensors)

Psi = np.array([load_Psi()[key] for key in sensors], dtype = object)

dt = 1/100
initial_x = np.array([0.3, 0, 0]) # mid-stance
initial_Sigma = np.diag([1e-1, 1e-1, 1e-1])
Q = np.array([[1e-10, 1e-10, 1e-10], [1e-10, 1e-2, 1e-2], [1e-10, 1e-2, 1e-2]]) * dt
U = np.diag([1, 1, 1])
R = U @ measurement_noise_covariance(*sensors) @ U.T
#hetero_cov = heteroscedastic_measurement_noise_covariance(*sensors)
saturation_range = np.array([1.3, 0, 1.9, 0])

# Stride in which kidnapping occurs
kidnap_stride = 4
total_strides = 15

kidnap_percent_gait = np.random.uniform(0, 1)
phase_kidnap =  np.random.uniform(0, 1)
phase_dot_kidnap = np.random.uniform(0, 2)
step_length_kidnap = np.random.uniform(0, 2)
state_kidnap = np.array([phase_kidnap, phase_dot_kidnap, step_length_kidnap])

# Roecover Criteria
phase_recover_thr = 0.05
step_length_recover_thr = 0.1

def kf_test(dataset, subject, trial, side = 'left', kalman_filter = 'ekf', kidnap = False, plot = False):
    print(kalman_filter, "test: ", dataset,"/",subject, "/", trial, '/', side)
    
    # 1) Use the incine experiment dataset
    if dataset == 'inclineExp':
        trial += 'i0'
        # load ground truth
        phases, phase_dots, step_lengths, ramps = Conti_state_vars(subject, trial, side)
        # load measurements
        globalThighAngle, globalThighVelocity, _, _, _, _ = load_Conti_measurement_data(subject, trial, side)
        atan2 = Continuous_atan2_scale_shift(subject, trial, side, plot = False) # use the shifted & scalsed version

        kneeAngle, ankleAngle = load_Conti_joints_angles(subject, trial, side)

        heel_strike_index = Conti_heel_strikes(subject, trial, side) - Conti_heel_strikes(subject, trial, side)[0]
        total_step = int(heel_strike_index[total_strides]) + 1
    
    # 2) Use the R01 dataset
    elif dataset == 'Reznick':
        # here, trial is equivalent to speed: s0x8, s1, s1x2, all
        (phases, phase_dots, step_lengths, ramps, globalThighAngle, globalThighVelocity, _, kneeAngle, ankleAngle) \
        = load_Streaming_data(subject, trial)
        atan2 = Streaming_atan2_scale_shift(subject, trial, plot = False) # use the scaled&shifted version

        LHS = Streaming_data['Streaming'][subject]['Tread']['i0']['events']['LHS'][:][:,0]
        cutPoints = Streaming_data['Streaming'][subject]['Tread']['i0']['events']['cutPoints'][:]
        if trial == 'all':
            start_idx = min(int(cutPoints[0,0]), int(cutPoints[0,1]), int(cutPoints[0,2]))
            end_idx = max(int(cutPoints[1,0]), int(cutPoints[1,1]), int(cutPoints[1,2]))
        elif trial == 's0x8':
            start_idx = int(cutPoints[0,0])
            end_idx = int(cutPoints[1,0])
        elif trial == 's1':
            start_idx = int(cutPoints[0,1])
            end_idx = int(cutPoints[1,1])
        elif trial == 's1x2':
            start_idx = int(cutPoints[0,2])
            end_idx = int(cutPoints[1,2])
        elif trial == 'a0x2':
            start_idx = int(cutPoints[0,3])
            end_idx = int(cutPoints[1,5])
        elif trial == 'a0x5':
            start_idx = int(cutPoints[0,4])
            end_idx = int(cutPoints[1,6])
        
        heel_strike_index = LHS[ np.logical_and((LHS > start_idx), (LHS < end_idx)) ] - start_idx
        if trial == 'all' or trial == 'a0x2' or trial == 'a0x5':
            total_step = len(phases)#int(heel_strike_index[-1]) + 1
        else:
             total_step = int(heel_strike_index[total_strides]) + 1

    else:
        exit("You need to choose a dataset: inclineExp or Reznick")
    
    phases = phases[0 : total_step]
    phase_dots = phase_dots[0 : total_step]
    step_lengths = step_lengths[0 : total_step]
    #ramps = ramps[0 : total_step]
    
    globalThighAngle = globalThighAngle[0 : total_step]
    globalThighVelocity = globalThighVelocity[0 : total_step]
    atan2 = atan2[0 : total_step]
    
    kneeAngle = kneeAngle[0 : total_step]
    ankleAngle = ankleAngle[0 : total_step]

    z_full = np.array([globalThighAngle, globalThighVelocity, atan2])
    z = z_full[sensor_id, :]

    # Create an EKF 
    sys = myStruct()
    sys.f = process_model
    sys.A = A
    sys.h = m_model
    sys.Psi = Psi
    Q = np.array([[1e-10, 1e-10, 1e-10], [1e-10, 1e-3, 1e-4], [1e-10, 1e-4, 1e-3]]) * dt
    sys.Q = Q
    sys.R = R
    init = myStruct()
    init.x = initial_x
    init.Sigma = initial_Sigma
    ekf = extended_kalman_filter(sys, init)
    
    # Create an UKF 
    sys = myStruct()
    sys.f = process_model
    sys.h = m_model
    sys.Psi = Psi
    Q = np.array([[1e-10, 1e-10, 1e-10], [1e-10, 1e-3, 1e-4], [1e-10, 1e-4, 1e-3]]) * dt
    sys.Q = Q
    sys.R = R
    sys.alpha = 1
    sys.beta = 0
    sys.kappa = 0
    init = myStruct()
    init.x = initial_x
    init.Sigma = initial_Sigma
    ukf = unscented_kalman_filter(sys, init)
    
    if kalman_filter == 'ekf':
        filter = ekf
    elif kalman_filter == 'ukf':
        filter = ukf
    else:
        exit("Select a filter to be used: EKF or UKF")
    
    if kidnap != False:
        kidnap_index = int(heel_strike_index[kidnap_stride] +
                           kidnap_percent_gait* (heel_strike_index[kidnap_stride+1] - heel_strike_index[kidnap_stride]))
        print("state_kidnap = [%4.2f, %4.2f, %4.2f]" % (state_kidnap[0], state_kidnap[1], state_kidnap[2]))

    x = np.zeros((total_step, 3))  # state estimate
    std = np.zeros((total_step, 3))  # state estimate
    z_pred = np.zeros((total_step, len(sensors)))
    
    estimate_error = np.zeros((total_step, 3))
    
    kneeAngle_kmd = np.zeros((total_step, 1))
    ankleAngle_kmd = np.zeros((total_step, 1))
    
    for i in range(total_step):
        if kidnap != False:
            if i == kidnap_index:
                filter.x[kidnap] = state_kidnap[kidnap]
        filter.prediction(dt)
        #filter.state_saturation(saturation_range)
        filter.correction(z[:, i], using_atan2)
        filter.state_saturation(saturation_range)

        z_pred[i,:] = filter.z_hat.T
        x[i,:] = filter.x
        std[i, :] = np.sqrt(np.diag(filter.Sigma))
        estimate_error[i, :] = (filter.x - np.array([phases[i], phase_dots[i], step_lengths[i]]))#.reshape(-1)
        if estimate_error[i, 0] > 0.5:
            estimate_error[i, 0] = estimate_error[i, 0] - 1
        elif estimate_error[i, 0] < -0.5:
            estimate_error[i, 0] = 1 + estimate_error[i, 0]

        ## Joints control commands 
        joint_angles = joints_control(x[i,0], x[i,1], x[i,2])
        kneeAngle_kmd[i] = joint_angles[0]
        ankleAngle_kmd[i] = joint_angles[1]

    if kidnap == False:
        start_check_idx = int(3/np.average(phase_dots)/dt)
        SE_phase = np.sum(estimate_error[start_check_idx:, 0] ** 2)
        SE_phase_dot = np.sum(estimate_error[start_check_idx:, 1] ** 2)
        SE_step_length = np.sum(estimate_error[start_check_idx:, 2] ** 2)
        T = len(estimate_error[start_check_idx:, 0])
        RMSE_phase = np.sqrt(SE_phase/T)
        RMSE_phase_dot = np.sqrt(SE_phase_dot/T)
        RMSE_step_length = np.sqrt(SE_step_length/T)
        print("RMSE phase = %5.3f" % RMSE_phase)
        print("RMSE phase_dot = %5.3f" % RMSE_phase_dot)
        print("RMSE step_length = %5.3f" % RMSE_step_length)

        result = (SE_phase, SE_phase_dot, SE_step_length, T)

    if plot == True:
        # plot results
        tt = dt * np.arange(len(phases))
        plt.figure()
        plt.subplot(311)
        plt.title('State Estimation')
        plt.plot(tt, phases, 'k-')
        plt.plot(tt, x[:, 0], 'r--')
        plt.fill_between(tt, x[:, 0]-3*std[:, 0], x[:, 0]+3*std[:, 0], color='red', alpha=0.3)
        plt.legend(('ground truth', 'estimate'))
        plt.ylabel('$\phi$')
        plt.ylim([0, 2])
        plt.xlim([0, tt[-1]+0.1])
        plt.grid()
        plt.subplot(312)
        plt.plot(tt, phase_dots, 'k-')
        plt.plot(tt, x[:, 1], 'r--')
        plt.fill_between(tt, x[:, 1]-3*std[:, 1], x[:, 1]+3*std[:, 1], color='red', alpha=0.3)
        plt.ylabel('$\dot{\phi}~(s^{-1})$')
        plt.xlim([0, tt[-1]+0.1])
        plt.ylim([-0.01, 1.5])
        plt.grid()
        plt.subplot(313)
        plt.plot(tt, step_lengths, 'k-')
        plt.plot(tt, x[:, 2], 'r--')
        plt.fill_between(tt, x[:, 2]-3*std[:, 2], x[:, 2]+3*std[:, 2], color='red', alpha=0.3)
        plt.ylabel('$l~(m)$')
        plt.xlabel('time (s)')
        plt.xlim([0, tt[-1]+0.1])
        plt.ylim([-0.01, 2])
        plt.grid()

        plt.figure()
        plt.subplot(311)
        plt.title('Estimation Errors')
        plt.plot(tt, abs(estimate_error[:, 0].T))
        plt.ylabel('$\phi$ error')
        plt.ylim([0, 0.5])
        plt.xlim([0, tt[-1]+0.1])
        plt.grid()
        plt.subplot(312)
        plt.plot(tt, abs(estimate_error[:, 1].T))
        plt.ylabel('$\dot{\phi}$ error (1/s)')
        plt.xlim([0, tt[-1]+0.1])
        plt.grid()
        plt.subplot(313)
        plt.plot(tt, abs(estimate_error[:, 2].T))
        plt.ylabel('$l$ error (m)')
        plt.xlim([0, tt[-1]+0.1])
        plt.ylim([0, 1])
        plt.grid()
        plt.xlabel('time (s)')
        plt.grid()

        """
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
        
        plt.figure()
        for i in range(len(sensors)):
            plt.subplot(int(str(len(sensors)) + "1" + str(i+1)))
            plt.plot(tt, z[i, :], 'k-')
            plt.plot(tt, z_pred[:, i], 'r--')
            plt.xlim([0, tt[-1]+0.1])
            plt.grid()
            if i == 0:
                plt.title("Measurements")
                plt.legend(('actual', 'predicted'))
            elif i == len(sensors)-1:
                plt.xlabel("time (s)")
        
        plt.figure()
        plt.title("Control Commands: Joint Angles")
        plt.subplot(211)
        plt.plot(tt, kneeAngle, 'k-')
        plt.plot(tt, kneeAngle_kmd, 'm-')
        plt.legend(('actual', 'kinematic model'))
        plt.ylabel('knee angle (deg)')
        plt.subplot(212)
        plt.plot(tt, ankleAngle, 'k-')
        plt.plot(tt, ankleAngle_kmd, 'm-')
        plt.legend(('actual', 'kinematic model'))
        plt.ylabel('ankle angle (deg)')
        plt.xlabel('time (s)')
        
        #plt.show()
    
    if kidnap == False:
        return result
    else:
        print("The test program does not return anything for the kidnapping case")

def kf_bank_test(dataset, subject, trial, side, N = 30, kalman_filter = 'ekf', kidnap = [0, 1, 2, 3], plot = True):
    # N: number of EKFs in the EKF-bank
    print("Monte-Carlo Test: ", dataset, "/", subject, "/", trial, 'i0/', side)

    # 1) Use the incine experiment dataset
    if dataset == 'inclineExp':
        trial += 'i0'
        # load ground truth
        phases, phase_dots, step_lengths, ramps = Conti_state_vars(subject, trial, side)
        # load measurements
        globalThighAngle, globalThighVelocity, atan2, globalFootAngle, ankleMoment, tibiaForce = load_Conti_measurement_data(subject, trial, side)

        heel_strike_index = Conti_heel_strikes(subject, trial, side) - Conti_heel_strikes(subject, trial, side)[0]
    
    # 2) Use the R01 dataset
    elif dataset == 'Reznick':
        # here, trial is equivalent to speed: s0x8, s1, s1x2, all
        (phases, phase_dots, step_lengths, ramps, globalThighAngle, globalThighVelocity, atan2) = load_Streaming_data(subject, trial)
        LHS = Streaming_data['Streaming'][subject]['Tread']['i0']['events']['LHS'][:][:,0]
        
        cutPoints = Streaming_data['Streaming'][subject]['Tread']['i0']['events']['cutPoints'][:]
        if trial == 'all':
            start_idx = min(int(cutPoints[0,0]), int(cutPoints[0,1]), int(cutPoints[0,2]))
            end_idx = max(int(cutPoints[1,0]), int(cutPoints[1,1]), int(cutPoints[1,2]))
        elif trial == 's0x8':
            start_idx = int(cutPoints[0,0])
            end_idx = int(cutPoints[1,0])
        elif trial == 's1':
            start_idx = int(cutPoints[0,1])
            end_idx = int(cutPoints[1,1])
        elif trial == 's1x2':
            start_idx = int(cutPoints[0,2])
            end_idx = int(cutPoints[1,2])
    
        heel_strike_index = LHS[ np.logical_and((LHS > start_idx), (LHS < end_idx)) ] - start_idx

    else:
        exit("You need to choose a dataset: inclineExp or Reznick")
    
    total_step = int(heel_strike_index[total_strides]) + 1
    phases = phases[0 : total_step]
    phase_dots = phase_dots[0 : total_step]
    step_lengths = step_lengths[0 : total_step]
    ramps = ramps[0 : total_step]

    z_full = np.array([[globalThighAngle], [globalThighVelocity], [atan2]])#, [globalFootAngle], [ankleMoment], [tibiaForce]
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
    
    kidnap_index = np.random.randint(heel_strike_index[kidnap_stride], heel_strike_index[kidnap_stride+1]) # step at which kidnapping occurs
    x = np.zeros((N+1, total_step, 4))  # state estimate
    estimate_error = np.zeros((N+1, total_step, 4))
    phase_converge_dist = np.zeros((N+1, total_step))
    r11 = 0
    r13 = 0
    r33 = 0
    r15 = 0
    r55 = 0

    for n in range(N+1):
        # n = 0: normal test w/0 kidnapping
        # initialize the state
        init.x = np.array([[phases[0]], [phase_dots[0]], [step_lengths[0]], [ramps[0]]])
        #init.x = np.array([[phases[0]], [phase_dots[0]], [step_lengths[0]], [0]])
        init.Sigma = inital_Sigma
        # build EKF
        filter = extended_kalman_filter(sys, init)
        
        phase_kidnap = np.random.uniform(0, 1)
        phase_dot_kidnap = np.random.uniform(0, 5)
        step_length_kidnap = np.random.uniform(0, 2)
        ramp_kidnap = np.random.uniform(-45, 45)
        state_kidnap = np.array([[phase_kidnap], [phase_dot_kidnap], [step_length_kidnap], [ramp_kidnap]])

        directRamp = np.zeros((total_step, 1))
        L_cop = np.zeros((total_step, 1))
        stance = False
        stance_prev = False
        stance_idxs = 0
        stance_idx1 = 0
        stance_idx2 = 0
        for i in range(total_step):
            # kidnap
            if i == kidnap_index and n > 0:
                filter.x[kidnap] = state_kidnap[kidnap]
            
            filter.prediction(dt)
            filter.state_saturation(saturation_range)
            """
            if using_ankleMoment or using_tibiaForce or using_footAngles or using_directRamp:
                if tibiaForce[i] <= tibiaForce_threshold:
                    # Location of the centoer of pressure
                    L_cop[i] = -ankleMoment[i] / tibiaForce[i]
                    if stance_prev == False and L_cop[i] > L_cop_lower and L_cop[i] <= L_cop_upper and (i - stance_idxs) > 0.5/dt:
                        stance_idxs = i
                        stance = True
                        stance_prev = stance
                        
                    elif stance_prev == True and L_cop[i] > L_cop_upper:
                        stance_idx2 = i
                        stance_idx1 = stance_idxs
                        stance = False
                        stance_prev = stance
                else:
                    L_cop[i] = -1 # arbitraty number
                    if stance_prev == True:
                        stance_idx2 = i
                        stance_idx1 = stance_idxs
                    stance = False
                    stance_prev = stance
                
                if stance_idx1 < stance_idx2:
                    directRamp[i] = np.mean(globalFootAngle[stance_idx1:stance_idx2])
                else:
                    directRamp[i] = 1e-4
                
                #if stance == True:
                if tibiaForce[i] <= tibiaForce_threshold:
                    filter.h = m_model
                    if heteroscedastic == True:
                        filter.R = np.diag(hetero_cov[:, int(filter.x[0, 0]*150)])
                        if using_directRamp:
                            filter.R = np.diag(np.append(np.diag(filter.R), R_directRamp))               
                    else:
                        filter.R = R

                    if using_directRamp:
                        filter.correction(z[:, i], Psi, using_atan2, direct_ramp = directRamp[i])
                    else:
                        filter.correction(z[:, i], Psi, using_atan2, direct_ramp = False)
                else: # swing
                    filter.h = m_model_swing
                    if heteroscedastic == True:
                        filter.R = np.diag(hetero_cov[sensor_swing_id, int(filter.x[0, 0]*150)])
                        if using_directRamp:
                            filter.R = np.diag(np.append(np.diag(filter.R), R_directRamp))
                    else:
                        filter.R = R_swing
                    
                    if using_directRamp:
                        filter.correction(z_full[sensor_swing_id, i], Psi_swing, using_atan2, direct_ramp = directRamp[i])
                    else:
                        filter.correction(z_full[sensor_swing_id, i], Psi_swing, using_atan2, direct_ramp = False)       

            else:
                if heteroscedastic == True:
                    filter.R = np.diag(hetero_cov[:, int(filter.x[0, 0]*150)])
                else:
                    filter.R = R
            """
            filter.correction(z[:, i], Psi, using_atan2)
            filter.state_saturation(saturation_range)

            x[n, i,:] = filter.x.T

            estimate_error[n, i, :] = (filter.x - np.array([[phases[i]], [phase_dots[i]], [step_lengths[i]], [ramps[i]]])).reshape(-1)
            if estimate_error[n, i, 0] > 0.5:
                estimate_error[n, i, 0] = estimate_error[n, i, 0] - 1
            elif estimate_error[n, i, 0] < -0.5:
                estimate_error[n, i, 0] = 1 + estimate_error[n, i, 0]
        if n > 0:
            #idx_1 = int(kidnap_index + 1/np.average(phase_dots)/dt)
            idx_1 = int(heel_strike_index[kidnap_stride + 2])
            #idx_3 = int(kidnap_index + 3/np.average(phase_dots)/dt)
            idx_3 = int(heel_strike_index[kidnap_stride + 4])
            #idx_5 = int(kidnap_index + 5/np.average(phase_dots)/dt)
            idx_5 = int(heel_strike_index[kidnap_stride + 6])
            
            # 1) Absolute estimate error approach
            """
            #phase_recover_1 = np.all(abs(estimate_error[n, int(kidnap_index + 1/np.average(phase_dots)/dt):, 0]) < phase_recover_thr)
            #phase_recover_3 = np.all(abs(estimate_error[n, int(kidnap_index + 3/np.average(phase_dots)/dt):, 0]) < phase_recover_thr)
            #step_length_recover_1 = np.all(abs(estimate_error[n, int(kidnap_index + 1/np.average(phase_dots)/dt):, 2]) < step_length_recover_thr)
            #step_length_recover_3 = np.all(abs(estimate_error[n, int(kidnap_index + 3/np.average(phase_dots)/dt):, 2]) < step_length_recover_thr)
            #ramp_recover_1 = np.all(abs(estimate_error[n, int(kidnap_index + 1/np.average(phase_dots)/dt):, 3]) < ramp_recover_thr)
            #ramp_recover_3 = np.all(abs(estimate_error[n, int(kidnap_index + 3/np.average(phase_dots)/dt):, 3]) < ramp_recover_thr)
            """

            # 2) Convergence approach
            phase_converge_dist[n, :] = phase_error(x[n,:, 0], x[0,:, 0])
            phase_recover_1 = np.all(phase_converge_dist[n,idx_1:] < phase_recover_thr)
            step_length_recover_1 = np.all(abs(x[n, idx_1:, 2] - x[0, idx_1:, 2]) < step_length_recover_thr)
            ramp_recover_1 = np.all(abs(x[n, idx_1:, 3] - x[0, idx_1:, 3]) < ramp_recover_thr)
            phase_recover_3 = np.all(phase_converge_dist[n,idx_3:] < phase_recover_thr)
            step_length_recover_3 = np.all(abs(x[n, idx_3:, 2] - x[0, idx_3:, 2]) < step_length_recover_thr)
            ramp_recover_3 = np.all(abs(x[n, idx_3:, 3] - x[0, idx_3:, 3]) < ramp_recover_thr)
            phase_recover_5 = np.all(phase_converge_dist[n,idx_5:] < phase_recover_thr)
            step_length_recover_5 = np.all(abs(x[n, idx_5:, 2] - x[0, idx_5:, 2]) < step_length_recover_thr)
            ramp_recover_5 = np.all(abs(x[n, idx_5:, 3] - x[0, idx_5:, 3]) < ramp_recover_thr)

            track_11 = (phase_recover_1 and step_length_recover_1 and ramp_recover_1)
            track_13 = (phase_recover_1 and step_length_recover_3 and ramp_recover_3)
            track_33 = (phase_recover_3 and step_length_recover_3 and ramp_recover_3)
            track_15 = (phase_recover_1 and step_length_recover_5 and ramp_recover_5)
            track_55 = (phase_recover_5 and step_length_recover_5 and ramp_recover_5)
            print("n:", n)
            print("i, j | track_ij = phase_i & step_len_j & ramp_j")
            print("1, 1 |  ", track_11, "  =  ", phase_recover_1, " &    ", step_length_recover_1, "  & ", ramp_recover_1)
            print("1, 3 |  ", track_13, "  =  ", phase_recover_1, " &    ", step_length_recover_3, "  & ", ramp_recover_3)
            print("3, 3 |  ", track_33, "  =  ", phase_recover_3, " &    ", step_length_recover_3, "  & ", ramp_recover_3)
            print("1, 5 |  ", track_15, "  =  ", phase_recover_1, " &    ", step_length_recover_5, "  & ", ramp_recover_5)
            print("5, 5 |  ", track_55, "  =  ", phase_recover_5, " &    ", step_length_recover_5, "  & ", ramp_recover_5)
            
            if track_11:
                r11 += 1
            if track_13:
                r13 += 1
            if track_33:
                r33 += 1
            if track_15:
                r15 += 1
            if track_55:
                r55 += 1
        
    robustness_11 = r11 / N * 100
    robustness_13 = r13 / N * 100
    robustness_33 = r33 / N * 100
    robustness_15 = r15 / N * 100
    robustness_55 = r55 / N * 100

    print("R_11(%) = ", robustness_11, "|| R_13(%) = ", robustness_13, "|| R_15(%) = ",
          robustness_15, "|| R_33(%) = ", robustness_33, "|| R_55(%) = ", robustness_55)
    print("---------------------------------------------------------------------------")

    if plot == True:
        # plot results
        tt = dt * np.arange(len(phases))

        plt.figure("State Estimate")
        plt.subplot(411)
        plt.title('EKFs-Bank Test')
        plt.plot(tt, phases, 'k-', linewidth=2)
        plt.plot(tt,  x[1:, :, 0].T, '--', alpha = 0.5)#
        plt.plot(tt,  x[0, :, 0].T, 'r--', linewidth=2)
        plt.ylabel('$\phi$')
        plt.legend(('ground truth', 'estimate'))
        #plt.legend(('ground truth', 'estimate'), bbox_to_anchor=(1, 1.05))
        plt.ylim([0, 2])
        plt.xlim([0, tt[-1]+0.1])
        plt.grid()
        plt.subplot(412)
        plt.plot(tt, phase_dots, 'k-', linewidth=2)
        plt.plot(tt, x[1:, :, 1].T, '--', alpha = 0.5)
        plt.plot(tt, x[0, :, 1].T, 'r--', linewidth=2)
        plt.ylabel('$\dot{\phi}~(s^{-1})$')
        plt.xlim([0, tt[-1]+0.1])
        plt.ylim([0.2, 1.5])
        plt.grid()
        plt.subplot(413)
        plt.plot(tt, step_lengths, 'k-', linewidth=2)
        plt.plot(tt, x[1:, :, 2].T, '--', alpha = 0.5)
        plt.plot(tt, x[0, :, 2].T, 'r--', linewidth=2)
        #plt.plot(tt[idx_1:], step_lengths[idx_1:] + step_length_est_error_mean + step_length_recover_thr, 'k--')
        #plt.plot(tt[idx_1:], step_lengths[idx_1:] + step_length_est_error_mean - step_length_recover_thr, 'k--')
        plt.ylabel('$l~(m)$')
        plt.xlim([0, tt[-1]+0.1])
        plt.ylim([0, 2])
        plt.grid()
        plt.subplot(414)
        plt.plot(tt, ramps, 'k-', linewidth=2)
        plt.plot(tt, x[1:, :, 3].T, '--', alpha = 0.5)
        plt.plot(tt, x[0, :, 3].T, 'r--', linewidth=2)
        #plt.plot(tt[idx_1:], ramps[idx_1:] + ramp_est_error_mean + ramp_recover_thr, 'k--')
        #plt.plot(tt[idx_1:], ramps[idx_1:] + ramp_est_error_mean - ramp_recover_thr, 'k--')
        plt.ylabel('$\\alpha~(deg)$')
        plt.xlim([0, tt[-1]+0.1])
        plt.ylim([-10, 10])
        plt.grid()
        plt.xlabel('time (s)')

        plt.figure("Estimation Errors")
        plt.subplot(411)
        plt.title('Estimation Errors')
        plt.plot(tt, abs(estimate_error[:, :, 0].T))
        plt.ylabel('$\phi$ error')
        plt.ylim([0, 0.5])
        plt.xlim([0, tt[-1]+0.1])
        plt.grid()
        plt.subplot(412)
        plt.plot(tt, abs(estimate_error[:, :, 1].T))
        plt.ylabel('$\dot{\phi}$ error (1/s)')
        plt.xlim([0, tt[-1]+0.1])
        plt.grid()
        plt.subplot(413)
        plt.plot(tt, abs(estimate_error[:, :, 2].T))
        plt.ylabel('$l$ error (m)')
        plt.xlim([0, tt[-1]+0.1])
        plt.grid()
        plt.subplot(414)
        plt.plot(tt, abs(estimate_error[:, :, 3].T))
        plt.ylabel('$\\alpha$ error (deg)')
        plt.xlim([0, tt[-1]+0.1])
        plt.grid()
        plt.xlabel('time (s)')
        
        plt.figure("Covergence Distance")
        plt.subplot(411)
        plt.title('Covergence Distance')
        plt.plot(tt, phase_converge_dist[1:,:].T, '--')
        plt.plot(tt[idx_1:], phase_recover_thr*np.ones(len(tt[idx_1:])), 'k-', alpha = 0.4, linewidth=2)
        plt.plot(idx_1*dt*np.ones(5), np.linspace(0,1,5), 'k-', alpha = 0.4, linewidth=2)
        plt.plot(idx_3*dt*np.ones(5), np.linspace(0,1,5), 'b-', alpha = 0.4, linewidth=2)
        plt.plot(idx_5*dt*np.ones(5), np.linspace(0,1,5), 'r-', alpha = 0.4, linewidth=2)
        plt.ylabel('$\Delta \phi$')
        plt.ylim([0, 0.5])
        plt.xlim([0, tt[-1]+0.1])
        plt.grid()
        plt.subplot(412)
        plt.plot(tt, abs(x[1:,:,1] - x[0,:,1]).T, '--')
        plt.ylabel('$\Delta \dot{\phi}$ (1/s)')
        plt.xlim([0, tt[-1]+0.1])
        plt.grid()
        plt.subplot(413)
        plt.plot(tt, abs(x[1:,:,2] - x[0,:,2]).T, '--')
        plt.plot(tt[idx_1:], step_length_recover_thr*np.ones(len(tt[idx_1:])), 'k-', alpha = 0.4, linewidth=2)
        plt.plot(idx_1*dt*np.ones(5), np.linspace(0,2,5), 'k-', alpha = 0.4, linewidth=2)
        plt.plot(idx_3*dt*np.ones(5), np.linspace(0,2,5), 'b-', alpha = 0.4, linewidth=2)
        plt.plot(idx_5*dt*np.ones(5), np.linspace(0,2,5), 'r-', alpha = 0.4, linewidth=2)
        plt.ylabel('$\Delta l$ (m)')
        plt.xlim([0, tt[-1]+0.1])
        plt.grid()
        plt.subplot(414)
        plt.plot(tt, abs(x[1:,:,3] - x[0,:,3]).T, '--')
        plt.plot(tt[idx_1:], ramp_recover_thr*np.ones(len(tt[idx_1:])), 'k-', alpha = 0.4, linewidth=2)
        plt.plot(idx_1*dt*np.ones(5), np.linspace(0,20,5), 'k-', alpha = 0.4, linewidth=2)
        plt.plot(idx_3*dt*np.ones(5), np.linspace(0,20,5), 'b-', alpha = 0.4, linewidth=2)
        plt.plot(idx_5*dt*np.ones(5), np.linspace(0,20,5), 'r-', alpha = 0.4, linewidth=2)
        plt.ylabel('$\Delta \\alpha$ (deg)')
        plt.xlim([0, tt[-1]+0.1])
        plt.grid()
        plt.xlabel('time (s)')
        plt.show()
    
    return (robustness_11, robustness_13, robustness_33, robustness_15, robustness_55)

def kf_robustness(kidnap = True, kalman_filter = 'ekf',):
    total_trials = 0
    robustness_11 = 0
    robustness_13 = 0 
    robustness_33 = 0
    robustness_15 = 0
    robustness_55 = 0

    SE_phase_total = 0
    SE_step_length_total = 0
    SE_ramp_total = 0
    SE_directRamp_total = 0
    T_total = 0

    poor_step_length_est = 0
    poor_ramp_est = 0
    poor_task_est = 0

    for dataset in ['Reznick', 'inclineExp']:
        for subject in Conti_subject_names(): 
            #for trial in Conti_trial_names(subject):
            for trial in ['s0x8', 's1', 's1x2']:
                if trial == 'subjectdetails':
                    continue
                for side in ['left']:
                    if dataset == 'inclineExp' and nan_dict[subject][trial+'i0'][side] == False:
                        print('inclineExp/' + subject + "/"+ trial + "/"+ side + ": Trial skipped!")
                        continue
                    if dataset == 'Reznick' and ((subject == 'AB08' and trial == 's0x8') or subject == 'AB10'):
                        continue
                    total_trials = total_trials + 1
                    
                    if kidnap != False:
                        (R_11, R_13, R_33, R_15, R_55) = ekf_bank_test(dataset, subject, trial, side, 5, heteroscedastic, kidnap, plot = False)
                        robustness_11 += R_11
                        robustness_13 += R_13
                        robustness_33 += R_33
                        robustness_15 += R_15
                        robustness_55 += R_55
                        print("**Current Average R_11 = %4.1f %%" % (robustness_11 / total_trials), 
                            "|| R_13 = %4.1f %%" % (robustness_13 / total_trials),
                            "|| R_15 = %4.1f %%" % (robustness_15 / total_trials),
                            "|| R_33 = %4.1f %%" % (robustness_33 / total_trials),
                            "|| R_55 = %4.1f %%" % (robustness_55 / total_trials))
                    else:
                        SE_phase, _, SE_step_length, SE_ramp, SE_directRamp, T = ekf_test(dataset, subject, trial, side, heteroscedastic, kidnap, plot = False)
                        SE_phase_total += SE_phase
                        SE_step_length_total += SE_step_length
                        SE_ramp_total += SE_ramp
                        SE_directRamp_total += SE_directRamp
                        T_total += T

                        print("Current RMSE phase = %5.3f" % np.sqrt(SE_phase_total/T_total))
                        print("Current RMSE step_length = %5.3f" % np.sqrt(SE_step_length_total/T_total))
                        print("----------------------------------------------------------------")
                        #print("Current RMSE ramp = %5.3f" % np.sqrt(SE_ramp_total/T_total))
                        #print("Current RMSE directRamp = %5.3f" % np.sqrt(SE_directRamp_total/T_total))

                        """
                        if RMSE_phase > 0.05 or RMSE_step_length > 0.1 or RMSE_ramp > 2:
                            print(subject, "/", trial, '/', side, ": RMSE exceeds the threshold!")
                            print("RMSE phase = %5.3f" % RMSE_phase)
                            print("RMSE step_length = %5.3f" % RMSE_step_length)
                            print("RMSE ramp = %5.3f" % RMSE_ramp)
                            print("==================================================")
                        
                        # Number of trials whose task esimates are poor
                        if RMSE_step_length > 0.1:
                            poor_step_length_est += 1
                        if RMSE_ramp > 2:
                            poor_ramp_est += 1
                        if RMSE_step_length > 0.1 and RMSE_ramp > 2:
                            poor_task_est += 1
                        """

    if kidnap != False:
        robustness_11 = robustness_11 / total_trials
        robustness_13 = robustness_13 / total_trials
        robustness_33 = robustness_33 / total_trials
        robustness_15 = robustness_15 / total_trials
        robustness_55 = robustness_55 / total_trials

        print("==========================================")
        print("Overall Robustness_11 = %4.1f %%" % robustness_11)
        print("Overall Robustness_13 = %4.1f %%" % robustness_13)
        print("Overall Robustness_33 = %4.1f %%" % robustness_33)
        print("Overall Robustness_15 = %4.1f %%" % robustness_15)
        print("Overall Robustness_55 = %4.1f %%" % robustness_55)
    else:
        print("Total RMSE phase = %5.3f" % np.sqrt(SE_phase_total/T_total))
        print("Total RMSE step_length = %5.3f" % np.sqrt(SE_step_length_total/T_total))
        print("Total RMSE ramp = %5.3f" % np.sqrt(SE_ramp_total/T_total))
        print("Total RMSE directRamp = %5.3f" % np.sqrt(SE_directRamp_total/T_total))

        """
        print("Max RMSE phase = %5.3f" % np.max(RMSE_phase_list))
        print("Max RMSE step_length = %5.3f" % np.max(RMSE_step_length_list))
        print("Max RMSE ramp = %5.3f" % np.max(RMSE_ramp_list))

        print("Percentage of trials with poorly estimated step lengths: %4.2f %%" % (poor_step_length_est / total_trials * 100))
        print("Percentage of trials with poorly estimated ramps: %4.2f %%" % (poor_ramp_est / total_trials * 100))
        print("Percentage of trials with poorly estimated step lengths and ramps %4.2f %%" % (poor_task_est / total_trials * 100))

        plt.figure()
        plt.plot(RMSE_step_length_list, RMSE_ramp_list, 'r.')
        plt.plot(0.1 * np.ones(50), np.linspace(0,5,50), 'k--')
        plt.plot(np.linspace(0,0.5,50), 2 * np.ones(50), 'k--')
        plt.xlim([0, max(0.2, np.max(RMSE_step_length_list)+0.05)])
        plt.ylim([0, max(3, np.max(RMSE_ramp_list)+0.2)])
        plt.xlabel("RMSE step_length")
        plt.ylabel("RMSE ramp")
        plt.grid()
        plt.show()
        """

if __name__ == '__main__':
    dataset = 'inclineExp'
    subject = 'AB05'
    trial = 's1'
    side = 'left'

    if nan_dict[subject][trial+'i0'][side] == False:
        print(subject + "/"+ trial + "/"+ side+ ": This trial should be skipped!")
    
    #dataset = 'Reznick'
    #subject = 'AB07'
    #trial = 's1'

    kf_test(dataset, subject, trial, side, kalman_filter = 'ekf', kidnap = [0, 1, 2], plot = True)
    kf_test(dataset, subject, trial, side, kalman_filter = 'ukf', kidnap = [0, 1, 2], plot = True)
    plt.show()
