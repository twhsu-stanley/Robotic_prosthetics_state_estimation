from pickle import FALSE
import numpy as np
import seaborn as sns
import time
import matplotlib.pyplot as plt
from Filters.EKF import *
from Filters.UKF import *
from wrapping import *
from load_Psi import *
from model_framework import *
from continuous_data_incExp import *
from continuous_data_R01 import *
from basis_model_fitting import measurement_noise_covariance
import csv

# Robustification mechanisms
state_saturation = True
adaptive_cov = False
reset = True

# Dictionary of all sensors
sensors_dict = {'globalThighAngles':0, 'globalThighVelocities':1, 'atan2':2, 'globalFootAngles':3}

# Determine what sensors to be used
sensors = ['globalThighAngles', 'globalThighVelocities','atan2', 'globalFootAngles'] #

sensor_id = [sensors_dict[key] for key in sensors]

model_name = "Measurement_model"
for s in sensors:
    model_name += ('_' + s)
model_name += ".pickle"
m_model = model_loader(model_name)
#m_model = model_loader('Measurement_model_globalThighAngles_globalThighVelocities_atan2_globalFootAngles.pickle')
    
using_atan2 = np.any(np.array(sensors) == 'atan2')

print("Using sensors:", sensors)

Psi = np.array([load_Psi()[key] for key in sensors], dtype = object)

dt = 1/100

# Filter settings
initial_Sigma = np.diag([0.5, 2, 1, 5]) * 0.1 #np.diag([1e-2, 1e-1, 1e-1, 1e-1]) #
Q_ekf = np.diag([0, 1e-3, 5e-2, 5]) * dt # np.diag[0, 1e-3, 5e-2, 5]
Q_ukf = np.diag([1e-20, 1e-1, 5e-1, 5]) * dt  #np.diag([0, 1e-1, 5e-1, 5e-1]) * dt #

if using_atan2:
    U_ekf = np.diag([2, 1.5, 1, 1])
else:
    U_ekf = np.diag([2, 1.5, 1])
R_ekf = U_ekf @ measurement_noise_covariance(*sensors) @ U_ekf.T

if using_atan2:
    U_ukf = np.diag([2, 1.5, 1, 1])
else:
    U_ukf = np.diag([2, 1.5, 1])
R_ukf = U_ukf @ measurement_noise_covariance(*sensors) @ U_ukf.T

# saturation_range = [phase_dots_max, phase_dots_min, step_lengths_max, step_lengths_min, ramp_max, ramp_min]
saturation_range = np.array([1.17, 0.59, 1.81, 0.82, 11, -11])

# Stride in which kidnapping occurs
kidnap_stride = 2 # 5
total_strides = 5 # 11

# Roecover Criteria
phase_recover_thr = 0.08 #0.05 # 
step_length_recover_thr = 0.15 #0.2 #
ramp_recover_thr = 2.2 #2.5 #

# for multi-filter plotting
num_plots = 0
fig_est, axs_est = plt.subplots(4, 1)
#fig_zpred, axs_zpred = plt.subplots(3, 1)

def kf_test(dataset, subject, trial, side, kalman_filter = 'ekf', kidnap = False, plot = False):
    print(kalman_filter, "test: ", dataset,"/",subject, "/", trial, '/', side)

    # 1) Use the incine experiment dataset
    if dataset == 'inclineExp' or dataset[0] == 'inclineExp':
        # load ground truth
        phases, phase_dots, step_lengths, ramps = get_Continuous_state_vars(subject, trial, side)
        # load measurements
        globalThighAngle, globalThighVelocity, atan2, globalFootAngle, ankleMoment, tibiaForce = get_Continuous_measurement_data(subject, trial, side)
        atan2 = get_Continuous_atan2_scale_shift(subject, trial, side, plot = False) # use the shifted & scaled version
        kneeAngle, ankleAngle = get_Continuous_joints_angles(subject, trial, side)

        heel_strike_index = get_Continuous_heel_strikes(subject, trial, side) - get_Continuous_heel_strikes(subject, trial, side)[0]
        total_step = int(heel_strike_index[total_strides]) + 1
    
    # 2) Use the R01 dataset
    elif dataset == 'Reznick' or dataset[0] == 'Reznick':
        # here, trial is equivalent to speed: s0x8, s1, s1x2, all
        if 'i' in trial:
            speed = trial.split('i')[0]
            incline = 'i' + trial.split('i')[1]
        elif 'd' in trial:
            trial = trial.split('d')[0]
            incline = 'd' + trial.split('kd')[1]

        (phases, phase_dots, step_lengths, ramps, globalThighAngle, globalThighVelocity, atan2, kneeAngle, ankleAngle) = load_Streaming_data(subject, speed)
        #atan2 = get_Streaming_atan2_scale_shift(subject, speed, plot = False) # use the scaled & shifted version

        LHS = Streaming_data['Streaming'][subject]['Tread'][incline]['events']['LHS'][:][:,0]
        cutPoints = Streaming_data['Streaming'][subject]['Tread'][incline]['events']['cutPoints'][:]
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
    ramps = ramps[0 : total_step]
    # initialize the prior with ground truth
    initial_x = np.array([phases[0], phase_dots[0], step_lengths[0], ramps[0]])
    
    globalThighAngle = globalThighAngle[0 : total_step]
    globalThighVelocity = globalThighVelocity[0 : total_step]
    atan2 = atan2[0 : total_step]
    globalFootAngle = globalFootAngle[0 : total_step]
    
    kneeAngle = kneeAngle[0 : total_step]
    ankleAngle = ankleAngle[0 : total_step]
    
    z_full = np.array([globalThighAngle, globalThighVelocity, atan2, globalFootAngle])
    z = z_full[sensor_id, :]
    
    if kalman_filter == 'ekf':
        # Create an EKF 
        sys = myStruct()
        sys.f = process_model
        sys.A = A
        sys.h = m_model
        sys.Psi = Psi
        sys.Q = Q_ekf
        sys.R = R_ekf
        sys.saturation = state_saturation
        sys.saturation_range = saturation_range
        sys.reset = reset
        sys.adapt = adaptive_cov
        init = myStruct()
        init.x = initial_x
        init.Sigma = initial_Sigma
        ekf = extended_kalman_filter(sys, init)
        filter = ekf
        clr = 'r' # color

    elif kalman_filter == 'ukf':
        # Create an UKF 
        sys = myStruct()
        sys.f = process_model
        sys.h = m_model
        sys.Psi = Psi
        sys.Q = Q_ukf
        sys.R = R_ukf
        sys.alpha = 1e-3
        sys.beta = 2
        sys.kappa = 0
        sys.saturation = state_saturation
        sys.saturation_range = saturation_range
        sys.reset = reset
        init = myStruct()
        init.x = initial_x
        init.Sigma = initial_Sigma
        ukf = unscented_kalman_filter(sys, init)
        filter = ukf
        clr = 'b'
    else:
        exit("Select a filter to be used: EKF or UKF")
    
    if kidnap != False:
        kidnap_percent_gait = np.random.uniform(0.05, 0.95)
        phase_kidnap =  np.random.uniform(0, 1)
        phase_dot_kidnap = np.random.uniform(0, 5)
        step_length_kidnap = np.random.uniform(0, 2)
        ramp_kidnap = np.random.uniform(-10, 10)
        state_kidnap = np.array([phase_kidnap, phase_dot_kidnap, step_length_kidnap, ramp_kidnap])

        kidnap_index = int(heel_strike_index[kidnap_stride] +
                           kidnap_percent_gait* (heel_strike_index[kidnap_stride+1] - heel_strike_index[kidnap_stride]))
        print("state_kidnap = [%4.2f, %4.2f, %4.2f, %4.2f]" % (state_kidnap[0], state_kidnap[1], state_kidnap[2], state_kidnap[3]))

    x = np.zeros((total_step, 4))  # state estimate
    std = np.zeros((total_step, 4))  # state estimate
    z_pred = np.zeros((total_step, len(sensors)))
    
    md = np.zeros(total_step)
    estimate_error = np.zeros((total_step, 4))
    
    #kneeAngle_kmd = np.zeros(total_step)
    #ankleAngle_kmd = np.zeros(total_step)
    
    time_0 = time.time()
    for i in range(total_step):
        if kidnap != False:
            if i == kidnap_index:
                filter.x[kidnap] = state_kidnap[kidnap]
        filter.prediction(dt)
        filter.correction(z[:, i], using_atan2)

        z_pred[i,:] = filter.z_hat.T
        x[i,:] = filter.x
        std[i, :] = np.sqrt(np.diag(filter.Sigma))
        md[i] = np.sqrt(filter.MD_square)
        estimate_error[i, :] = (filter.x - np.array([phases[i], phase_dots[i], step_lengths[i], ramps[i]]))
        if estimate_error[i, 0] > 0.5:
            estimate_error[i, 0] = estimate_error[i, 0] - 1
        elif estimate_error[i, 0] < -0.5:
            estimate_error[i, 0] = 1 + estimate_error[i, 0]

        ## Joints control commands 
        #joint_angles = joints_control(x[i,0], x[i,1], x[i,2], x[i,3])
        #kneeAngle_kmd[i] = joint_angles[0]
        #ankleAngle_kmd[i] = joint_angles[1]
    
    time_per = (time.time() - time_0)/total_step * 1000
    print("Time per step (ms) =", time_per)

    if kidnap == False:
        start_check_idx = int(5/np.average(phase_dots)/dt)
        SE_phase = np.sum(estimate_error[start_check_idx:, 0] ** 2)
        SE_phase_dot = np.sum(estimate_error[start_check_idx:, 1] ** 2)
        SE_step_length = np.sum(estimate_error[start_check_idx:, 2] ** 2)
        SE_ramp = np.sum(estimate_error[start_check_idx:, 3] ** 2)
        T = len(estimate_error[start_check_idx:, 0])
        RMSE_phase = np.sqrt(SE_phase/T)
        RMSE_phase_dot = np.sqrt(SE_phase_dot/T)
        RMSE_step_length = np.sqrt(SE_step_length/T)
        RMSE_ramp = np.sqrt(SE_ramp/T)
        print("RMSE phase = %5.3f" % RMSE_phase)
        print("RMSE phase_dot = %5.3f" % RMSE_phase_dot)
        print("RMSE step_length = %5.3f" % RMSE_step_length)
        print("RMSE ramp = %5.3f" % RMSE_ramp)
        result = (SE_phase, SE_phase_dot, SE_step_length, SE_ramp, T)
    
    if plot == True:
        global num_plots
        num_plots += 1
        tt = dt * np.arange(len(phases))
        # plot results
        #plt.figure()
        #plt.subplot(311)
        axs_est[0].set_title('State Estimation', fontsize = 16)
        if num_plots == 1:
            axs_est[0].plot(tt, phases, 'k-')
        axs_est[0].plot(tt, x[:, 0], color = clr, linestyle = 'dashed')
        axs_est[0].fill_between(tt, x[:, 0]-3*std[:, 0], x[:, 0]+3*std[:, 0], color = clr, alpha=0.2)
        #axs_est[0].legend(('ground truth', 'estimates'), fontsize = 14)
        axs_est[0].set_ylabel('$\phi$', fontsize = 14)
        #axs_est[0].set_ylim([0, 1.2])
        axs_est[0].set_xlim([0, tt[-1]+0.1])
        axs_est[0].tick_params(axis='both', bottom=False, top=False,labelbottom=False)
        axs_est[0].grid(True)
        #plt.subplot(312)
        axs_est[1].plot(tt, phase_dots, 'k-')
        axs_est[1].plot(tt, x[:, 1], color = clr, linestyle = 'dashed')
        axs_est[1].fill_between(tt, x[:, 1]-3*std[:, 1], x[:, 1]+3*std[:, 1], color = clr, alpha=0.2)
        axs_est[1].set_ylabel('$\dot{\phi}~(1/s)$', fontsize = 14)
        axs_est[1].set_xlim([0, tt[-1]+0.1])
        #axs_est[1].set_ylim([-0.01, 1.5])
        axs_est[1].tick_params(axis='both', bottom=False, top=False,labelbottom=False)
        axs_est[1].grid(True)
        #plt.subplot(313)
        axs_est[2].plot(tt, step_lengths, 'k-')
        axs_est[2].plot(tt, x[:, 2], color = clr, linestyle = 'dashed')
        axs_est[2].fill_between(tt, x[:, 2]-3*std[:, 2], x[:, 2]+3*std[:, 2], color = clr, alpha=0.2)
        axs_est[2].set_ylabel('$l$', fontsize = 14)
        axs_est[2].set_xlim([0, tt[-1]+0.1])
        #axs_est[2].set_ylim([-0.01, 2])
        axs_est[2].tick_params(axis='both', bottom=False, top=False,labelbottom=False)
        axs_est[2].grid(True)

        axs_est[3].plot(tt, ramps, 'k-')
        axs_est[3].plot(tt, x[:, 3], color = clr, linestyle = 'dashed')
        axs_est[3].fill_between(tt, x[:, 3]-3*std[:, 3], x[:, 3]+3*std[:, 3], color = clr, alpha=0.2)
        axs_est[3].set_ylabel('$r$ (deg)', fontsize = 14)
        axs_est[3].set_xlabel('time (s)', fontsize = 14)
        axs_est[3].set_xlim([0, tt[-1]+0.1])
        axs_est[3].set_ylim([-15, 15])
        axs_est[3].tick_params(axis='both', labelsize=12)
        axs_est[3].grid(True)

        plt.figure()
        plt.subplot(411)
        plt.title('Estimation Errors')
        plt.plot(tt, abs(estimate_error[:, 0].T))
        plt.ylabel('$\phi$ error')
        plt.xlim([0, tt[-1]+0.1])
        plt.grid()
        plt.subplot(412)
        plt.plot(tt, abs(estimate_error[:, 1].T))
        plt.ylabel('$\dot{\phi}$ error (1/s)')
        plt.xlim([0, tt[-1]+0.1])
        plt.grid()
        plt.subplot(413)
        plt.plot(tt, abs(estimate_error[:, 2].T))
        plt.ylabel('$l$ error (m)')
        plt.xlim([0, tt[-1]+0.1])
        plt.grid()
        plt.grid()
        plt.subplot(414)
        plt.plot(tt, abs(estimate_error[:, 3].T))
        plt.ylabel('ramp error (deg)')
        plt.xlim([0, tt[-1]+0.1])
        plt.grid()
        plt.xlabel('time (s)')
        plt.grid()
        
        plt.figure()
        plt.title("Mahalanobis Distance")
        plt.plot(tt, md, label = 'MD')
        plt.ylabel('MD')
        plt.legend()
        plt.xlabel('time (s)')

        plt.figure()
        for i in range(len(sensors)):
            plt.subplot(int(str(len(sensors)) + "1" + str(i+1)))
            plt.plot(tt, z[i, :], 'k-')
            plt.plot(tt, z_pred[:, i], '--', color = clr)
            plt.xlim([0, tt[-1]+0.1])
            plt.grid()
            if i == 0:
                plt.title("Measurements")
                #plt.legend(('actual', 'predicted'))
            elif i == len(sensors)-1:
                plt.xlabel("time (s)")
            if i < len(sensors)-1:
                plt.tick_params(axis='x', which='both', bottom=False, top=False,labelbottom=False)
        """
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
        """
        plt.show()
    
    if kidnap == False:
        return result
        
def kf_bank_test(dataset, subject, trial, side, N = 10, kalman_filter = 'ekf', kidnap = [0, 1, 2, 3], plot = False):
    # N: number of EKFs in the EKF-bank
    print("Randomized Kidnapping Test: ", dataset, "/", subject, "/", trial, '/', side)

    # 1) Use the incine experiment dataset
    if dataset == 'inclineExp' or dataset[0] == 'inclineExp':
        # load ground truth
        phases, phase_dots, step_lengths, ramps = get_Continuous_state_vars(subject, trial, side)
        # load measurements
        globalThighAngle, globalThighVelocity, atan2, globalFootAngle, ankleMoment, tibiaForce = get_Continuous_measurement_data(subject, trial, side)
        atan2 = get_Continuous_atan2_scale_shift(subject, trial, side, plot = False) # use the shifted & scalsed version
        kneeAngle, ankleAngle = get_Continuous_joints_angles(subject, trial, side)

        heel_strike_index = get_Continuous_heel_strikes(subject, trial, side) - get_Continuous_heel_strikes(subject, trial, side)[0]
        total_step = int(heel_strike_index[total_strides]) + 1
    
    # 2) Use the R01 dataset
    elif dataset == 'Reznick' or dataset[0] == 'Reznick':
        if 'i' in trial:
            speed = trial.split('i')[0]
            incline = 'i' + trial.split('i')[1]
        elif 'd' in trial:
            trial = trial.split('d')[0]
            incline = 'd' + trial.split('d')[1]

        (phases, phase_dots, step_lengths, ramps, globalThighAngle, globalThighVelocity, atan2, kneeAngle, ankleAngle) = load_Streaming_data(subject, speed)
        atan2 = get_Streaming_atan2_scale_shift(subject, speed, plot = False) # use the scaled & shifted version

        LHS = Streaming_data['Streaming'][subject]['Tread'][incline]['events']['LHS'][:][:,0]
        cutPoints = Streaming_data['Streaming'][subject]['Tread'][incline]['events']['cutPoints'][:]
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
    ramps = ramps[0 : total_step]
    
    globalThighAngle = globalThighAngle[0 : total_step]
    globalThighVelocity = globalThighVelocity[0 : total_step]
    atan2 = atan2[0 : total_step]
    globalFootAngle = globalFootAngle[0 : total_step]
    
    kneeAngle = kneeAngle[0 : total_step]
    ankleAngle = ankleAngle[0 : total_step]

    z_full = np.array([globalThighAngle, globalThighVelocity, atan2, globalFootAngle])
    z = z_full[sensor_id, :]

    if kalman_filter == 'ekf':
        # Create an EKF 
        sys = myStruct()
        sys.f = process_model
        sys.A = A
        sys.h = m_model
        sys.Psi = Psi
        sys.Q = Q_ekf
        sys.R = R_ekf
        sys.saturation = state_saturation
        sys.saturation_range = saturation_range
        sys.reset = reset
        sys.adapt = adaptive_cov

    elif kalman_filter == 'ukf':
        # Create an UKF 
        sys = myStruct()
        sys.f = process_model
        sys.h = m_model
        sys.Psi = Psi
        sys.Q = Q_ukf
        sys.R = R_ukf
        sys.alpha = 1e-3
        sys.beta = 2
        sys.kappa = 0
        sys.saturation = state_saturation
        sys.saturation_range = saturation_range
        sys.reset = reset
    
    x = np.zeros((N+1, total_step, 4))  # state estimate
    #std = np.zeros((N+1, total_step, 3))  # state estimate std
    estimate_error = np.zeros((N+1, total_step, 4))
    phase_converge_dist = np.zeros((N+1, total_step))
    r11 = 0
    r13 = 0
    r15 = 0
    r33 = 0
    r35 = 0
    r55 = 0

    for n in range(N+1):
        initial_x = np.array([phases[0], phase_dots[0], step_lengths[0], ramps[0]])
        init = myStruct()
        init.x = initial_x
        init.Sigma = initial_Sigma
        if kalman_filter == 'ekf':
            filter = extended_kalman_filter(sys, init)
            clr = 'r' # color
        elif kalman_filter == 'ukf':
            filter = unscented_kalman_filter(sys, init)
            clr = 'b'
        
        kidnap_percent_gait = np.random.uniform(0.05, 0.95)
        phase_kidnap =  np.random.uniform(0, 1)
        phase_dot_kidnap = np.random.uniform(0, 5)
        step_length_kidnap = np.random.uniform(0, 2)
        ramp_kidnap = np.random.uniform(-10, 10)
        state_kidnap = np.array([phase_kidnap, phase_dot_kidnap, step_length_kidnap, ramp_kidnap])

        kidnap_index = int(heel_strike_index[kidnap_stride] +
                        kidnap_percent_gait* (heel_strike_index[kidnap_stride+1] - heel_strike_index[kidnap_stride]))

        for i in range(total_step):
            # kidnap
            if i == kidnap_index and n > 0:
                filter.x[kidnap] = state_kidnap[kidnap]
            filter.prediction(dt)
            filter.correction(z[:, i], using_atan2)

            x[n, i,:] = filter.x
            #std[n, i, :] = np.sqrt(np.diag(filter.Sigma))
            estimate_error[n, i, :] = (filter.x - np.array([phases[i], phase_dots[i], step_lengths[i], ramps[i]]))
            if estimate_error[n, i, 0] > 0.5:
                estimate_error[n, i, 0] = estimate_error[n, i, 0] - 1
            elif estimate_error[n, i, 0] < -0.5:
                estimate_error[n, i, 0] = 1 + estimate_error[n, i, 0]
        if n == 0:
            start_check_idx = int(3/np.average(phase_dots)/dt)
            SE_phase = np.sum(estimate_error[0, start_check_idx:, 0] ** 2)
            T = len(estimate_error[0, start_check_idx:, 0])
            RMSE_phase = np.sqrt(SE_phase/T)
            if RMSE_phase > 0.1:
                print("The filter gets lost in nominal case.")
                print("R_11(%) = ", 0, "|| R_13(%) = ", 0, "|| R_15(%) = ", 0, "|| R_33(%) = ", 0 , "|| R_35(%) = ", 0, "|| R_55(%) = ", 0)
                print("---------------------------------------------------------------------------")
                #return (np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN)
                return (0, 0, 0, 0, 0, 0)

        if n > 0:
            idx_1 = int(heel_strike_index[kidnap_stride + 2])
            idx_3 = int(heel_strike_index[kidnap_stride + 4])
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
            #
            phase_recover_3 = np.all(phase_converge_dist[n,idx_3:] < phase_recover_thr)
            step_length_recover_3 = np.all(abs(x[n, idx_3:, 2] - x[0, idx_3:, 2]) < step_length_recover_thr)
            ramp_recover_3 = np.all(abs(x[n, idx_3:, 3] - x[0, idx_3:, 3]) < ramp_recover_thr)
            #
            phase_recover_5 = np.all(phase_converge_dist[n,idx_5:] < phase_recover_thr)
            step_length_recover_5 = np.all(abs(x[n, idx_5:, 2] - x[0, idx_5:, 2]) < step_length_recover_thr)
            ramp_recover_5 = np.all(abs(x[n, idx_5:, 3] - x[0, idx_5:, 3]) < ramp_recover_thr)

            track_111 = (phase_recover_1 and step_length_recover_1 and ramp_recover_1)
            track_133 = (phase_recover_1 and step_length_recover_3 and ramp_recover_3)
            track_155 = (phase_recover_1 and step_length_recover_5 and ramp_recover_5)
            track_333 = (phase_recover_3 and step_length_recover_3 and ramp_recover_3)
            track_355 = (phase_recover_3 and step_length_recover_5 and ramp_recover_5)
            track_555 = (phase_recover_5 and step_length_recover_5 and ramp_recover_5)
            print("n:", n)
            print("i, j, k | track_ijk = phase_i & step_len_j & ramp_k")
            print("1, 1, 1 |  ", track_111, "  =  ", phase_recover_1, " &&   ", step_length_recover_1, " &&   ", ramp_recover_1)
            print("1, 3, 3 |  ", track_133, "  =  ", phase_recover_1, " &&   ", step_length_recover_3, " &&   ", ramp_recover_3)
            print("1, 5, 5 |  ", track_155, "  =  ", phase_recover_1, " &&   ", step_length_recover_5, " &&   ", ramp_recover_5)
            print("3, 3, 3 |  ", track_333, "  =  ", phase_recover_3, " &&   ", step_length_recover_3, " &&   ", ramp_recover_3)
            print("3, 5, 5 |  ", track_355, "  =  ", phase_recover_3, " &&   ", step_length_recover_5, " &&   ", ramp_recover_5)
            print("5, 5, 5 |  ", track_555, "  =  ", phase_recover_5, " &&   ", step_length_recover_5, " &&   ", ramp_recover_5)

            if track_111:
                r11 += 1
            if track_133:
                r13 += 1
            if track_155:
                r15 += 1    
            if track_333:
                r33 += 1
            if track_355:
                r35 += 1    
            if track_555:
                r55 += 1

    robustness_11 = r11 / N * 100
    robustness_13 = r13 / N * 100
    robustness_15 = r15 / N * 100
    robustness_33 = r33 / N * 100
    robustness_35 = r35 / N * 100
    robustness_55 = r55 / N * 100

    print("R_11(%) = ", robustness_11, "|| R_13(%) = ", robustness_13, "|| R_15(%) = ", robustness_15,
         "|| R_33(%) = ", robustness_33 , "|| R_35(%) = ", robustness_35, "|| R_55(%) = ", robustness_55)
    print("---------------------------------------------------------------------------")

    if plot == True:
        global num_plots
        num_plots += 1

        # plot results
        tt = dt * np.arange(len(phases))
        #plt.figure("State Estimate")
        #plt.subplot(311)
        axs_est[0].set_title('Randomized Kidnapping Test', fontsize = 16)
        if num_plots == 1:
            axs_est[0].plot(tt, phases, 'k-', linewidth=2, label = 'ground truth')
        axs_est[0].plot(tt,  x[1:, :, 0].T, color = clr, linestyle = 'dashed', alpha = 0.3)
        if kalman_filter == 'ekf':    
            axs_est[0].plot(tt,  x[0, :, 0].T, color = clr, linestyle = 'dashed', linewidth=2, label='EKF')
        elif kalman_filter == 'ukf': 
            axs_est[0].plot(tt,  x[0, :, 0].T, color = clr, linestyle = 'dashed', linewidth=2, label='UKF')
        axs_est[0].set_ylabel('$\phi$', fontsize = 14)
        #axs_est[0].legend( fontsize = 14)
        #plt.legend(('ground truth', 'estimate'), bbox_to_anchor=(1, 1.05))
        #axs_est[0].set_ylim([0, 2])
        axs_est[0].set_xlim([0, tt[-1]+0.1])
        axs_est[0].tick_params(axis='both', bottom=False, top=False,labelbottom=False)
        axs_est[0].grid(True)
        #plt.subplot(312)
        axs_est[1].plot(tt, phase_dots, 'k-', linewidth=2)
        axs_est[1].plot(tt, x[1:, :, 1].T, color = clr, linestyle = 'dashed', alpha = 0.3)
        axs_est[1].plot(tt, x[0, :, 1].T, color = clr, linestyle = 'dashed', linewidth=2)
        axs_est[1].set_ylabel('$\dot{\phi}~(1/s)$', fontsize = 14)
        axs_est[1].set_xlim([0, tt[-1]+0.1])
        #axs_est[1].set_ylim([0, 2])
        axs_est[1].tick_params(axis='both', bottom=False, top=False,labelbottom=False)
        axs_est[1].grid(True)
        #plt.subplot(313)
        axs_est[2].plot(tt, step_lengths, 'k-', linewidth=2)
        axs_est[2].plot(tt, x[1:, :, 2].T, color = clr, linestyle = 'dashed', alpha = 0.3)
        axs_est[2].plot(tt, x[0, :, 2].T, color = clr, linestyle = 'dashed', linewidth=2)
        axs_est[2].set_ylabel('$l$', fontsize = 14)
        axs_est[2].set_xlim([0, tt[-1]+0.1])
        #axs_est[2].set_ylim([0, 2.05])
        axs_est[2].tick_params(axis='both', bottom=False, top=False,labelbottom=False)
        axs_est[2].grid(True)
        #
        axs_est[3].plot(tt, ramps, 'k-', linewidth=2)
        axs_est[3].plot(tt, x[1:, :, 3].T, color = clr, linestyle = 'dashed', alpha = 0.3)
        axs_est[3].plot(tt, x[0, :, 3].T, color = clr, linestyle = 'dashed', linewidth=2)
        axs_est[3].set_ylabel('r (deg)', fontsize = 14)
        axs_est[3].set_xlabel('time (s)', fontsize = 14)
        axs_est[3].set_xlim([0, tt[-1]+0.1])
        axs_est[3].set_ylim([-12, 12])
        axs_est[3].tick_params(axis='both', labelsize=12)
        axs_est[3].grid(True)
        
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
        plt.ylabel('$l$ error')
        plt.xlim([0, tt[-1]+0.1])
        plt.grid()
        plt.subplot(414)
        plt.plot(tt, abs(estimate_error[:, :, 3].T))
        plt.ylabel('ramp error')
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
        plt.ylabel('$\Delta l$')
        plt.xlim([0, tt[-1]+0.1])
        plt.grid()
        plt.subplot(414)
        plt.plot(tt, abs(x[1:,:,3] - x[0,:,3]).T, '--')
        plt.plot(tt[idx_1:], ramp_recover_thr*np.ones(len(tt[idx_1:])), 'k-', alpha = 0.4, linewidth=2)
        plt.plot(idx_1*dt*np.ones(5), np.linspace(0,2,5), 'k-', alpha = 0.4, linewidth=2)
        plt.plot(idx_3*dt*np.ones(5), np.linspace(0,2,5), 'b-', alpha = 0.4, linewidth=2)
        plt.plot(idx_5*dt*np.ones(5), np.linspace(0,2,5), 'r-', alpha = 0.4, linewidth=2)
        plt.ylabel('$\Delta$  r(deg)')
        plt.xlim([0, tt[-1]+0.1])
        plt.grid()

        plt.xlabel('time (s)')
        
        plt.show()
    
    return (robustness_11, robustness_13, robustness_15, robustness_33, robustness_35, robustness_55)

def kf_robustness(kidnap = True, kalman_filter = 'ekf', datasets = ['inclineExp']):
    # Calculate the robustness metrics:
    # 1) Nominal case: compute the total RMSE
    # 2) Randomized kidnapping test: compute the robustness (%)

    total_trials = 0
    skipped_trials = 0

    robustness_11 = 0
    robustness_13 = 0 
    robustness_15 = 0
    robustness_33 = 0
    robustness_35 = 0
    robustness_55 = 0

    SE_phase_total = 0
    SE_phase_dot_total = 0
    SE_step_length_total = 0
    SE_ramp_total = 0
    T_total = 0

    # Store trials with large RMSEs
    R_dict = dict()

    # Skip trials with problematic measurements
    with open('Continuous_data_incExp/Measurements_with_Nan.pickle', 'rb') as file:
        nan_dict = pickle.load(file)
    
    for dataset in datasets: # datasets = ['inclineExp', 'Reznick']
        R_dict[dataset] = dict()
        for subject in get_Continuous_subject_names(): 
            R_dict[dataset][subject] = dict()
            for trial in get_Continuous_trial_names(subject):
                if trial == 'subjectdetails':
                    continue
                R_dict[dataset][subject][trial] = dict()
                for side in ['left']: #, 'right'
                    if dataset == 'inclineExp' and nan_dict[subject][trial][side] == False:
                        print('inclineExp/' + subject + "/"+ trial + "/"+ side + ": trial skipped because of nans in measurements")
                        skipped_trials += 1
                        continue
                    if dataset == 'inclineExp' and subject == 'AB05' and ('s0x8' in trial):
                        print('inclineExp/' + subject + "/"+ trial + "/"+ side + ": AB05 s0x8 trial skipped because of abnormal measurements")
                        skipped_trials += 1
                        continue
                    if dataset == 'Reznick' and (((subject == 'AB08' or subject == 'AB04' or subject == 'AB07' or subject == 'AB09') and 's0x8' in trial) or subject == 'AB10'):
                        print('Reznick/' + subject + "/"+ trial + "/"+ side + ": trial skipped because of abnormal measurements")
                        skipped_trials += 1
                        continue
                    
                    if kidnap != False:
                        N = 5
                        (R_11, R_13, R_15, R_33, R_35, R_55) = kf_bank_test(dataset, subject, trial, side, N, kalman_filter = kalman_filter, kidnap = [0, 1, 2, 3])
                        if math.isnan(R_11) or math.isnan(R_13) or math.isnan(R_15) or math.isnan(R_33) or math.isnan(R_35) or math.isnan(R_55):
                            skipped_trials += 1
                            continue
                        else:
                            total_trials = total_trials + 1

                            robustness_11 += R_11
                            robustness_13 += R_13
                            robustness_15 += R_15
                            robustness_33 += R_33
                            robustness_35 += R_35
                            robustness_55 += R_55
                            print("**Current Average R_11 = %4.1f %%" % (robustness_11 / total_trials), 
                                "|| R_13 = %4.1f %%" % (robustness_13 / total_trials),
                                "|| R_15 = %4.1f %%" % (robustness_15 / total_trials),
                                "|| R_33 = %4.1f %%" % (robustness_33 / total_trials),
                                "|| R_35 = %4.1f %%" % (robustness_35 / total_trials),
                                "|| R_55 = %4.1f %%" % (robustness_55 / total_trials))
                            R_dict[dataset][subject][trial][side] = np.array([R_11, R_13, R_15, R_33, R_35, R_55])
                    
                    else:
                        (SE_phase, SE_phase_dot, SE_step_length, SE_ramp, T) = kf_test(dataset, subject, trial, side, kalman_filter)

                        R_dict[dataset][subject][trial][side] = np.array([np.sqrt(SE_phase/T), np.sqrt(SE_phase_dot/T),np.sqrt(SE_step_length/T), np.sqrt(SE_ramp/T)])

                        if np.sqrt(SE_phase/T) < 0.1:
                            SE_phase_total += SE_phase
                            SE_phase_dot_total += SE_phase_dot
                            SE_step_length_total += SE_step_length
                            SE_ramp_total += SE_ramp
                            T_total += T

                            print("Current RMSE phase = %5.3f" % np.sqrt(SE_phase_total/T_total))
                            print("Current RMSE phase_dot= %5.3f" % np.sqrt(SE_phase_dot_total/T_total))
                            print("Current RMSE step_length = %5.3f" % np.sqrt(SE_step_length_total/T_total))
                            print("Current RMSE ramp = %5.3f" % np.sqrt(SE_ramp_total/T_total))
                            print("----------------------------------------------------------------")

    if kidnap != False:
        robustness_11 = robustness_11 / total_trials
        robustness_13 = robustness_13 / total_trials
        robustness_15 = robustness_15 / total_trials
        robustness_33 = robustness_33 / total_trials
        robustness_35 = robustness_35 / total_trials
        robustness_55 = robustness_55 / total_trials

        print("==========================================")
        print("Overall Robustness_11 = %4.1f %%" % robustness_11)
        print("Overall Robustness_13 = %4.1f %%" % robustness_13)
        print("Overall Robustness_15 = %4.1f %%" % robustness_15)
        print("Overall Robustness_33 = %4.1f %%" % robustness_33)
        print("Overall Robustness_35 = %4.1f %%" % robustness_35)
        print("Overall Robustness_55 = %4.1f %%" % robustness_55)
        print("Total trials = %d" % total_trials)
        print("Skipped trials = %d" % skipped_trials)

        R_dict[dataset]['R11'] = robustness_11
        R_dict[dataset]['R13'] = robustness_13
        R_dict[dataset]['R15'] = robustness_15
        R_dict[dataset]['R33'] = robustness_33
        R_dict[dataset]['R35'] = robustness_35
        R_dict[dataset]['R55'] = robustness_55

        # TODO: move this to some other folder
        with open('Filters/'+kalman_filter+'_Robustness_data.pickle', 'wb') as file:
            pickle.dump(R_dict, file)

    else:
        print("Total RMSE phase = %5.3f" % np.sqrt(SE_phase_total/T_total))
        print("Total RMSE phase_dot = %5.3f" % np.sqrt(SE_phase_dot_total/T_total))
        print("Total RMSE step_length = %5.3f" % np.sqrt(SE_step_length_total/T_total))
        print("Total RMSE ramp = %5.3f" % np.sqrt(SE_ramp_total/T_total))

        R_dict[dataset]['RMSE_phase'] = np.sqrt(SE_phase_total/T_total)
        R_dict[dataset]['RMSE_phase_dot'] = np.sqrt(SE_phase_dot_total/T_total)
        R_dict[dataset]['RMSE_step_length'] = np.sqrt(SE_step_length_total/T_total)
        R_dict[dataset]['RMSE_ramp'] = np.sqrt(SE_ramp_total/T_total)
        
        # TODO: move this to some other folder
        with open('Filters/'+kalman_filter+'_RMSE_data.pickle', 'wb') as file:
            pickle.dump(R_dict, file)

def plot_kf_robustness_heatmap(kalman_filter = 'ekf'):
    
    with open('Filters/'+kalman_filter+'_RMSE_data.pickle', 'rb') as file:
        RMSE_data = pickle.load(file)

    with open('Filters/'+kalman_filter+'_Robustness_data.pickle',  'rb') as file:
        Robustness_data = pickle.load(file)  

    subjects = list(RMSE_data['inclineExp'].keys())[0:10]
    trials = list(RMSE_data['inclineExp']['AB01'].keys())
    RMSE_phase_map = np.zeros((len(subjects), len(trials)))
    RMSE_phase_dot_map = np.zeros((len(subjects), len(trials)))
    RMSE_step_length_map = np.zeros((len(subjects), len(trials)))
    RMSE_ramp_map = np.zeros((len(subjects), len(trials)))
    R11_map = np.zeros((len(subjects), len(trials)))
    R13_map = np.zeros((len(subjects), len(trials)))
    R33_map = np.zeros((len(subjects), len(trials)))
    for ks in range(len(subjects)):
        for kt in range(len(trials)):
            try:
                RMSE_phase_map[ks, kt] = RMSE_data['inclineExp'][subjects[ks]][trials[kt]]['left'][0]
            except:
                RMSE_phase_map[ks, kt] = np.NAN
            try:
                RMSE_phase_dot_map[ks, kt] = RMSE_data['inclineExp'][subjects[ks]][trials[kt]]['left'][1]
            except:
                RMSE_phase_dot_map[ks, kt] = np.NAN
            try:
                RMSE_step_length_map[ks, kt] = RMSE_data['inclineExp'][subjects[ks]][trials[kt]]['left'][2]
            except:
                RMSE_step_length_map[ks, kt] = np.NAN
            try:
                RMSE_ramp_map[ks, kt] = RMSE_data['inclineExp'][subjects[ks]][trials[kt]]['left'][3]
            except:
                RMSE_ramp_map[ks, kt] = np.NAN
            #
            try:
                R11_map[ks, kt] = Robustness_data['inclineExp'][subjects[ks]][trials[kt]]['left'][0]
            except:
                R11_map[ks, kt] = np.NAN
            try:
                R13_map[ks, kt] = Robustness_data['inclineExp'][subjects[ks]][trials[kt]]['left'][1]
            except:
                R13_map[ks, kt] = np.NAN
            try:
                R33_map[ks, kt] = Robustness_data['inclineExp'][subjects[ks]][trials[kt]]['left'][3]
            except:
                R33_map[ks, kt] = np.NAN
    
    plt.figure()
    sns.heatmap(RMSE_phase_map, annot = True, fmt=".2f", xticklabels = trials, yticklabels = subjects[0:9])
    plt.title('RMSE_phase_map; RMSE = %1.2f' % RMSE_data['inclineExp']['RMSE_phase'])
    plt.figure()
    sns.heatmap(RMSE_phase_dot_map, annot = True, fmt=".2f", xticklabels = trials, yticklabels = subjects[0:9])
    plt.title('RMSE_phase_dot_map; RMSE = %1.2f' % RMSE_data['inclineExp']['RMSE_phase_dot'])
    plt.figure()
    sns.heatmap(RMSE_step_length_map, annot = True, fmt=".2f", xticklabels = trials, yticklabels = subjects[0:9])
    plt.title('RMSE_step_length_map; RMSE = %1.2f' % RMSE_data['inclineExp']['RMSE_step_length'])
    plt.figure()
    sns.heatmap(RMSE_ramp_map, annot = True, fmt=".2f", xticklabels = trials, yticklabels = subjects[0:9])
    plt.title('RMSE_ramp_map; RMSE = %1.2f' % RMSE_data['inclineExp']['RMSE_ramp'])
    plt.figure()
    sns.heatmap(R11_map, annot = True, xticklabels = trials, yticklabels = subjects[0:9])
    plt.title('R11_map; average = %2.2f' % Robustness_data['inclineExp']['R11'])
    plt.figure()
    sns.heatmap(R13_map, annot = True, xticklabels = trials, yticklabels = subjects[0:9])
    plt.title('R13_map; average = %2.2f' % Robustness_data['inclineExp']['R13'])
    plt.figure()
    sns.heatmap(R33_map, annot = True, xticklabels = trials, yticklabels = subjects[0:9])
    plt.title('R33_map; average = %2.2f' % Robustness_data['inclineExp']['R33'])
    plt.show()

if __name__ == '__main__':
    #plot_kf_robustness_heatmap()

    dataset = 'inclineExp'

    subject = 'AB09'
    trial = 's1i2x5'
    side = 'left'

    #kf_bank_test(dataset, subject, trial, side, N = 10, kalman_filter = 'ekf', kidnap = [0, 1, 2, 3], plot = True)
    #kf_bank_test(dataset, subject, trial, side, N = 10, kalman_filter = 'ukf', kidnap = [0, 1, 2, 3], plot = True)

    #kf_test(dataset, subject, trial, side, kalman_filter = 'ekf', kidnap = True, plot = True)
    #kf_test(dataset, subject, trial, side, kalman_filter = 'ukf', kidnap = False, plot = True)

    #kf_robustness(kidnap = False, kalman_filter = 'ukf', datasets = ['inclineExp'])
    kf_robustness(kidnap = True, kalman_filter = 'ukf', datasets = ['inclineExp'])
    #print(" ==================== ")
    #kf_robustness(kidnap = False, kalman_filter = 'ekf', datasets = ['inclineExp'])
    #kf_robustness(kidnap = True, kalman_filter = 'ekf', datasets = ['inclineExp'])

    # Q=[0, 1e-3, 1e-3]
    #Total RMSE phase = 0.029
    #Total RMSE phase_dot = 0.041
    #Total RMSE step_length = 0.142