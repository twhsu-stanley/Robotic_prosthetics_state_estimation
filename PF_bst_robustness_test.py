import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from Filters.EKF import *
from Filters.UKF import *
from Filters.PF_bootstrap import *
from model_framework import *
from continuous_data_incExp import *
from continuous_data_R01 import *
from basis_model_fitting import measurement_noise_covariance, heteroscedastic_measurement_noise_covariance
import csv

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
initial_x = np.array([0.3, 1, 1]) # mid-stance
initial_Sigma = np.diag([1e-1, 1e-1, 1e-1])
Q = np.diag([1e-2, 1e-1, 1e-1]) * dt
U = np.diag([2, 2, 2])
R = U @ measurement_noise_covariance(*sensors) @ U.T

# Skip trials with problematic measurements
with open('Continuous_data/Measurements_with_Nan.pickle', 'rb') as file:
    nan_dict = pickle.load(file)

# Stride in which kidnapping occurs
kidnap_stride = 4
total_strides = 15

# Roecover Criteria
phase_recover_thr = 0.05
step_length_recover_thr = 0.1

def pf_test(dataset, subject, trial, side = 'left', heteroscedastic = False, kidnap = False, plot = False):
    print("EKF Test: ", dataset,"/",subject, "/", trial, '/', side , "| Heteroscedastic R:", heteroscedastic)
    
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
    
    globalThighAngle = globalThighAngle[0 : total_step]
    globalThighVelocity = globalThighVelocity[0 : total_step]
    atan2 = atan2[0 : total_step]
    
    kneeAngle = kneeAngle[0 : total_step]
    ankleAngle = ankleAngle[0 : total_step]

    z_full = np.array([globalThighAngle, globalThighVelocity, atan2])
    z = z_full[sensor_id, :]

    # Create a Boostrapt Particle fiter ##################################################################
    sys = myStruct()
    sys.f = process_model
    sys.h = m_model
    sys.Psi = Psi
    sys.Q = Q
    sys.R = R
    init = myStruct()
    init.n = 1000 # number of partcles
    init.mu = initial_x
    init.Sigma = initial_Sigma
    filter = bootstrap_particle_filter(sys, init)
    #################################################################################
    
    if kidnap != False:
        kidnap_index = np.random.randint(heel_strike_index[kidnap_stride], heel_strike_index[kidnap_stride+1])
        phase_kidnap =  np.random.uniform(0, 1)
        phase_dot_kidnap = np.random.uniform(0, 5)
        step_length_kidnap = np.random.uniform(0, 2)
        state_kidnap = np.array([phase_kidnap, phase_dot_kidnap, step_length_kidnap])
        print("state_kidnap = [%4.2f, %4.2f, %4.2f]" % (state_kidnap[0], state_kidnap[1], state_kidnap[2]))

    x = np.zeros((total_step, 3))  # state estimate
    std = np.zeros((total_step, 3))  # state estimate
    neff = np.zeros(total_step)

    #kneeAngle_kmd = np.zeros((total_step, 1))
    #ankleAngle_kmd = np.zeros((total_step, 1))
    
    for i in range(500):
        if kidnap != False:
            if i == kidnap_index:
                filter.x[kidnap] = state_kidnap[kidnap]
        filter.particles_propagation(dt)
        filter.importance_measurement(z[:, i], using_atan2)
        filter.mean_std()

        if i % 140 == 0:
            filter.plot_2d()
            plt.show()

        x[i,:] = filter.mu
        std[i, :] = filter.std
        neff[i] = filter.Neff

        ## Joints control commands 
        #joint_angles = joints_control(x[i,0], x[i,1], x[i,2])
        #kneeAngle_kmd[i] = joint_angles[0]
        #ankleAngle_kmd[i] = joint_angles[1]
    

    if plot == True:
        # plot results
        tt = dt * np.arange(len(phases))
        plt.figure("State Estimate")
        plt.subplot(311)
        plt.title('EKF Robustness Test')
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
        plt.title("Effective Sample Size: Neff")
        plt.plot(tt, neff, '.-')
        plt.xlim([0, tt[-1]+0.1])
        plt.ylabel('Neff')
        plt.xlabel('time (s)')
        plt.grid()

        plt.figure("Measurements")
        for i in range(len(sensors)):
            plt.subplot(int(str(len(sensors)) + "1" + str(i+1)))
            plt.plot(tt, z[i, :], 'k-')
            plt.xlim([0, tt[-1]+0.1])
            plt.grid()
            if i == 0:
                plt.title("Measurements")
            elif i == len(sensors)-1:
                plt.xlabel("time (s)")
        
        plt.show()

if __name__ == '__main__':
    dataset = 'inclineExp'
    subject = 'AB05'
    trial = 's1x2'
    side = 'left'
    if nan_dict[subject][trial+'i0'][side] == False:
        print(subject + "/"+ trial + "/"+ side+ ": This trial should be skipped!")
    pf_test(dataset, subject, trial, side, heteroscedastic = False, kidnap = False, plot = True)

    #dataset = 'Reznick'
    #subject = 'AB07'
    #trial = 's1'
    #pf_test(dataset, subject, trial, side, heteroscedastic = False, kidnap = [0,1,2], plot = True)
