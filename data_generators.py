import h5py
import pickle
import numpy as np
from incline_experiment_utils import *

## This section is dedicated to getting the data: effect-walking-incline-and-speed-human-leg-kinematics-kinetics-and-emg
# README for the dataset can be found in 
# https://ieee-dataport.org/open-access/
dataset_location = '../'
filename = 'InclineExperiment.mat'
raw_walking_data = h5py.File(dataset_location + filename, 'r')

#Example of the structure:
#left_hip = raw_walking_data['Gaitcycle']['AB01']['s0x8i10']['kinematics']['jointangles']['left']['hip']['x']

#This will return a list of the subjects names
def get_subject_names():
    return raw_walking_data['Gaitcycle'].keys()

#This will return a concatenated array with all the trials for a subject
def get_joint_angle(subject, joint, direction):
    #direction: x/y/z given by the dataset
    return np.concatenate(list(joint_angle_generator(subject, joint, direction)), axis = 0)

#Ground reaction wrench (6-DOF force & moment) in {ANKLE FRAME}
def get_reaction_wrench(subject):
    #force_moment: force/moment
    #axis_ankle: x_ankle/y_ankle/z_ankle
    return np.concatenate(list(reaction_wrench_generator(subject)), axis = 1) # axis = 1

#Global thigh angle
def get_global_thigh_angle(subject):
    #axis_global: X/Y/Z
    return np.concatenate(list(global_thigh_angle_generator(subject)), axis = 1) # axis = 1

#This will return a corresponding axis for the data in the model
def get_phase(array):
    return np.concatenate(list(phase_generator(array.shape[0], 150)), axis = 0)

def get_phase_dot(subject, mode):
    return np.concatenate(list(phase_dot_generator(subject, mode)), axis = 0)

#This will return the step lengths for every trial
def get_step_length(subject, mode):
    return np.concatenate(list(step_length_generator(subject, mode)), axis = 0)

def get_ramp(subject, mode):
    return np.concatenate(list(ramp_generator(subject, mode)), axis = 0)

def get_time_step(subject, mode):
    return np.concatenate(list(time_step_generator(subject, mode)), axis = 0)


#====================================== Generator functions ===============================================#
def joint_angle_generator(subject, joint, direction = 'x', left=True, right=True):
    #Note: coords of the dataset are different from coords of the world frame
    with open('Continuous_data/KneeAngles_with_Nan.pickle', 'rb') as file:
        nan_dict = pickle.load(file)
    
    for trial in raw_walking_data['Gaitcycle'][subject].keys():
        if trial == 'subjectdetails':
            continue
        if(left == True):
            if nan_dict[subject][trial]['left'] == False:
                #print(subject + "/"+ trial + "/"+ 'left' + ": Trial skipped!")
                continue
            yield raw_walking_data['Gaitcycle'][subject][trial]['kinematics']['jointangles']['left'][joint][direction][:]
     
    for trial in  raw_walking_data['Gaitcycle'][subject].keys():
        if trial == 'subjectdetails':
            continue
        if(right == True):
            if nan_dict[subject][trial]['right'] == False:
                #print(subject + "/"+ trial + "/"+ 'right' + ": Trial skipped!")
                continue
            yield raw_walking_data['Gaitcycle'][subject][trial]['kinematics']['jointangles']['right'][joint][direction][:]

def reaction_wrench_generator(subject, left=True, right=True):
    #vicon_offset in {vicon frame}
    vicon_leftbelt_offset = np.array([-768, 885])*1e-3 #[m]
    vicon_rightbelt_offset = np.array([-255, 885])*1e-3 #[m]
    for trial in raw_walking_data['Gaitcycle'][subject].keys():
        if trial == 'subjectdetails':
            continue
        if(left == True):
            forceplate = raw_walking_data['Gaitcycle'][subject][trial]['kinetics']['forceplate'] #original unit: force[N]/moment[N*mm]/COP[mm]
            markers = raw_walking_data['Gaitcycle'][subject][trial]['kinematics']['markers'] #original unit: markers[mm]
            force_ankle_x, force_ankle_y, force_ankle_z, moment_ankle_x, moment_ankle_y, moment_ankle_z\
                 = wrench_ankle(forceplate['left']['force'], forceplate['left']['moment'], markers['left'], vicon_leftbelt_offset)
            
            yield np.array([force_ankle_x, force_ankle_y, force_ankle_z, moment_ankle_x, moment_ankle_y, moment_ankle_z])

    for trial in raw_walking_data['Gaitcycle'][subject].keys():
        if trial == 'subjectdetails':
            continue
        if(right == True):
            forceplate = raw_walking_data['Gaitcycle'][subject][trial]['kinetics']['forceplate'] #original unit: force[N]/moment[N*mm]/COP[mm]
            markers = raw_walking_data['Gaitcycle'][subject][trial]['kinematics']['markers'] #original unit: markers[mm]
            force_ankle_x, force_ankle_y, force_ankle_z, moment_ankle_x, moment_ankle_y, moment_ankle_z\
                 = wrench_ankle(forceplate['right']['force'], forceplate['right']['moment'], markers['right'], vicon_rightbelt_offset)
            
            yield np.array([force_ankle_x, force_ankle_y, force_ankle_z, moment_ankle_x, moment_ankle_y, moment_ankle_z])
            
def global_thigh_angle_generator(subject, left=True, right=True):
    with open('Continuous_data/GlobalThighAngles_with_Nan.pickle', 'rb') as file:
        nan_dict = pickle.load(file)
    
    for trial in raw_walking_data['Gaitcycle'][subject].keys():
        if trial == 'subjectdetails':
            continue
        if(left == True):
            if nan_dict[subject][trial]['left'] == False:
                continue

            jointangles = raw_walking_data['Gaitcycle'][subject][trial]['kinematics']['jointangles'] #deg
            data_shape = np.shape(jointangles['left']['pelvis']['x'][:])
            Y_th = np.zeros(data_shape)
            X_th = np.zeros(data_shape)
            Z_th = np.zeros(data_shape)
            for i in np.arange(data_shape[0]):
                for j in np.arange(data_shape[1]):
                    R_wp = YXZ_Euler_rotation(-jointangles['left']['pelvis']['x'][i,j], jointangles['left']['pelvis']['y'][i,j], -jointangles['left']['pelvis']['z'][i,j])
                    R_pt = YXZ_Euler_rotation(jointangles['left']['hip']['x'][i,j], -jointangles['left']['hip']['y'][i,j], -jointangles['left']['hip']['z'][i,j])
                    R_wt = R_wp @ R_pt
                    Y_th[i,j], X_th[i,j], Z_th[i,j] = YXZ_Euler_angles(R_wt)
            
            yield np.array([Y_th, X_th, Z_th])
        
    for trial in raw_walking_data['Gaitcycle'][subject].keys():
        if trial == 'subjectdetails':
            continue
        if(right == True):
            if nan_dict[subject][trial]['right'] == False:
                continue

            jointangles = raw_walking_data['Gaitcycle'][subject][trial]['kinematics']['jointangles'] #deg
            data_shape = np.shape(jointangles['right']['pelvis']['x'][:])
            Y_th = np.zeros(data_shape)
            X_th = np.zeros(data_shape)
            Z_th = np.zeros(data_shape)
            for i in np.arange(data_shape[0]):
                for j in np.arange(data_shape[1]):
                    R_wp = YXZ_Euler_rotation(-jointangles['right']['pelvis']['x'][i,j], -jointangles['right']['pelvis']['y'][i,j], jointangles['right']['pelvis']['z'][i,j])
                    R_pt = YXZ_Euler_rotation(jointangles['right']['hip']['x'][i,j], jointangles['right']['hip']['y'][i,j], jointangles['right']['hip']['z'][i,j])
                    R_wt = R_wp @ R_pt
                    Y_th[i,j], X_th[i,j], Z_th[i,j] = YXZ_Euler_angles(R_wt)

            yield np.array([Y_th, X_th, Z_th])

def phase_generator(n, length):
    #We really just care about getting n copies
    for i in range(n):
        yield np.linspace(0,1,length).reshape(1,length)

def phase_dot_generator(subject, mode):
    if mode == 'global_thigh_angle_Y' or mode == 'global_thigh_angVel_2hz' or mode == 'atan2':
        dir = 'Continuous_data/GlobalThighAngles_with_Nan.pickle'
    elif mode == 'knee_angle' or mode == 'ankle_angle':
        dir = 'Continuous_data/KneeAngles_with_Nan.pickle'
    
    with open(dir, 'rb') as file:
        nan_dict = pickle.load(file)

    for trial in raw_walking_data['Gaitcycle'][subject].keys():
        if trial == 'subjectdetails':
            continue
        
        #Get the h5py object pointer for the walking speed
        time_info_left = raw_walking_data['Gaitcycle'][subject][trial]['cycles']['left']['time']
        time_info_right = raw_walking_data['Gaitcycle'][subject][trial]['cycles']['right']['time']
        phase_step = 1/150

        for step_left in time_info_left:
            if nan_dict[subject][trial]['left'] == False:
                continue   
            #Calculate the step length for the given walking config
            #Get delta time of step
            delta_time_left = step_left[1] - step_left[0]
            #Set the steplength for the 
            yield np.full((1,150), phase_step / delta_time_left)

        for step_right in time_info_right:
            if nan_dict[subject][trial]['right'] == False:
                continue     
            #Get delta time of step
            delta_time_right = step_right[1] - step_right[0]
            #Set the steplength for the 
            yield np.full((1,150), phase_step / delta_time_right)

def step_length_generator(subject, mode):
    if mode == 'global_thigh_angle_Y' or mode == 'global_thigh_angVel_2hz' or mode == 'atan2':
        dir = 'Continuous_data/GlobalThighAngles_with_Nan.pickle'
    elif mode == 'knee_angle' or mode == 'ankle_angle':
        dir = 'Continuous_data/KneeAngles_with_Nan.pickle'
    
    with open(dir, 'rb') as file:
        nan_dict = pickle.load(file)

    for trial in raw_walking_data['Gaitcycle'][subject].keys():
        if trial == 'subjectdetails':
            continue
        
        #Get the h5py object pointer for the walking speed
        ptr = raw_walking_data['Gaitcycle'][subject][trial]['description'][1][0]
        walking_speed = raw_walking_data[ptr]
        time_info_left = raw_walking_data['Gaitcycle'][subject][trial]['cycles']['left']['time']
        time_info_right = raw_walking_data['Gaitcycle'][subject][trial]['cycles']['right']['time']

        for step_left in time_info_left:
            if nan_dict[subject][trial]['left'] == False:
                continue
            #Calculate the step length for the given walking config
            #Get delta time of step
            delta_time_left = step_left[149] - step_left[0]
            #Set the steplength for the 
            yield np.full((1,150), walking_speed * delta_time_left)

        for step_right in time_info_right:
            if nan_dict[subject][trial]['right'] == False:
                continue
            #Get delta time of step
            delta_time_right = step_right[149] - step_right[0]
            #Set the steplength for the 
            yield np.full((1,150), walking_speed * delta_time_right)

def ramp_generator(subject, mode):
    if mode == 'global_thigh_angle_Y' or mode == 'global_thigh_angVel_2hz' or mode == 'atan2':
        dir = 'Continuous_data/GlobalThighAngles_with_Nan.pickle'
    elif mode == 'knee_angle' or mode == 'ankle_angle':
        dir = 'Continuous_data/KneeAngles_with_Nan.pickle'
    
    with open(dir, 'rb') as file:
        nan_dict = pickle.load(file)

    #Generate for the left leg
    for trial in raw_walking_data['Gaitcycle'][subject].keys():
        if trial == 'subjectdetails':
            continue
        if nan_dict[subject][trial]['left'] == False:
            continue
        
        #Get the h5py object pointer for the walking speed
        ptr = raw_walking_data['Gaitcycle'][subject][trial]['description'][1][1]
        ramp = raw_walking_data[ptr]

        #This just gets the amount of ramps you need per step
        time_info = raw_walking_data['Gaitcycle'][subject][trial]['cycles']['left']['time']
       
        for step in time_info:        
            yield np.full((1,150), ramp)

    #Generate for the right leg
    for trial in raw_walking_data['Gaitcycle'][subject].keys():
        if trial == 'subjectdetails':
            continue
        if nan_dict[subject][trial]['right'] == False:
            continue
        
        #Get the h5py object pointer for the walking speed
        ptr = raw_walking_data['Gaitcycle'][subject][trial]['description'][1][1]
        ramp = raw_walking_data[ptr]

        #This just gets the amount of ramps you need per step
        time_info = raw_walking_data['Gaitcycle'][subject][trial]['cycles']['right']['time']
       
        for step in time_info:        
            yield np.full((1,150), ramp)

def time_step_generator(subject, mode):
    if mode == 'global_thigh_angle_Y' or mode == 'global_thigh_angVel_2hz' or mode == 'atan2':
        dir = 'Continuous_data/GlobalThighAngles_with_Nan.pickle'
    
    elif mode == 'knee_angle' or mode == 'ankle_angle':
        dir = 'Continuous_data/KneeAngles_with_Nan.pickle'
    
    with open(dir, 'rb') as file:
        nan_dict = pickle.load(file)
    
    for trial in raw_walking_data['Gaitcycle'][subject].keys():
        if trial == 'subjectdetails':
            continue
        #Get the h5py object pointer for the walking speed
        time_info_left = raw_walking_data['Gaitcycle'][subject][trial]['cycles']['left']['time']
        time_info_right = raw_walking_data['Gaitcycle'][subject][trial]['cycles']['right']['time']

        for step_left in time_info_left:
            if nan_dict[subject][trial]['left'] == False:
                continue
            delta_time_left = step_left[1] - step_left[0]
            yield np.full((1,150), delta_time_left)

        for step_right in time_info_right:
            if nan_dict[subject][trial]['right'] == False:
                continue   
            delta_time_right = step_right[1] - step_right[0]
            yield np.full((1,150), delta_time_right)

if __name__ == '__main__':
    subject_names = get_subject_names()
    
    # Global thigh angle
    """
    dict_gt = dict()
    for subject in subject_names:
        dict_gt[subject] = get_global_thigh_angle(subject)
    with open('Gait_cycle_data/Global_thigh_angle_withoutNan.npz', 'wb') as file:
        np.savez(file, **dict_gt)
    """
    
    # Global thigh angle velocities and atan2
    #"""
    #dict_gt_bp = dict()
    #dict_gtv_5hz = dict()
    #dict_gtv_2x5hz = dict()
    dict_gtv_2hz = dict()
    #dict_atan2 = dict()
    dict_atan2_s = dict()

    with open('Gait_cycle_data/Global_thigh_angle_withoutNan.npz', 'rb') as file:
        gt_Y = np.load(file)
        for subject in subject_names:
            dt = get_time_step(subject, mode = 'global_thigh_angle_Y')
            #gt_bp = np.zeros(np.shape(gt_Y[subject][0]))
            #gtv_5hz = np.zeros(np.shape(gt_Y[subject][0]))
            #gtv_2x5hz = np.zeros(np.shape(gt_Y[subject][0]))
            gtv_2hz = np.zeros(np.shape(gt_Y[subject][0]))
            #atan2 = np.zeros(np.shape(gt_Y[subject][0]))
            atan2_s = np.zeros(np.shape(gt_Y[subject][0]))
            for i in range(np.shape(gt_Y[subject][0])[0]):
                # compute angular velocities w/ LP filters w/ different cutoff frequencies
                v = np.diff(gt_Y[subject][0][i, :]) / dt[i, 0]
                gtv = np.insert(v, 0, 0)
                gtv_stack = np.array([gtv, gtv, gtv, gtv, gtv]).reshape(-1)
                #gtv_5hz_stack = butter_lowpass_filter(gtv_stack, 5, 1/dt[i, 0], order = 1)
                #gtv_2x5hz_stack = butter_lowpass_filter(gtv_stack, 2.5, 1/dt[i, 0], order = 1)
                gtv_2hz_stack = butter_lowpass_filter(gtv_stack, 2, 1/dt[i, 0], order = 1)

                #gtv_5hz[i, :] = gtv_5hz_stack[2 * len(gt_Y[subject][0][i, :]): 3 * len(gt_Y[subject][0][i, :])]
                #gtv_2x5hz[i, :] = gtv_2x5hz_stack[2 * len(gt_Y[subject][0][i, :]): 3 * len(gt_Y[subject][0][i, :])]
                gtv_2hz[i, :] = gtv_2hz_stack[2 * len(gt_Y[subject][0][i, :]): 3 * len(gt_Y[subject][0][i, :])]

                # compute atan2 w/ a band-pass filter
                gt_stack = np.array([gt_Y[subject][0][i, :], gt_Y[subject][0][i, :], gt_Y[subject][0][i, :],\
                                   gt_Y[subject][0][i, :], gt_Y[subject][0][i, :]]).reshape(-1)
                gt_bp_stack = butter_bandpass_filter(gt_stack, 0.5, 2, 1/dt[i, 0], order = 2)
                gt_bp = gt_bp_stack[2 * len(gt_Y[subject][0][i, :]): 3 * len(gt_Y[subject][0][i, :])]

                #gt_bp = butter_bandpass_filter(gt_Y[subject][0][i, :], 0.5, 2, 1/dt[i, 0], order = 1)
                v_bp = np.diff(gt_bp) / dt[i, 0]
                #gtv_bp = butter_lowpass_filter(np.insert(v_bp, 0, 0), 2, 1/dt[i, 0], order = 1)
                gtv_bp = np.insert(v_bp, 0, 0)
                gtv_bp_stack = np.array([gtv_bp, gtv_bp, gtv_bp, gtv_bp, gtv_bp]).reshape(-1)
                gtv_blp_stack = butter_lowpass_filter(gtv_bp_stack, 2, 1/dt[i, 0], order = 1)
                gtv_blp = gtv_blp_stack[2 * len(gt_Y[subject][0][i, :]): 3 * len(gt_Y[subject][0][i, :])]
                
                atan2_s[i, :] = np.arctan2(-gtv_blp/(2*np.pi*0.8), gt_bp)
                for j in range(np.shape(atan2_s[i, :])[0]):
                    if atan2_s[i, j] < 0:
                        atan2_s[i, j] = atan2_s[i, j] + 2 * np.pi
            #dict_gt_bp[subject] = gt_bp
            #dict_gtv_5hz[subject] = gtv_5hz
            #dict_gtv_2x5hz[subject] = gtv_2x5hz
            dict_gtv_2hz[subject] = gtv_2hz
            dict_atan2_s[subject] = atan2_s

    #with open('Gait_cycle_data/Global_thigh_angle_bp.npz', 'wb') as file:
        #np.savez(file, **dict_gt_bp)
    #with open('Gait_cycle_data/Global_thigh_angVel_5hz.npz', 'wb') as file:
    #    np.savez(file, **dict_gtv_5hz)
    #with open('Gait_cycle_data/Global_thigh_angVel_2x5hz.npz', 'wb') as file:
    #    np.savez(file, **dict_gtv_2x5hz)
    with open('Gait_cycle_data/Global_thigh_angVel_2hz_withoutNan.npz', 'wb') as file:
        np.savez(file, **dict_gtv_2hz)
    with open('Gait_cycle_data/atan2_s_withoutNan.npz', 'wb') as file:
        np.savez(file, **dict_atan2_s)
    #"""

    # Knee and ankle 
    """
    dict_knee = dict()
    for subject in subject_names:
        dict_knee[subject] = -get_joint_angle(subject, 'knee', 'x')
    with open('Gait_cycle_data/knee_angle.npz', 'wb') as file:
        np.savez(file, **dict_knee)
    
    dict_ankle = dict()
    for subject in subject_names:
        dict_ankle[subject] = -get_joint_angle(subject, 'ankle', 'x')
    with open('Gait_cycle_data/ankle_angle.npz', 'wb') as file:
        np.savez(file, **dict_ankle)
    """

    
    """
    dict_rw = dict()
    for subject in subject_names:
        dict_rw[subject] = get_reaction_wrench(subject)
    with open('Reaction_wrench.npz', 'wb') as file:
        np.savez(file, **dict_rw)

    # test code
    
    dict_ja = dict()
    for subject in subject_names:
        dict_ja[subject] = get_joint_angle(subject, 'ankle', 'y')
    with open('ankle_angle.npy', 'wb') as file:
        np.savez(file, **dict_ja)
    
    subject = 'AB01'
    RW = get_reaction_wrench(subject, 'force', 'x_ankle')
    with open('RW_test.npz', 'wb') as file:
        np.savez(file, RW)
    print(np.shape(RW))
    """
    