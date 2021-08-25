import h5py
import pickle
import numpy as np
from numpy.core.fromnumeric import mean
from numpy.lib.twodim_base import tri
from incline_experiment_utils import *
import time

dataset_location = '../'
filename = 'InclineExperiment.mat'
raw_walking_data = h5py.File(dataset_location + filename, 'r')

def get_subject_names():
    return raw_walking_data['Gaitcycle'].keys()

def jointAngles_statistics(joint):
    """ Compute mean and standard deviaton of joint angles across the phase
    """
    # Input: 
    #   joint = 'knee' or 'ankle'
    
    subject_names = get_subject_names()

    data_mean_std = dict()
    
    for trial in raw_walking_data['Gaitcycle']['AB01'].keys():
        if trial == 'subjectdetails':
            continue
        data_mean_std[trial] = dict()
        for subject in subject_names:
            data_left = -raw_walking_data['Gaitcycle'][subject][trial]['kinematics']['jointangles']['left'][joint]['x'][:]
            data_right = -raw_walking_data['Gaitcycle'][subject][trial]['kinematics']['jointangles']['right'][joint]['x'][:]
            
            # Add offset to knee angles if "over extension" (knee angle > 0) occurs
            if joint == 'knee':
                if np.max(data_left) > 0:
                    data_left -= np.max(data_left)
                
                if np.max(data_right) > 0:
                    data_right -= np.max(data_right)
            
            if joint == 'foot':
                data_left -= 90
                data_right -= 90
        
            if subject == 'AB01':
                data = data_left
                data = np.vstack((data, data_right))
            else:
                data = np.vstack((data, data_left))
                data = np.vstack((data, data_right))

        data_mean_std[trial]['mean'] = np.mean(data, axis = 0)
        data_mean_std[trial]['std'] = np.std(data, axis = 0)
    
        #plt.figure(trial)
        #plt.plot(range(150), data.T, 'k-', alpha = 0.2)
        #plt.plot(range(150), data_mean_std[trial]['mean'])
        #plt.plot(range(150), data_mean_std[trial]['mean'] + 3 * data_mean_std[trial]['std'])
        #plt.plot(range(150), data_mean_std[trial]['mean'] - 3 * data_mean_std[trial]['std'])
        #plt.show()

    with open(('Gait_data_statistics/' + joint + 'Angles_mean_std.pickle'), 'wb') as file:
    	pickle.dump(data_mean_std, file)

def globalThighAngles_statistics():
    """ 
    1. Compute mean and standard deviaton of global thigh angles across the phase
    *2. Store global thigh angles in a dictionary for future use
    """
    subject_names = get_subject_names()

    globalThighAngles_mean_std = dict()
    globalThighAngles = dict()
    
    for trial in raw_walking_data['Gaitcycle']['AB01'].keys():
        if trial == 'subjectdetails':
            continue
        print("trial: ", trial)

        globalThighAngles_mean_std[trial] = dict()
        globalThighAngles[trial] = dict()

        for subject in subject_names:
            print("subject: ", subject)

            globalThighAngles[trial][subject] = dict()
            
            jointangles = raw_walking_data['Gaitcycle'][subject][trial]['kinematics']['jointangles'] #deg
            
            # left
            data_shape = np.shape(jointangles['left']['pelvis']['x'][:])
            globalThighAngles_left = np.zeros(data_shape)
            for i in np.arange(data_shape[0]):
                for j in np.arange(data_shape[1]):
                    R_wp = YXZ_Euler_rotation(-jointangles['left']['pelvis']['x'][i,j], jointangles['left']['pelvis']['y'][i,j], -jointangles['left']['pelvis']['z'][i,j])
                    R_pt = YXZ_Euler_rotation(jointangles['left']['hip']['x'][i,j], -jointangles['left']['hip']['y'][i,j], -jointangles['left']['hip']['z'][i,j])
                    R_wt = R_wp @ R_pt
                    globalThighAngles_left[i,j], _, _ = YXZ_Euler_angles(R_wt)
            globalThighAngles[trial][subject]['left'] = globalThighAngles_left

            # right
            data_shape = np.shape(jointangles['right']['pelvis']['x'][:])
            globalThighAngles_right = np.zeros(data_shape)
            for i in np.arange(data_shape[0]):
                for j in np.arange(data_shape[1]):
                    R_wp = YXZ_Euler_rotation(-jointangles['right']['pelvis']['x'][i,j], -jointangles['right']['pelvis']['y'][i,j], jointangles['right']['pelvis']['z'][i,j])
                    R_pt = YXZ_Euler_rotation(jointangles['right']['hip']['x'][i,j], jointangles['right']['hip']['y'][i,j], jointangles['right']['hip']['z'][i,j])
                    R_wt = R_wp @ R_pt
                    globalThighAngles_right[i,j], _, _ = YXZ_Euler_angles(R_wt)
            globalThighAngles[trial][subject]['right'] = globalThighAngles_right

            if subject == 'AB01':
                data = globalThighAngles_left
                data = np.vstack((data, globalThighAngles_right))
            else:
                data = np.vstack((data, globalThighAngles_left))
                data = np.vstack((data, globalThighAngles_right))

        globalThighAngles_mean_std[trial]['mean'] = np.mean(data, axis = 0)
        globalThighAngles_mean_std[trial]['std'] = np.std(data, axis = 0)

        #plt.figure(trial)
        #plt.plot(range(150), data.T, 'k-', alpha = 0.2)
        #plt.plot(range(150), globalThighAngles_mean_std[trial]['mean'])
        #plt.plot(range(150), globalThighAngles_mean_std[trial]['mean'] + 3 * globalThighAngles_mean_std[trial]['std'])
        #plt.plot(range(150), globalThighAngles_mean_std[trial]['mean'] - 3 * globalThighAngles_mean_std[trial]['std'])
        #plt.show()

        # Store data iteratively
        #with open('Gait_data_statistics/globalThighAngles_mean_std.pickle', 'wb') as file:
    	#    pickle.dump(globalThighAngles_mean_std, file)
    
        #with open('Gait_training_data/globalThighAngles_original.pickle', 'wb') as file:
    	#    pickle.dump(globalThighAngles, file)

    with open('Gait_data_statistics/globalThighAngles_mean_std.pickle', 'wb') as file:
    	pickle.dump(globalThighAngles_mean_std, file)
    
    with open('Gait_training_data/globalThighAngles_original.pickle', 'wb') as file:
    	pickle.dump(globalThighAngles, file)

def derivedMeasurements_statistics():
    """ 
    1. Compute mean and standard deviaton of global thigh angle velocities and the Atn2 signal across the phase
    *2. Store global thigh angle velocities and the Atn2 signal in a dictionary for future use
    """

    with open('Gait_training_data/globalThighAngles_original.pickle', 'rb') as file:
        globalThighAngles = pickle.load(file)

    subject_names = get_subject_names()

    globalThighVelocities_mean_std = dict()
    atan2_mean_std = dict()
    globalThighVelocities = dict()
    atan2 = dict()
    
    for trial in raw_walking_data['Gaitcycle']['AB01'].keys():
        if trial == 'subjectdetails':
            continue
        print("trial: ", trial)

        globalThighVelocities_mean_std[trial] = dict()
        atan2_mean_std[trial] = dict()
        globalThighVelocities[trial] = dict()
        atan2[trial] = dict()

        for subject in subject_names:
            print("subject: ", subject)

            globalThighVelocities[trial][subject] = dict()
            atan2[trial][subject] = dict()

            data_left = globalThighAngles[trial][subject]['left']
            data_right = globalThighAngles[trial][subject]['right']

            time_info_left = raw_walking_data['Gaitcycle'][subject][trial]['cycles']['left']['time']
            time_info_right = raw_walking_data['Gaitcycle'][subject][trial]['cycles']['right']['time']

            dt_left = []
            for step_left in time_info_left:
                delta_time_left = step_left[1] - step_left[0]
                dt_left.append(np.full((1,150), delta_time_left))
            dt_left = np.squeeze(np.array(dt_left))

            dt_right = []
            for step_right in time_info_right:
                delta_time_right = step_right[1] - step_right[0]
                dt_right.append(np.full((1,150), delta_time_right))
            dt_right = np.squeeze(np.array(dt_right))

            # Compute global thigh angle velocities and the atan2 signal
            # left
            globalThighVelocities_left = np.zeros(np.shape(data_left))
            atan2_left = np.zeros(np.shape(data_left))
            for i in range(np.shape(data_left)[0]):
                v = np.diff(data_left[i, :]) / dt_left[i, 0]
                gtv = np.insert(v, 0, 0)
                gtv_stack = np.array([gtv, gtv, gtv, gtv, gtv]).reshape(-1)
                gtv_lp_stack = butter_lowpass_filter(gtv_stack, 2, 1/dt_left[i, 0], order = 1)
                globalThighVelocities_left[i, :] = gtv_lp_stack[2 * len(data_left[i, :]): 3 * len(data_left[i, :])]

                # compute atan2 w/ a band-pass filter
                gt_stack = np.array([data_left[i, :], data_left[i, :], data_left[i, :],\
                                   data_left[i, :], data_left[i, :]]).reshape(-1)
                gt_bp_stack = butter_bandpass_filter(gt_stack, 0.5, 2, 1/dt_left[i, 0], order = 2)
                gt_bp = gt_bp_stack[2 * len(data_left[i, :]): 3 * len(data_left[i, :])]

                v_bp = np.diff(gt_bp) / dt_left[i, 0]
                gtv_bp = np.insert(v_bp, 0, 0)
                gtv_bp_stack = np.array([gtv_bp, gtv_bp, gtv_bp, gtv_bp, gtv_bp]).reshape(-1)
                gtv_blp_stack = butter_lowpass_filter(gtv_bp_stack, 2, 1/dt_left[i, 0], order = 1)
                gtv_blp = gtv_blp_stack[2 * len(data_left[i, :]): 3 * len(data_left[i, :])]
                
                atan2_left[i, :] = np.arctan2(-gtv_blp/(2*np.pi*0.8), gt_bp)  # arctan2 and scaling
                for j in range(np.shape(atan2_left[i, :])[0]):
                    if atan2_left[i, j] < 0:
                        atan2_left[i, j] = atan2_left[i, j] + 2 * np.pi
                
                globalThighVelocities[trial][subject]['left'] = globalThighVelocities_left
                atan2[trial][subject]['left'] = atan2_left

            # right
            globalThighVelocities_right = np.zeros(np.shape(data_right))
            atan2_right = np.zeros(np.shape(data_right))
            for i in range(np.shape(data_right)[0]):
                v = np.diff(data_right[i, :]) / dt_right[i, 0]
                gtv = np.insert(v, 0, 0)
                gtv_stack = np.array([gtv, gtv, gtv, gtv, gtv]).reshape(-1)
                gtv_lp_stack = butter_lowpass_filter(gtv_stack, 2, 1/dt_right[i, 0], order = 1)
                globalThighVelocities_right[i, :] = gtv_lp_stack[2 * len(data_right[i, :]): 3 * len(data_right[i, :])]

                # compute atan2 w/ a band-pass filter
                gt_stack = np.array([data_right[i, :], data_right[i, :], data_right[i, :],\
                                   data_right[i, :], data_right[i, :]]).reshape(-1)
                gt_bp_stack = butter_bandpass_filter(gt_stack, 0.5, 2, 1/dt_right[i, 0], order = 2)
                gt_bp = gt_bp_stack[2 * len(data_right[i, :]): 3 * len(data_right[i, :])]

                v_bp = np.diff(gt_bp) / dt_right[i, 0]
                gtv_bp = np.insert(v_bp, 0, 0)
                gtv_bp_stack = np.array([gtv_bp, gtv_bp, gtv_bp, gtv_bp, gtv_bp]).reshape(-1)
                gtv_blp_stack = butter_lowpass_filter(gtv_bp_stack, 2, 1/dt_right[i, 0], order = 1)
                gtv_blp = gtv_blp_stack[2 * len(data_right[i, :]): 3 * len(data_right[i, :])]
                
                atan2_right[i, :] = np.arctan2(-gtv_blp/(2*np.pi*0.8), gt_bp)  # arctan2 and scaling
                for j in range(np.shape(atan2_right[i, :])[0]):
                    if atan2_right[i, j] < 0:
                        atan2_right[i, j] = atan2_right[i, j] + 2 * np.pi
                
                globalThighVelocities[trial][subject]['right'] = globalThighVelocities_right
                atan2[trial][subject]['right'] = atan2_right

            if subject == 'AB01':
                data_1 = globalThighVelocities_left
                data_1 = np.vstack((data_1, globalThighVelocities_right))

                data_2 = atan2_left
                data_2 = np.vstack((data_2, atan2_right))
            else:
                data_1 = np.vstack((data_1, globalThighVelocities_left))
                data_1 = np.vstack((data_1, globalThighVelocities_right))

                data_2 = np.vstack((data_2, atan2_left))
                data_2 = np.vstack((data_2, atan2_right))

        globalThighVelocities_mean_std[trial]['mean'] = np.mean(data_1, axis = 0)
        globalThighVelocities_mean_std[trial]['std'] = np.std(data_1, axis = 0)
        atan2_mean_std[trial]['mean'] = np.mean(data_2, axis = 0)
        atan2_mean_std[trial]['std'] = np.std(data_2, axis = 0)

        #plt.figure()
        #plt.plot(range(150), data_1.T, 'k-', alpha = 0.2)
        #plt.plot(range(150), globalThighVelocities_mean_std[trial]['mean'])
        #plt.plot(range(150), globalThighVelocities_mean_std[trial]['mean'] + 3 * globalThighVelocities_mean_std[trial]['std'])
        #plt.plot(range(150), globalThighVelocities_mean_std[trial]['mean'] - 3 * globalThighVelocities_mean_std[trial]['std'])
        
        #plt.figure()
        #plt.plot(range(150), data_2.T, 'k-', alpha = 0.2)
        #plt.plot(range(150), atan2_mean_std[trial]['mean'])
        #plt.plot(range(150), atan2_mean_std[trial]['mean'] + 3 * atan2_mean_std[trial]['std'])
        #plt.plot(range(150), atan2_mean_std[trial]['mean'] - 3 * atan2_mean_std[trial]['std'])
        #plt.show()
    
    with open('Gait_data_statistics/globalThighVelocities_mean_std.pickle', 'wb') as file:
    	pickle.dump(globalThighVelocities_mean_std, file)
    
    with open('Gait_data_statistics/atan2_mean_std.pickle', 'wb') as file:
    	pickle.dump(atan2_mean_std, file)

    with open('Gait_training_data/globalThighVelocities_original.pickle', 'wb') as file:
    	pickle.dump(globalThighVelocities, file)
    
    with open('Gait_training_data/atan2_original.pickle', 'wb') as file:
    	pickle.dump(atan2, file)

def ankleMoment_statistics():
    """ Compute mean and standard deviaton of ankle moment (normalized w.r.t. subject's weight) across the phase
    """    
    subject_names = get_subject_names()

    data_mean_std = dict()
    
    for trial in raw_walking_data['Gaitcycle']['AB01'].keys():
        if trial == 'subjectdetails':
            continue
        data_mean_std[trial] = dict()
        for subject in subject_names:
            data_left = raw_walking_data['Gaitcycle'][subject][trial]['kinetics']['jointmoment']['left']['ankle']['x'][:] / 1000 # N-mm to N-m
            data_right = raw_walking_data['Gaitcycle'][subject][trial]['kinetics']['jointmoment']['right']['ankle']['x'][:] / 1000 # N-mm to N-m

            # Delete additional rows with zeros
            all_zeros_left = []
            for i in range(np.shape(data_left)[0]):
                if np.count_nonzero(data_left[i,:] == 0) > 50:
                    all_zeros_left.append(i)
            data_left = np.delete(data_left, all_zeros_left, 0) # remove rows

            all_zeros_right = []
            for i in range(np.shape(data_right)[0]):
                if np.count_nonzero(data_right[i,:] == 0) > 50:
                    all_zeros_right.append(i)
            data_right = np.delete(data_right, all_zeros_right, 0) # remove rows


            if subject == 'AB01':
                data = data_left
                data = np.vstack((data, data_right))
            else:
                data = np.vstack((data, data_left))
                data = np.vstack((data, data_right))
            
            #if subject == 'AB05' and trial == 's1x2d7x5':
            #    print(all_zeros_left)
            #    print(all_zeros_right)
            #    plt.figure()
            #    plt.plot(range(150), data_left.T, 'k-')
            #    plt.show()
            #    print(" ")
            

        data_mean_std[trial]['mean'] = np.mean(data, axis = 0)
        data_mean_std[trial]['std'] = np.std(data, axis = 0)

        #plt.figure(trial)
        #plt.plot(range(150), data.T, 'k-', alpha = 0.2)
        #plt.plot(range(150), data_mean_std[trial]['mean'])
        #plt.plot(range(150), data_mean_std[trial]['mean'] + 3 * data_mean_std[trial]['std'])
        #plt.plot(range(150), data_mean_std[trial]['mean'] - 3 * data_mean_std[trial]['std'])
        #plt.show()
    
    with open('Gait_data_statistics/ankleMoment_mean_std.pickle', 'wb') as file:
    	pickle.dump(data_mean_std, file)

def tibiaForce_statistics():
    """ Compute mean and standard deviaton of tibia axial force (normalized w.r.t. subject's weight) across the phase
    """    
    subject_names = get_subject_names()

    data_mean_std = dict()
    
    for trial in raw_walking_data['Gaitcycle']['AB01'].keys():
        if trial == 'subjectdetails':
            continue
        data_mean_std[trial] = dict()
        for subject in subject_names:
            data_left = raw_walking_data['Gaitcycle'][subject][trial]['kinetics']['jointforce']['left']['knee']['z'][:]
            data_right = raw_walking_data['Gaitcycle'][subject][trial]['kinetics']['jointforce']['right']['knee']['z'][:]

            # Delete additional rows with zeros
            all_zeros_left = []
            for i in range(np.shape(data_left)[0]):
                if np.count_nonzero(data_left[i,:] == 0) > 50:
                    all_zeros_left.append(i)
            data_left = np.delete(data_left, all_zeros_left, 0) # remove rows

            all_zeros_right = []
            for i in range(np.shape(data_right)[0]):
                if np.count_nonzero(data_right[i,:] == 0) > 50:
                    all_zeros_right.append(i)
            data_right = np.delete(data_right, all_zeros_right, 0) # remove rows


            if subject == 'AB01':
                data = data_left
                data = np.vstack((data, data_right))
            else:
                data = np.vstack((data, data_left))
                data = np.vstack((data, data_right))
            
            #if subject == 'AB05' and trial == 's1x2d7x5':
            #    print(all_zeros_left)
            #    print(all_zeros_right)
            #    plt.figure()
            #    plt.plot(range(150), data_left.T, 'k-')
            #    plt.show()
            #    print(" ")

        data_mean_std[trial]['mean'] = np.mean(data, axis = 0)
        data_mean_std[trial]['std'] = np.std(data, axis = 0)

        plt.figure(trial)
        plt.plot(range(150), data.T, 'k-', alpha = 0.2)
        plt.plot(range(150), data_mean_std[trial]['mean'])
        plt.plot(range(150), data_mean_std[trial]['mean'] + 3 * data_mean_std[trial]['std'])
        plt.plot(range(150), data_mean_std[trial]['mean'] - 3 * data_mean_std[trial]['std'])
        plt.show()
    
    with open('Gait_data_statistics/tibiaForce_mean_std.pickle', 'wb') as file:
    	pickle.dump(data_mean_std, file)

def gait_training_data_generator(mode):

    """
    Drived measurements should use the delete list of the origial data
    """

    if mode == 'globalThighAngles' or mode == 'globalThighVelocities' or mode == 'atan2':
        with open(('Gait_data_statistics/globalThighAngles_mean_std.pickle'), 'rb') as file:
            data_stats = pickle.load(file)
    else:
        with open(('Gait_data_statistics/' + mode + '_mean_std.pickle'), 'rb') as file:
            data_stats = pickle.load(file)
    
    if mode == 'globalThighAngles' or mode == 'globalThighVelocities' or mode == 'atan2':
        with open('Gait_training_data/globalThighAngles_original.pickle', 'rb') as file:
            globalThighAngles = pickle.load(file)
    
    if mode == 'globalThighVelocities':
        with open('Gait_training_data/globalThighVelocities_original.pickle', 'rb') as file:
            globalThighVelocities = pickle.load(file)
    elif mode == 'atan2':
        with open('Gait_training_data/atan2_original.pickle', 'rb') as file:
            atan2 = pickle.load(file)

    subject_names = get_subject_names()

    num_trials = 0
    error_trials = 0
    for trial in raw_walking_data['Gaitcycle']['AB01'].keys():
        if trial == 'subjectdetails':
            continue
        for subject in subject_names:
            # 1) Gait data
            if mode == 'kneeAngles':
                data_left = -raw_walking_data['Gaitcycle'][subject][trial]['kinematics']['jointangles']['left']['knee']['x'][:]
                data_right = -raw_walking_data['Gaitcycle'][subject][trial]['kinematics']['jointangles']['right']['knee']['x'][:]
                # Add offset to knee angles if "over extension" (knee angle > 0) occurs
                if np.max(data_left) > 0:
                    data_left -= np.max(data_left)
                if np.max(data_right) > 0:
                    data_right -= np.max(data_right)
            elif mode == 'ankleAngles':
                data_left = -raw_walking_data['Gaitcycle'][subject][trial]['kinematics']['jointangles']['left']['ankle']['x'][:]
                data_right = -raw_walking_data['Gaitcycle'][subject][trial]['kinematics']['jointangles']['right']['ankle']['x'][:]
            elif mode == 'footAngles':
                data_left = -raw_walking_data['Gaitcycle'][subject][trial]['kinematics']['jointangles']['left']['foot']['x'][:]-90
                data_right = -raw_walking_data['Gaitcycle'][subject][trial]['kinematics']['jointangles']['right']['foot']['x'][:]-90
            elif mode == 'globalThighAngles' or mode == 'globalThighVelocities' or mode == 'atan2':
                data_left = globalThighAngles[trial][subject]['left']
                data_right = globalThighAngles[trial][subject]['right']
            elif mode == 'ankleMoment':
                data_left = raw_walking_data['Gaitcycle'][subject][trial]['kinetics']['jointmoment']['left']['ankle']['x'][:] / 1000 # N-mm to N-m
                data_right = raw_walking_data['Gaitcycle'][subject][trial]['kinetics']['jointmoment']['right']['ankle']['x'][:] / 1000 # N-mm to N-m
            elif mode == 'tibiaForce':
                data_left = raw_walking_data['Gaitcycle'][subject][trial]['kinetics']['jointforce']['left']['knee']['z'][:]
                data_right = raw_walking_data['Gaitcycle'][subject][trial]['kinetics']['jointforce']['right']['knee']['z'][:]

            if mode == 'globalThighVelocities':
                derived_data_left = globalThighVelocities[trial][subject]['left']
                derived_data_right = globalThighVelocities[trial][subject]['right']

            elif mode == 'atan2':
                derived_data_left = atan2[trial][subject]['left']
                derived_data_right = atan2[trial][subject]['right']
            
            # 2) Phase dots
            time_info_left = raw_walking_data['Gaitcycle'][subject][trial]['cycles']['left']['time']
            time_info_right = raw_walking_data['Gaitcycle'][subject][trial]['cycles']['right']['time']
            phase_step = 1/150

            phase_dot_left = []
            for step_left in time_info_left:
                delta_time_left = step_left[1] - step_left[0]
                phase_dot_left.append(np.full((1, 150), phase_step / delta_time_left))
            phase_dot_left = np.squeeze(np.array(phase_dot_left))

            phase_dot_right = []
            for step_right in time_info_right:
                delta_time_right = step_right[1] - step_right[0]
                phase_dot_right.append(np.full((1, 150), phase_step / delta_time_right)) 
            phase_dot_right = np.squeeze(np.array(phase_dot_right))
                        
            # 3) Step lengths
            ptr = raw_walking_data['Gaitcycle'][subject][trial]['description'][1][0]
            walking_speed = raw_walking_data[ptr] # m/s

            ptr = raw_walking_data['Gaitcycle'][subject]['subjectdetails'][1][4]
            leg_length_left = raw_walking_data[ptr] # mm
            ptr = raw_walking_data['Gaitcycle'][subject]['subjectdetails'][1][5]
            leg_length_right = raw_walking_data[ptr] # mm

            step_length_left = []
            for step_left in time_info_left:
                delta_time_left = step_left[149] - step_left[0]
                step_length_left.append(np.full((1, 150), walking_speed * delta_time_left / leg_length_left * 1000)) # normalized step length
            step_length_left = np.squeeze(np.array(step_length_left))

            step_length_right = []
            for step_right in time_info_right:
                delta_time_right = step_right[149] - step_right[0]
                step_length_right.append(np.full((1,150), walking_speed * delta_time_right / leg_length_right * 1000))
            step_length_right = np.squeeze(np.array(step_length_right))

            # 4) Ramp angles
            ptr = raw_walking_data['Gaitcycle'][subject][trial]['description'][1][1]
            incline = raw_walking_data[ptr]
            
            ramp_left = []
            for i in range(np.shape(data_left)[0]):
                ramp_left.append(np.full((1, 150), incline))
            ramp_left = np.squeeze(np.array(ramp_left))
            
            ramp_right = []
            for i in range(np.shape(data_right)[0]):
                ramp_right.append(np.full((1, 150), incline))
            ramp_right = np.squeeze(np.array(ramp_right))

            if len(data_left) != np.shape(phase_dot_left)[0]:
                #print(trial + '/' + subject + '/left')
                error_trials += 1
                continue
            if len(data_right) != np.shape(phase_dot_right)[0]:
                #print(trial + '/' + subject + '/right')
                error_trials += 1
                continue

            # Step 1: Remove outliers ==========================================================================================
            remove_left = []
            for i in range(np.shape(data_left)[0]):
                outlier = False
                for p in range(np.shape(data_left)[1]): # 150
                    if abs(data_left[i, p] - data_stats[trial]['mean'][p]) > 3 * data_stats[trial]['std'][p]:
                        outlier = True
                        break
                if outlier == True:
                    remove_left.append(i)
            
            if mode == 'globalThighVelocities' or mode == 'atan2':
                derived_data_left = np.delete(derived_data_left, remove_left, 0)
            data_left = np.delete(data_left, remove_left, 0) # remove rows
            phase_dot_left = np.delete(phase_dot_left, remove_left, 0)
            step_length_left = np.delete(step_length_left, remove_left, 0)
            ramp_left = np.delete(ramp_left, remove_left, 0)
            
            remove_right = []
            for i in range(np.shape(data_right)[0]):
                outlier = False
                for p in range(np.shape(data_right)[1]): # 150
                    if abs(data_right[i, p] - data_stats[trial]['mean'][p]) > 3 * data_stats[trial]['std'][p]:
                        outlier = True
                        break
                if outlier == True:
                    remove_right.append(i)
            
            if mode == 'globalThighVelocities' or mode == 'atan2':
                derived_data_right = np.delete(derived_data_right, remove_right, 0)
            data_right = np.delete(data_right, remove_right, 0) # remove rows
            phase_dot_right = np.delete(phase_dot_right, remove_right, 0)
            step_length_right = np.delete(step_length_right, remove_right, 0)
            ramp_right = np.delete(ramp_right, remove_right, 0)

            #===================================================================================================================
                
            # Step 2: Remove strides with NaN values
            if mode == 'footAngles':
                nan_val = -90
            else:
                nan_val = 0

            remove_left = []
            for i in range(np.shape(data_left)[0]):
                has_nan = False
                for p in range(3, np.shape(data_left)[1]): # 150
                    if data_left[i, p] == nan_val and data_left[i, p-1] == nan_val and data_left[i, p-2] == nan_val and data_left[i, p-3] == nan_val:
                        has_nan = True
                        break
                if has_nan == True:
                    remove_left.append(i)
            
            if mode == 'globalThighVelocities' or mode == 'atan2':
                derived_data_left = np.delete(derived_data_left, remove_left, 0)
            data_left = np.delete(data_left, remove_left, 0) # remove rows
            phase_dot_left = np.delete(phase_dot_left, remove_left, 0)
            step_length_left = np.delete(step_length_left, remove_left, 0)
            ramp_left = np.delete(ramp_left, remove_left, 0)

            remove_right = []
            for i in range(np.shape(data_right)[0]):
                has_nan = False
                for p in range(3, np.shape(data_right)[1]): # 150
                    if data_right[i, p] == nan_val and data_right[i, p-1] == nan_val and data_right[i, p-2] == nan_val and data_right[i, p-3] == nan_val:
                        has_nan = True
                        break
                if has_nan == True:
                    remove_right.append(i)
            
            if mode == 'globalThighVelocities' or mode == 'atan2':
                derived_data_right = np.delete(derived_data_right, remove_right, 0)
            data_right = np.delete(data_right, remove_right, 0) # remove rows
            phase_dot_right = np.delete(phase_dot_right, remove_right, 0)
            step_length_right = np.delete(step_length_right, remove_right, 0)
            ramp_right = np.delete(ramp_right, remove_right, 0)

            #===================================================================================================================

            # Step 3: Store to training data
            if num_trials == 0:
                if mode == 'globalThighVelocities' or mode == 'atan2':
                    data = derived_data_left
                    data = np.vstack((data, derived_data_right))
                else:
                    data = data_left
                    data = np.vstack((data, data_right))
                
                phase_dot = phase_dot_left
                phase_dot = np.vstack((phase_dot, phase_dot_right))

                step_length = step_length_left
                step_length = np.vstack((step_length, step_length_right))

                ramp = ramp_left
                ramp = np.vstack((ramp, ramp_right))

            else:
                if mode == 'globalThighVelocities' or mode == 'atan2':
                    data = np.vstack((data, derived_data_left))
                    data = np.vstack((data, derived_data_right))
                else:
                    data = np.vstack((data, data_left))
                    data = np.vstack((data, data_right))

                phase_dot = np.vstack((phase_dot, phase_dot_left))
                phase_dot = np.vstack((phase_dot, phase_dot_right))

                step_length = np.vstack((step_length, step_length_left))
                step_length = np.vstack((step_length, step_length_right))

                ramp = np.vstack((ramp, ramp_left))
                ramp = np.vstack((ramp, ramp_right))
            
            num_trials += 1
            #===================================================================================================================
    
    phase = []
    for i in range(np.shape(data)[0]):
        phase.append(np.linspace(0, 1, np.shape(data)[1]).reshape(1, np.shape(data)[1]))
    phase = np.squeeze(np.array(phase))
    
    gait_training_dataset = {'training_data':data, 'phase':phase, 'phase_dot':phase_dot, 'step_length':step_length, 'ramp':ramp}
    
    print("Shape of data: ", np.shape(data))
    print("Shape of phase: ", np.shape(phase))
    print("Shape of phase dot: ", np.shape(phase_dot))
    print("Shape of step length: ", np.shape(step_length))
    print("Shape of ramp: ", np.shape(ramp))
    print("Total # of used trials: ", num_trials)
    print("Total # of trials with errors: ", error_trials)

    with open(('Gait_training_data/' + mode + '_NSL_training_dataset.pickle'), 'wb') as file:
    	pickle.dump(gait_training_dataset, file)
    
if __name__ == '__main__':
    #tibiaForce_statistics()
    #derivedMeasurements_statistics()
    
    #jointAngles_statistics('knee')
    #jointAngles_statistics('ankle')
    #jointAngles_statistics('foot')
    #globalThighAngles_statistics()

    #time.sleep(3)

    #gait_training_data_generator('kneeAngles')
    #gait_training_data_generator('ankleAngles')
    gait_training_data_generator('footAngles')
    #gait_training_data_generator('globalThighAngles')
    #gait_training_data_generator('globalThighVelocities')
    #gait_training_data_generator('atan2')

    #ankleMoment_statistics()
    #gait_training_data_generator('ankleMoment')
    #gait_training_data_generator('tibiaForce')
