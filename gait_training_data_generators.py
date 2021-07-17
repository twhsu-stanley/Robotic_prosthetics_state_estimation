import h5py
import pickle
import numpy as np
from numpy.core.fromnumeric import mean
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


def gait_training_data_generator(mode):
    with open(('Gait_data_statistics/' + mode + '_mean_std.pickle'), 'rb') as file:
    	data_stats = pickle.load(file)
    
    if mode == 'globalThighAngles':
        with open('Gait_training_data/globalThighAngles_original.pickle', 'rb') as file:
            globalThighAngles = pickle.load(file)

    subject_names = get_subject_names()

    for trial in raw_walking_data['Gaitcycle']['AB01'].keys():
        if trial == 'subjectdetails':
            continue
        for subject in subject_names:
            # 1) Joint angles
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
            
            elif mode == 'globalThighAngles':
                data_left = globalThighAngles[trial][subject]['left']
                data_right = globalThighAngles[trial][subject]['right']
            
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
            walking_speed = raw_walking_data[ptr]

            step_length_left = []
            for step_left in time_info_left:
                delta_time_left = step_left[149] - step_left[0]
                step_length_left.append(np.full((1, 150), walking_speed * delta_time_left)) 
            step_length_left = np.squeeze(np.array(step_length_left))

            step_length_right = []
            for step_right in time_info_right:
                delta_time_right = step_right[149] - step_right[0]
                step_length_right.append(np.full((1,150), walking_speed * delta_time_right))
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
            
            data_right = np.delete(data_right, remove_right, 0) # remove rows
            phase_dot_right = np.delete(phase_dot_right, remove_right, 0)
            step_length_right = np.delete(step_length_right, remove_right, 0)
            ramp_right = np.delete(ramp_right, remove_right, 0)

            #===================================================================================================================
                
            # Step 2: Remove strides with NaN values
            remove_left = []
            for i in range(np.shape(data_left)[0]):
                has_nan = False
                for p in range(3, np.shape(data_left)[1]): # 150
                    if data_left[i, p] == 0 and data_left[i, p-1] == 0 and data_left[i, p-2] == 0 and data_left[i, p-3] == 0:
                        has_nan = True
                        break
                if has_nan == True:
                    remove_left.append(i)
            
            data_left = np.delete(data_left, remove_left, 0) # remove rows
            phase_dot_left = np.delete(phase_dot_left, remove_left, 0)
            step_length_left = np.delete(step_length_left, remove_left, 0)
            ramp_left = np.delete(ramp_left, remove_left, 0)

            remove_right = []
            for i in range(np.shape(data_right)[0]):
                has_nan = False
                for p in range(3, np.shape(data_right)[1]): # 150
                    if data_right[i, p] == 0 and data_right[i, p-1] == 0 and data_right[i, p-2] == 0 and data_right[i, p-3] == 0:
                        has_nan = True
                        break
                if has_nan == True:
                    remove_right.append(i)
            
            data_right = np.delete(data_right, remove_right, 0) # remove rows
            phase_dot_right = np.delete(phase_dot_right, remove_right, 0)
            step_length_right = np.delete(step_length_right, remove_right, 0)
            ramp_right = np.delete(ramp_right, remove_right, 0)

            #===================================================================================================================

            # Step 3: Store to training data
            if subject == 'AB01' and trial == 's0x8d10':
                data = data_left
                data = np.vstack((data, data_right))
                
                phase_dot = phase_dot_left
                phase_dot = np.vstack((phase_dot, phase_dot_right))

                step_length = step_length_left
                step_length = np.vstack((step_length, step_length_right))

                ramp = ramp_left
                ramp = np.vstack((ramp, ramp_right))

            else:
                data = np.vstack((data, data_left))
                data = np.vstack((data, data_right))

                phase_dot = np.vstack((phase_dot, phase_dot_left))
                phase_dot = np.vstack((phase_dot, phase_dot_right))

                step_length = np.vstack((step_length, step_length_left))
                step_length = np.vstack((step_length, step_length_right))

                ramp = np.vstack((ramp, ramp_left))
                ramp = np.vstack((ramp, ramp_right))
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

    with open(('Gait_training_data/' + mode + '_training_dataset.pickle'), 'wb') as file:
    	pickle.dump(gait_training_dataset, file)
    
if __name__ == '__main__':

    #jointAngles_statistics('knee')
    #jointAngles_statistics('ankle')
    #globalThighAngles_statistics()

    time.sleep(3)

    #gait_training_data_generator('kneeAngles')
    #gait_training_data_generator('ankleAngles')
    #gait_training_data_generator('globalThighAngles')
    