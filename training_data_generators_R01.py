import h5py
import pickle
import numpy as np
from incline_experiment_utils import *

dataset_location = '../Reznick_Dataset/'
Normalized_data = h5py.File(dataset_location + 'Normalized.mat', 'r')
#Streaming_data = h5py.File(dataset_location + 'Streaming.mat', 'r')

def get_subject_names():
    return Normalized_data['Normalized'].keys()

def globalThighAngles_R01data():
    """
    # Compute level-ground global thigh angle data from the R01 dataset
    """
    globalThighAngles_walking = dict()
    globalThighAngles_running = dict()
    incline = 'i0'
    for subject in get_subject_names():
        print("Subject:", subject)
        globalThighAngles_walking[subject] = dict()
        globalThighAngles_running[subject] = dict()
        
        # 1) Walking
        mode = 'Walk'
        globalThighAngles_walking[subject][mode] = dict()
        for speed in ['s0x8', 's1', 's1x2']:
            try:
                print(" Walk:", speed)
                jointAngles = Normalized_data['Normalized'][subject][mode][speed][incline]['jointAngles']
                globalThighAngles_Sagi = np.zeros((np.shape(jointAngles['PelvisAngles'][:])[0], 150))
                for n in range(np.shape(jointAngles['PelvisAngles'][:])[0]):
                    for i in range(150):
                        R_wp = YXZ_Euler_rotation(-jointAngles['PelvisAngles'][:][n,0,i], -jointAngles['PelvisAngles'][:][n,1,i], jointAngles['PelvisAngles'][:][n,2,i])
                        R_pt = YXZ_Euler_rotation(jointAngles['HipAngles'][:][n,0,i], jointAngles['HipAngles'][:][n,1,i], jointAngles['HipAngles'][:][n,2,i])
                        R_wt = R_wp @ R_pt
                        globalThighAngles_Sagi[n,i], _, _ = YXZ_Euler_angles(R_wt)
                globalThighAngles_walking[subject][mode][speed] = globalThighAngles_Sagi
            except:
                continue

        # 2) Running
        mode = 'Run'
        globalThighAngles_running[subject][mode] = dict()
        for speed in ['s1x8', 's2x0', 's2x2', 's2x4']:
            try:
                print(" Run:", speed)
                jointAngles = Normalized_data['Normalized'][subject][mode][speed][incline]['jointAngles']
                globalThighAngles_Sagi = np.zeros((np.shape(jointAngles['PelvisAngles'][:])[0], 150))
                for n in range(np.shape(jointAngles['PelvisAngles'][:])[0]):
                    for i in range(150):
                        R_wp = YXZ_Euler_rotation(-jointAngles['PelvisAngles'][:][n,0,i], -jointAngles['PelvisAngles'][:][n,1,i], jointAngles['PelvisAngles'][:][n,2,i])
                        R_pt = YXZ_Euler_rotation(jointAngles['HipAngles'][:][n,0,i], jointAngles['HipAngles'][:][n,1,i], jointAngles['HipAngles'][:][n,2,i])
                        R_wt = R_wp @ R_pt
                        globalThighAngles_Sagi[n,i], _, _ = YXZ_Euler_angles(R_wt)
                globalThighAngles_running[subject][mode][speed] = globalThighAngles_Sagi
            except:
                continue
        
    with open('Gait_training_R01data/globalThighAngles_walking_R01data.pickle', 'wb') as file:
    	pickle.dump(globalThighAngles_walking, file)

    with open('Gait_training_R01data/globalThighAngles_running_R01data.pickle', 'wb') as file:
    	pickle.dump(globalThighAngles_running, file)

def derivedMeasurements_R01data():
    """ 
    # Compute level-ground global thigh angle velocities and atan2 data  from the R01 dataset
    """

    with open('Gait_training_data/globalThighAngles_walking_R01data.pickle', 'rb') as file:
        globalThighAngles_walking = pickle.load(file)
    
    with open('Gait_training_data/globalThighAngles_running_R01data.pickle', 'rb') as file:
        globalThighAngles_running = pickle.load(file)

    globalThighVelocities_walking = dict()
    globalThighVelocities_running = dict()
    atan2_walking = dict()
    atan2_running = dict()

    incline = 'i0'
    for subject in get_subject_names():
        print("Subject:", subject)
        globalThighVelocities_walking[subject] = dict()
        globalThighVelocities_running[subject] = dict()
        atan2_walking[subject] = dict()
        atan2_running[subject] = dict()

        # 1) Walking
        mode = 'Walk'
        globalThighVelocities_walking[subject][mode] = dict()
        for speed in ['s0x8', 's1', 's1x2']:
            try:
                print(" Walk:", speed)
                globalThighAngles = globalThighAngles_walking[subject][mode][speed]
                time_info = Normalized_data['Normalized'][subject][mode][speed][incline]['events']['StrideDetails']
                dt = time_info[:,2] / 100 # 100Hz

                globalThighVelocities = np.zeros(np.shape(globalThighAngles))
                atan2 = np.zeros(np.shape(globalThighAngles))
                for i in range(np.shape(globalThighAngles)[0]):
                    v = np.diff(globalThighAngles[i, :]) / dt[i]
                    gtv = np.insert(v, 0, 0)
                    gtv_stack = np.array([gtv, gtv, gtv, gtv, gtv]).reshape(-1)
                    gtv_lp_stack = butter_lowpass_filter(gtv_stack, 2, 1/dt[i], order = 1)
                    globalThighVelocities[i, :] = gtv_lp_stack[2 * len(globalThighAngles[i, :]): 3 * len(globalThighAngles[i, :])]

                    # compute atan2 w/ a band-pass filter
                    gt_stack = np.array([globalThighAngles[i, :], globalThighAngles[i, :], globalThighAngles[i, :],\
                                         globalThighAngles[i, :], globalThighAngles[i, :]]).reshape(-1)
                    gt_bp_stack = butter_bandpass_filter(gt_stack, 0.5, 2, 1/dt[i], order = 2)
                    gt_bp = gt_bp_stack[2 * len(globalThighAngles[i, :]): 3 * len(globalThighAngles[i, :])]

                    v_bp = np.diff(gt_bp) / dt[i]
                    gtv_bp = np.insert(v_bp, 0, 0)
                    gtv_bp_stack = np.array([gtv_bp, gtv_bp, gtv_bp, gtv_bp, gtv_bp]).reshape(-1)
                    gtv_blp_stack = butter_lowpass_filter(gtv_bp_stack, 2, 1/dt[i], order = 1)
                    gtv_blp = gtv_blp_stack[2 * len(globalThighAngles[i, :]): 3 * len(globalThighAngles[i, :])]
                    
                    atan2[i, :] = np.arctan2(-gtv_blp/(2*np.pi*0.8), gt_bp)  # arctan2 and scaling
                    for j in range(np.shape(atan2[i, :])[0]):
                        if atan2[i, j] < 0:
                            atan2[i, j] = atan2[i, j] + 2 * np.pi
                    
                    globalThighVelocities_walking[subject][mode][speed] = globalThighVelocities
                    atan2_walking[subject][mode][speed] = atan2
            except:
                continue

        # 2) Running
        mode = 'Run'
        globalThighVelocities_running[subject][mode] = dict()
        for speed in ['s1x8', 's2x0', 's2x2', 's2x4']:
            try:
                print(" Run:", speed)
                globalThighAngles_running[subject][mode][speed]
            except:
                continue
    
    for trial in raw_walking_data['Gaitcycle']['AB01'].keys():
        if trial == 'subjectdetails':
            continue
        print("trial: ", trial)

        globalThighVelocities[trial] = dict()
        atan2[trial] = dict()

        for subject in subject_names:
            print("subject: ", subject)

            globalThighVelocities[trial][subject] = dict()
            atan2[trial][subject] = dict()

            data_left = globalThighAngles[trial][subject]['left']

            time_info_left = raw_walking_data['Gaitcycle'][subject][trial]['cycles']['left']['time']

            dt_left = []
            for step_left in time_info_left:
                delta_time_left = step_left[1] - step_left[0]
                dt_left.append(np.full((1,150), delta_time_left))
            dt_left = np.squeeze(np.array(dt_left))

    
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

            
    with open('Gait_training_R01data/globalThighVelocities_walking_R01data.pickle', 'wb') as file:
    	pickle.dump(globalThighVelocities_walking, file)
    with open('Gait_training_R01data/globalThighVelocities_running_R01data.pickle', 'wb') as file:
    	pickle.dump(globalThighVelocities_running, file)
    with open('Gait_training_R01data/atan2_walking_R01data.pickle', 'wb') as file:
    	pickle.dump(atan2_walking, file)
    with open('Gait_training_R01data/atan2_running_R01data.pickle', 'wb') as file:
    	pickle.dump(atan2_running, file)

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
    globalThighAngles_R01data()

    """
    subject = 'AB02'
    mode = 'Run'
    speed = 's1x8'
    jointAngles = Normalized_data['Normalized'][subject][mode][speed]['i0']['jointAngles']
    
    plt.figure()
    print(np.shape(jointAngles['PelvisAngles'][:]))
    for n in range(np.shape(jointAngles['PelvisAngles'][:])[0]):

        globalThighAngles = jointAngles['HipAngles'][:][n] - jointAngles['PelvisAngles'][:][n]
        
        globalThighAngles_Euler = np.zeros(150)
        for i in range(150):
            R_wp = YXZ_Euler_rotation(-jointAngles['PelvisAngles'][:][n,0,i], -jointAngles['PelvisAngles'][:][n,1,i], jointAngles['PelvisAngles'][:][n,2,i])
            R_pt = YXZ_Euler_rotation(jointAngles['HipAngles'][:][n,0,i], jointAngles['HipAngles'][:][n,1,i], jointAngles['HipAngles'][:][n,2,i])
            R_wt = R_wp @ R_pt
            globalThighAngles_Euler[i], _, _ = YXZ_Euler_angles(R_wt)
        
        
        plt.plot(range(150), globalThighAngles[0,:].T)
        plt.plot(range(150), globalThighAngles_Euler)
        plt.show()
    """