import h5py
import pickle
import numpy as np
from incline_experiment_utils import *

dataset_location = '../Reznick_Dataset/'
Normalized_data = h5py.File(dataset_location + 'Normalized.mat', 'r')

def get_subject_names():
    return Normalized_data['Normalized'].keys()

def get_commanded_velocities(subject, speed_nominal):
    # Leg lengths here were measured by a measuring tape.
    leg_length = {'AB01': 0.860, 'AB02': 0.790, 'AB03': 0.770, 'AB04': 0.810, 'AB05':0.770, 
                  'AB06': 0.842, 'AB07': 0.824, 'AB08': 0.872, 'AB09': 0.830, 'AB10':0.755}

    height_avergae = (1.757 + 1.618) / 2; # Average of US male and female heights
    leg_length_avergae = 0.48 * height_avergae;  # Anthropomorphy from Biomechanics and Motor Control, David Winter
    g = 9.81
    
    speed_normalzied = speed_nominal / np.sqrt(g * leg_length_avergae)
    speed_command = speed_normalzied * np.sqrt(g * leg_length[subject])

    return speed_command

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
                print("Exception: something wrong occured!", subject + '/' + mode  + '/' + speed)
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
                print("Exception: something wrong occured!", subject + '/' + mode  + '/' + speed)
                continue
        
    with open('Gait_training_R01data/globalThighAngles_walking_R01data.pickle', 'wb') as file:
    	pickle.dump(globalThighAngles_walking, file)

    with open('Gait_training_R01data/globalThighAngles_running_R01data.pickle', 'wb') as file:
    	pickle.dump(globalThighAngles_running, file)

def derivedMeasurements_R01data():
    """ 
    # Compute level-ground global thigh angle velocities and atan2 data from the R01 dataset
    """

    with open('Gait_training_R01data/globalThighAngles_walking_R01data.pickle', 'rb') as file:
        globalThighAngles_walking = pickle.load(file)
    
    with open('Gait_training_R01data/globalThighAngles_running_R01data.pickle', 'rb') as file:
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
        atan2_walking[subject][mode] = dict()
        for speed in ['s0x8', 's1', 's1x2']:
            try:
                print(" Walk:", speed)
                globalThighAngles = globalThighAngles_walking[subject][mode][speed]
                stride_period = Normalized_data['Normalized'][subject][mode][speed][incline]['events']['StrideDetails'][2,:]/100

                globalThighVelocities = np.zeros(np.shape(globalThighAngles))
                atan2 = np.zeros(np.shape(globalThighAngles))
                for i in range(np.shape(globalThighAngles)[0]):
                    dt = stride_period[i] / 150
                    v = np.diff(globalThighAngles[i, :]) / dt
                    gtv = np.insert(v, 0, 0)
                    gtv_stack = np.array([gtv, gtv, gtv, gtv, gtv]).reshape(-1)
                    gtv_lp_stack = butter_lowpass_filter(gtv_stack, 2, 1/dt, order = 1)
                    globalThighVelocities[i, :] = gtv_lp_stack[2 * len(globalThighAngles[i, :]): 3 * len(globalThighAngles[i, :])]

                    # compute atan2 w/ a band-pass filter
                    gt_stack = np.array([globalThighAngles[i, :], globalThighAngles[i, :], globalThighAngles[i, :],\
                                         globalThighAngles[i, :], globalThighAngles[i, :]]).reshape(-1)
                    gt_bp_stack = butter_bandpass_filter(gt_stack, 0.5, 2, 1/dt, order = 2)
                    gt_bp = gt_bp_stack[2 * len(globalThighAngles[i, :]): 3 * len(globalThighAngles[i, :])]

                    v_bp = np.diff(gt_bp) / dt
                    gtv_bp = np.insert(v_bp, 0, 0)
                    gtv_bp_stack = np.array([gtv_bp, gtv_bp, gtv_bp, gtv_bp, gtv_bp]).reshape(-1)
                    gtv_blp_stack = butter_lowpass_filter(gtv_bp_stack, 2, 1/dt, order = 1)
                    gtv_blp = gtv_blp_stack[2 * len(globalThighAngles[i, :]): 3 * len(globalThighAngles[i, :])]
                    
                    atan2[i, :] = np.arctan2(-gtv_blp/(2*np.pi*0.8), gt_bp)  # arctan2 and scaling
                    for j in range(np.shape(atan2[i, :])[0]):
                        if atan2[i, j] < 0:
                            atan2[i, j] = atan2[i, j] + 2 * np.pi
                    
                globalThighVelocities_walking[subject][mode][speed] = globalThighVelocities
                atan2_walking[subject][mode][speed] = atan2
            except:
                print("Exception: something wrong occured!", subject + '/' + mode  + '/' + speed)
                continue

        # 2) Running
        mode = 'Run'
        globalThighVelocities_running[subject][mode] = dict()
        atan2_running[subject][mode] = dict()
        for speed in ['s1x8', 's2x0', 's2x2', 's2x4']:
            try:
                print(" Run:", speed)
                globalThighAngles = globalThighAngles_running[subject][mode][speed]
                stride_period = Normalized_data['Normalized'][subject][mode][speed][incline]['events']['StrideDetails'][2,:]/100

                globalThighVelocities = np.zeros(np.shape(globalThighAngles))
                atan2 = np.zeros(np.shape(globalThighAngles))
                for i in range(np.shape(globalThighAngles)[0]):
                    dt = stride_period[i] / 150
                    v = np.diff(globalThighAngles[i, :]) / dt
                    gtv = np.insert(v, 0, 0)
                    gtv_stack = np.array([gtv, gtv, gtv, gtv, gtv]).reshape(-1)
                    gtv_lp_stack = butter_lowpass_filter(gtv_stack, 2, 1/dt, order = 1)
                    globalThighVelocities[i, :] = gtv_lp_stack[2 * len(globalThighAngles[i, :]): 3 * len(globalThighAngles[i, :])]

                    # compute atan2 w/ a band-pass filter
                    gt_stack = np.array([globalThighAngles[i, :], globalThighAngles[i, :], globalThighAngles[i, :],\
                                         globalThighAngles[i, :], globalThighAngles[i, :]]).reshape(-1)
                    gt_bp_stack = butter_bandpass_filter(gt_stack, 0.5, 2, 1/dt, order = 2)
                    gt_bp = gt_bp_stack[2 * len(globalThighAngles[i, :]): 3 * len(globalThighAngles[i, :])]

                    v_bp = np.diff(gt_bp) / dt
                    gtv_bp = np.insert(v_bp, 0, 0)
                    gtv_bp_stack = np.array([gtv_bp, gtv_bp, gtv_bp, gtv_bp, gtv_bp]).reshape(-1)
                    gtv_blp_stack = butter_lowpass_filter(gtv_bp_stack, 2, 1/dt, order = 1)
                    gtv_blp = gtv_blp_stack[2 * len(globalThighAngles[i, :]): 3 * len(globalThighAngles[i, :])]
                    
                    atan2[i, :] = np.arctan2(-gtv_blp/(2*np.pi*0.8), gt_bp)  # arctan2 and scaling
                    for j in range(np.shape(atan2[i, :])[0]):
                        if atan2[i, j] < 0:
                            atan2[i, j] = atan2[i, j] + 2 * np.pi
                    
                globalThighVelocities_running[subject][mode][speed] = globalThighVelocities
                atan2_running[subject][mode][speed] = atan2
            except:
                print("Exception: something wrong occured!", subject + '/' + mode  + '/' + speed)
                continue
              
    with open('Gait_training_R01data/globalThighVelocities_walking_R01data.pickle', 'wb') as file:
    	pickle.dump(globalThighVelocities_walking, file)
    with open('Gait_training_R01data/globalThighVelocities_running_R01data.pickle', 'wb') as file:
    	pickle.dump(globalThighVelocities_running, file)
    with open('Gait_training_R01data/atan2_walking_R01data.pickle', 'wb') as file:
    	pickle.dump(atan2_walking, file)
    with open('Gait_training_R01data/atan2_running_R01data.pickle', 'wb') as file:
    	pickle.dump(atan2_running, file)

def gait_training_R01data_generator(gait_data):
    with open('Gait_training_R01data/' + gait_data + '_R01data.pickle', 'rb') as file:
        gait_data_dict = pickle.load(file)

    incline = 'i0'
    num_trials = 0
    for subject in get_subject_names():
        leg_length_left = Normalized_data[ Normalized_data['Normalized'][subject]['ParticipantDetails'][1,5] ][:][0,0] / 1000
        leg_length_right = Normalized_data[ Normalized_data['Normalized'][subject]['ParticipantDetails'][1,8] ][:][0,0] / 1000
        
        if gait_data == 'globalThighAngles_walking' or gait_data == 'globalThighVelocities_walking' or gait_data == 'atan2_walking':
            mode = 'Walk'
            for speed in ['s0x8', 's1', 's1x2']:
                try:
                    print(subject + '/' + mode  + '/' + speed)
                    # 1) gait data
                    data = gait_data_dict[subject][mode][speed]
                    
                    # 2) phase dot
                    stride_period = Normalized_data['Normalized'][subject][mode][speed][incline]['events']['StrideDetails'][2,:]/100
                    phase_dot = np.zeros(np.shape(data))
                    for n in range(np.shape(data)[0]):
                        phase_dot[n,:].fill(1 / stride_period[n])
                    #if min(stride_period) < 0.4:
                    #    print("Abnormally large phase rate: ", 1/min(stride_period))
                    #    print(subject + '/' + mode  + '/' + speed)
                    
                    # 3) stride length
                    if speed == 's0x8':
                        walking_speed = get_commanded_velocities(subject, 0.8)
                    elif speed == 's1':
                        walking_speed = get_commanded_velocities(subject, 1)
                    elif speed == 's1x2':
                        walking_speed = get_commanded_velocities(subject, 1.2)

                    step_length = np.zeros(np.shape(data))
                    for n in range(np.shape(data)[0]):
                        side = Normalized_data['Normalized'][subject][mode][speed][incline]['events']['StrideDetails'][3,n]
                        if side == 1: # left
                            step_length[n,:].fill(walking_speed * stride_period[n] / leg_length_left) # normalization
                        elif side == 2: # right
                            step_length[n,:].fill(walking_speed * stride_period[n] / leg_length_right)
                    
                    # 4) ramp angle 
                    # 'i0': level-ground
                    ramp = np.zeros(np.shape(data))

                    # Store data
                    if num_trials == 0:
                        data_stack = data
                        phase_dot_stack = phase_dot
                        step_length_stack = step_length
                        ramp_stack = ramp
                    else:
                        data_stack = np.vstack((data_stack, data))
                        phase_dot_stack = np.vstack((phase_dot_stack, phase_dot))
                        step_length_stack = np.vstack((step_length_stack, step_length))
                        ramp_stack = np.vstack((ramp_stack, ramp))

                    num_trials += 1
                
                except:
                    print("Exception: something wrong occured!", subject + '/' + mode  + '/' + speed)
                    continue
                
        elif gait_data == 'globalThighAngles_running' or gait_data == 'globalThighVelocities_running' or gait_data == 'atan2_running':
            mode = 'Run'
            for speed in ['s1x8', 's2x0', 's2x2', 's2x4']:
                if subject == 'AB10':
                    # because all AB10 running data seem to be problematic
                    continue

                try:
                    print(subject + '/' + mode  + '/' + speed)
                    # 1) gait data
                    data = gait_data_dict[subject][mode][speed]
                    
                    # 2) phase dot
                    stride_period = Normalized_data['Normalized'][subject][mode][speed][incline]['events']['StrideDetails'][2,:]/100
                    phase_dot = np.zeros(np.shape(data))
                    for n in range(np.shape(data)[0]):
                        phase_dot[n,:].fill(1 / stride_period[n])
                    #if min(stride_period) < 0.4:
                    #    print("Abnormally large phase rate: ", 1/min(stride_period))
                    #    print("  " + subject + '/' + mode  + '/' + speed)
                    
                    # 3) stride length
                    if speed == 's1x8':
                        walking_speed = get_commanded_velocities(subject, 1.8)
                    elif speed == 's2x0':
                        walking_speed = get_commanded_velocities(subject, 2)
                    elif speed == 's2x2':
                        walking_speed = get_commanded_velocities(subject, 2.2)
                    elif speed == 's2x4':
                        walking_speed = get_commanded_velocities(subject, 2.4)

                    step_length = np.zeros(np.shape(data))
                    for n in range(np.shape(data)[0]):
                        side = Normalized_data['Normalized'][subject][mode][speed][incline]['events']['StrideDetails'][3,n]
                        if side == 1: # left
                            step_length[n,:].fill(walking_speed * stride_period[n] / leg_length_left) # normalization
                        elif side == 2: # right
                            step_length[n,:].fill(walking_speed * stride_period[n] / leg_length_right)
                    
                    # 4) ramp angle 
                    # 'i0': level-ground
                    ramp = np.zeros(np.shape(data))

                    # Store data
                    if num_trials == 0:
                        data_stack = data
                        phase_dot_stack = phase_dot
                        step_length_stack = step_length
                        ramp_stack = ramp
                    else:
                        data_stack = np.vstack((data_stack, data))
                        phase_dot_stack = np.vstack((phase_dot_stack, phase_dot))
                        step_length_stack = np.vstack((step_length_stack, step_length))
                        ramp_stack = np.vstack((ramp_stack, ramp))

                    num_trials += 1
                
                except:
                    print("Exception: something wrong occured!", subject + '/' + mode  + '/' + speed)
                    continue
    
    #===================================================================================================================
    phase_stack = np.zeros(np.shape(data_stack))
    for n in range(np.shape(data_stack)[0]):
        phase_stack[n,:] = np.linspace(0, 1, 150).reshape(1, 150)

    gait_training_dataset = {'training_data':data_stack, 'phase':phase_stack, 'phase_dot':phase_dot_stack, 
                             'step_length':step_length_stack, 'ramp':ramp_stack}
    
    print("Shape of data: ", np.shape(data_stack))
    print("Shape of phase: ", np.shape(phase_stack))
    print("Shape of phase dot: ", np.shape(phase_dot_stack))
    print("Shape of step length: ", np.shape(step_length_stack))
    print("Shape of ramp: ", np.shape(ramp_stack))

    with open(('Gait_training_R01data/' + gait_data + '_NSL_training_dataset.pickle'), 'wb') as file:
    	pickle.dump(gait_training_dataset, file)
    
if __name__ == '__main__':
    #globalThighAngles_R01data()
    #derivedMeasurements_R01data()
    
    #gait_training_R01data_generator('globalThighAngles_walking')
    #gait_training_R01data_generator('globalThighVelocities_walking')
    #gait_training_R01data_generator('atan2_walking')

    #gait_training_R01data_generator('globalThighAngles_running')
    #gait_training_R01data_generator('globalThighVelocities_running')
    #gait_training_R01data_generator('atan2_running')
    
    #print(get_commanded_velocities('AB10', 1))
    
    
    subject = 'AB01'
    mode = 'Walk'
    speed = 'a0x2'
    
    jointAngles = Normalized_data['Normalized'][subject][mode][speed]['i0']['jointAngles']
    plt.figure()
    print(np.shape(jointAngles['PelvisAngles'][:]))
    for n in range(np.shape(jointAngles['PelvisAngles'][:])[0]):

        globalThighAngles = jointAngles['HipAngles'][:][n] - jointAngles['PelvisAngles'][:][n]
        
        #globalThighAngles_Euler = np.zeros(150)
        #for i in range(150):
            #R_wp = YXZ_Euler_rotation(-jointAngles['PelvisAngles'][:][n,0,i], -jointAngles['PelvisAngles'][:][n,1,i], jointAngles['PelvisAngles'][:][n,2,i])
            #R_pt = YXZ_Euler_rotation(jointAngles['HipAngles'][:][n,0,i], jointAngles['HipAngles'][:][n,1,i], jointAngles['HipAngles'][:][n,2,i])
            #R_wt = R_wp @ R_pt
            #globalThighAngles_Euler[i], _, _ = YXZ_Euler_angles(R_wt)
        
        
        plt.plot(range(150), globalThighAngles[0,:].T)
        #plt.plot(range(150), globalThighAngles_Euler)
    plt.show()
    