import h5py
import pickle
import numpy as np
from incline_experiment_utils import *

dataset_location = '../Reznick_Dataset/'
Normalized_data = h5py.File(dataset_location + 'Normalized.mat', 'r')

# Leg lengths here were measured by a measuring tape (m)
leg_length = {'AB01': 0.860, 'AB02': 0.790, 'AB03': 0.770, 'AB04': 0.810, 'AB05':0.770,
              'AB06': 0.842, 'AB07': 0.824, 'AB08': 0.872, 'AB09': 0.830, 'AB10':0.755}

# ramp inclination angles (deg)
ramp_angle = {'i0': 0.00, 'i5': 5.00, 'i10': 10.00, 'in5': -5.00, 'in10': -10.00}

def get_subject_names():
    return Normalized_data['Normalized'].keys()

def get_commanded_velocities(subject, speed_nominal):
    height_avergae = (1.757 + 1.618) / 2; # Average of US male and female heights
    leg_length_avergae = 0.48 * height_avergae;  # Anthropomorphy from Biomechanics and Motor Control, David Winter
    g = 9.81
    
    speed_normalzied = speed_nominal / np.sqrt(g * leg_length_avergae)
    speed_command = speed_normalzied * np.sqrt(g * leg_length[subject])

    return speed_command

def globalThighAngles_R01():
    """
    # Compute level-ground global thigh angle data from the R01 dataset
    """
    globalThighAngles_walking = dict()

    for subject in get_subject_names():
        print("Subject:", subject)
        globalThighAngles_walking[subject] = dict()

        mode = 'Walk'
        globalThighAngles_walking[subject][mode] = dict()
        for speed in ['s0x8', 's1', 's1x2']:
            print(" Walk:", speed)
            globalThighAngles_walking[subject][mode][speed] = dict()
            
            for incline in ['i10', 'i5', 'i0', 'in5', 'in10']:
                print("   Incline:", incline)
                try:
                    jointAngles = Normalized_data['Normalized'][subject][mode][speed][incline]['jointAngles']
                    globalThighAngles_Sagi = np.zeros((np.shape(jointAngles['PelvisAngles'][:])[0], 150))
                    for n in range(np.shape(jointAngles['PelvisAngles'][:])[0]):
                        if subject == 'AB04':
                            globalThighAngles = jointAngles['HipAngles'][:][n] - jointAngles['PelvisAngles'][:][n]
                            globalThighAngles_Sagi[n,:] = globalThighAngles[0,:]
                        else:
                            for i in range(150):
                                R_wp = YXZ_Euler_rotation(-jointAngles['PelvisAngles'][:][n,0,i], -jointAngles['PelvisAngles'][:][n,1,i], jointAngles['PelvisAngles'][:][n,2,i])
                                R_pt = YXZ_Euler_rotation(jointAngles['HipAngles'][:][n,0,i], jointAngles['HipAngles'][:][n,1,i], jointAngles['HipAngles'][:][n,2,i])
                                R_wt = R_wp @ R_pt
                                globalThighAngles_Sagi[n,i], _, _ = YXZ_Euler_angles(R_wt)
                    globalThighAngles_walking[subject][mode][speed][incline] = globalThighAngles_Sagi
                except:
                    print("Exception: something wrong occured!", subject + '/' + mode  + '/' + speed + '/' + incline)
                    continue

    with open('Gait_training_data_R01/globalThighAngles_walking.pickle', 'wb') as file:
        pickle.dump(globalThighAngles_walking, file)

def derivedMeasurements_R01():
    """ 
    # Compute level-ground global thigh angle velocities and atan2 data from the R01 dataset
    """

    with open('Gait_training_data_R01/globalThighAngles_walking.pickle', 'rb') as file:
        globalThighAngles_walking = pickle.load(file)
    
    globalThighVelocities_walking = dict()
    atan2_walking = dict()
    
    for subject in get_subject_names():
        print("Subject:", subject)
        globalThighVelocities_walking[subject] = dict()
        atan2_walking[subject] = dict()

        mode = 'Walk'
        globalThighVelocities_walking[subject][mode] = dict()
        atan2_walking[subject][mode] = dict()
    
        for speed in ['s0x8', 's1', 's1x2']:
            print(" Walk:", speed)
            globalThighVelocities_walking[subject][mode][speed] = dict()
            atan2_walking[subject][mode][speed] = dict()

            for incline in ['i10', 'i5', 'i0', 'in5', 'in10']:
                print("   Incline:", incline)
                try:
                    globalThighAngles = globalThighAngles_walking[subject][mode][speed][incline]
                    stride_period = Normalized_data['Normalized'][subject][mode][speed][incline]['events']['StrideDetails'][2,:]/100

                    globalThighVelocities = np.zeros(np.shape(globalThighAngles))
                    atan2 = np.zeros(np.shape(globalThighAngles))
                    for i in range(np.shape(globalThighAngles)[0]):
                        dt = stride_period[i] / 150
                        # 1. compute golabal thigh velocity with a low-pass filter
                        v = np.diff(globalThighAngles[i, :]) / dt
                        gtv = np.insert(v, 0, 0)
                        gtv_stack = np.array([gtv, gtv, gtv, gtv, gtv]).reshape(-1)
                        gtv_lp_stack = butter_lowpass_filter(gtv_stack, 2, 1/dt, order = 1)
                        globalThighVelocities[i, :] = gtv_lp_stack[2 * len(globalThighAngles[i, :]): 3 * len(globalThighAngles[i, :])]

                        # 2.1. compute atan2 with a band-pass filter
                        gta_stack = np.array([globalThighAngles[i, :], globalThighAngles[i, :], globalThighAngles[i, :],\
                                            globalThighAngles[i, :], globalThighAngles[i, :]]).reshape(-1)
                        """                     
                        gta_bp_stack = butter_bandpass_filter(gta_stack, 0.5, 2, 1/dt, order = 2)
                        gta_bp = gta_bp_stack[2 * len(globalThighAngles[i, :]): 3 * len(globalThighAngles[i, :])]

                        v_bp = np.diff(gta_bp) / dt
                        gtv_bp = np.insert(v_bp, 0, 0)
                        gtv_bp_stack = np.array([gtv_bp, gtv_bp, gtv_bp, gtv_bp, gtv_bp]).reshape(-1)
                        gtv_blp_stack = butter_lowpass_filter(gtv_bp_stack, 2, 1/dt, order = 1)
                        gtv_blp = gtv_blp_stack[2 * len(globalThighAngles[i, :]): 3 * len(globalThighAngles[i, :])]
                        
                        atan2[i, :] = np.arctan2(-gtv_blp/(2*np.pi*0.8), gt_bp)  # arctan2 and scaling
                        for j in range(np.shape(atan2[i, :])[0]):
                            if atan2[i, j] < 0:
                                atan2[i, j] = atan2[i, j] + 2 * np.pi
                        """
                        
                        # 2.2. compute shifted & scaled atan2 w/ a low-pass filter
                        gta_lp_stack = butter_lowpass_filter(gta_stack, 2, 1/dt, order = 1) # 1st, 2nd or 3rd order? 
                        gta_lp = gta_lp_stack[2 * len(globalThighAngles[i, :]): 3 * len(globalThighAngles[i, :])]
                        gtv_lp = np.insert(np.diff(gta_lp) / dt, 0, 0)
                        gta_max = max(gta_lp)
                        gta_min = min(gta_lp)
                        gtv_max = max(gtv_lp)
                        gtv_min = min(gtv_lp)

                        gta_shift = (gta_max + gta_min) / 2
                        gta_scale = abs(gtv_max - gtv_min) / abs(gta_max - gta_min)
                        gtv_shift = (gtv_max + gtv_min) / 2

                        phase_y = - (gtv_lp - gtv_shift)
                        phase_x = gta_scale * (gta_lp - gta_shift)
                        
                        atan2[i, :] = np.arctan2(phase_y, phase_x)
                        for j in range(np.shape(atan2[i, :])[0]):
                            if atan2[i, j] < 0:
                                atan2[i, j] = atan2[i, j] + 2 * np.pi

                    globalThighVelocities_walking[subject][mode][speed][incline] = globalThighVelocities
                    atan2_walking[subject][mode][speed][incline] = atan2

                except:
                    print("Exception: something wrong occured!", subject + '/' + mode  + '/' + speed + '/' + incline)
                    continue

    with open('Gait_training_data_R01/globalThighVelocities_walking.pickle', 'wb') as file:
        pickle.dump(globalThighVelocities_walking, file)

    with open('Gait_training_data_R01/atan2_walking.pickle', 'wb') as file:
        pickle.dump(atan2_walking, file)

def kneeAnkleFootAngles_R01():
    """
    # Compute level-ground global thigh angle data from the R01 dataset
    """
    kneeAngles_walking = dict()
    ankleAngles_walking = dict()
    globalFootAngles_walking = dict()

    for subject in get_subject_names():
        print("Subject:", subject)
        kneeAngles_walking[subject] = dict()
        ankleAngles_walking[subject] = dict()
        globalFootAngles_walking[subject] = dict()
        
        mode = 'Walk'
        kneeAngles_walking[subject][mode] = dict()
        ankleAngles_walking[subject][mode] = dict()
        globalFootAngles_walking[subject][mode] = dict()

        for speed in ['s0x8', 's1', 's1x2']:
            print(" Walk:", speed)
            kneeAngles_walking[subject][mode][speed] = dict()
            ankleAngles_walking[subject][mode][speed] = dict()
            globalFootAngles_walking[subject][mode][speed] = dict()

            for incline in ['i10', 'i5', 'i0', 'in5', 'in10']:
                print("   Incline:", incline)
                try:
                    jointAngles = Normalized_data['Normalized'][subject][mode][speed][incline]['jointAngles']
                    kneeAngles_Sagi = np.zeros((np.shape(jointAngles['KneeAngles'][:])[0], 150))
                    ankleAngles_Sagi = np.zeros((np.shape(jointAngles['AnkleAngles'][:])[0], 150))
                    footAngles_Sagi = np.zeros((np.shape(jointAngles['FootProgressAngles'][:])[0], 150))
                    for n in range(np.shape(jointAngles['PelvisAngles'][:])[0]):
                        #if subject == 'AB04':
                        kneeAngles = -jointAngles['KneeAngles'][:][n]
                        kneeAngles_Sagi[n,:] = kneeAngles[0,:]
                        #
                        ankleAngles = -jointAngles['AnkleAngles'][:][n]
                        ankleAngles_Sagi[n,:] = ankleAngles[0,:]
                        #
                        footAngles = -jointAngles['FootProgressAngles'][:][n]
                        footAngles -= 90
                        footAngles_Sagi[n,:] = footAngles[0,:]
                        
                    kneeAngles_walking[subject][mode][speed][incline] = kneeAngles_Sagi
                    ankleAngles_walking[subject][mode][speed][incline] = ankleAngles_Sagi
                    globalFootAngles_walking[subject][mode][speed][incline] = footAngles_Sagi

                except:
                    print("Exception: something wrong occured!", subject + '/' + mode  + '/' + speed + '/' + incline)
                    continue

    with open('Gait_training_data_R01/kneeAngles_walking.pickle', 'wb') as file:
        pickle.dump(kneeAngles_walking, file)
    #    
    with open('Gait_training_data_R01/ankleAngles_walking.pickle', 'wb') as file:
        pickle.dump(ankleAngles_walking, file)
    #
    with open('Gait_training_data_R01/globalFootAngles_walking.pickle', 'wb') as file:
        pickle.dump(globalFootAngles_walking, file)

def gait_training_data_generator_R01(gait_data):
    with open('Gait_training_data_R01/' + gait_data + '.pickle', 'rb') as file:
        gait_data_dict = pickle.load(file)

    num_trials = 0
    for subject in get_subject_names():
        leg_length_left = Normalized_data[ Normalized_data['Normalized'][subject]['ParticipantDetails'][1,5] ][:][0,0] / 1000
        leg_length_right = Normalized_data[ Normalized_data['Normalized'][subject]['ParticipantDetails'][1,8] ][:][0,0] / 1000
        
        if (gait_data == 'globalThighAngles_walking' or gait_data == 'globalThighVelocities_walking' or gait_data == 'atan2_walking'
            or gait_data == 'kneeAngles_walking' or gait_data == 'ankleAngles_walking' or gait_data == 'globalFootAngles_walking'):
            mode = 'Walk'
            for speed in ['s0x8', 's1', 's1x2']:
                for incline in ['i10', 'i5', 'i0', 'in5', 'in10']:
                    print(subject + '/' + mode  + '/' + speed + '/' + incline)
                    try:
                        # 1) gait data
                        data = gait_data_dict[subject][mode][speed][incline]
                        
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
                                step_length[n,:].fill(walking_speed * stride_period[n] / leg_length_left) # normalization leg_length_left
                            elif side == 2: # right
                                step_length[n,:].fill(walking_speed * stride_period[n] / leg_length_right)
                        
                        # 4) ramp angle 
                        ramp = np.zeros(np.shape(data))
                        for n in range(np.shape(data)[0]):
                            ramp[n,:].fill(ramp_angle[incline])

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

        else:
            raise ValueError("The input gait_data is not supported")
            

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

    with open(('Gait_training_data_R01/' + gait_data + '_training_dataset.pickle'), 'wb') as file:
        pickle.dump(gait_training_dataset, file)
    
if __name__ == '__main__':
    #globalThighAngles_R01()
    #derivedMeasurements_R01()
    kneeAnkleFootAngles_R01()

    #gait_training_data_generator_R01('globalThighAngles_walking')
    #gait_training_data_generator_R01('globalThighVelocities_walking')
    #gait_training_data_generator_R01('atan2_walking')
    #gait_training_data_generator_R01('kneeAngles_walking')
    #gait_training_data_generator_R01('ankleAngles_walking')
    gait_training_data_generator_R01('globalFootAngles_walking')
    