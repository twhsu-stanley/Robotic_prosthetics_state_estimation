import numpy as numpy
import h5py
import pickle
from EKF import load_Psi, wrapTo2pi
from incline_experiment_utils import *
from model_framework import *
from training_data_generators_R01 import get_commanded_velocities

dataset_location = '../Reznick_Dataset/'
Streaming_data = h5py.File(dataset_location + 'Streaming.mat', 'r')

# Leg lengths here were measured by a measuring tape.
leg_length = {'AB01': 0.860, 'AB02': 0.790, 'AB03': 0.770, 'AB04': 0.810, 'AB05': 0.770,
              'AB06': 0.842, 'AB07': 0.824, 'AB08': 0.872, 'AB09': 0.830, 'AB10': 0.755}

def get_subject_names():
    return Streaming_data['Streaming'].keys()

def Streaming_globalThighAngles():
    # ONLY CONSIDER THE LEFT SIDE
    mode = 'Tread'
    incline = 'i0'
    streaming_globalThighAngles_tread = dict()
    for subject in get_subject_names():
        print("Subject: ", subject)
        jointAngles = Streaming_data['Streaming'][subject][mode][incline]['jointAngles']
        if subject == 'AB04':
            globalThighAngles_Sagi = jointAngles['LHipAngles'][:][0,:] - jointAngles['LPelvisAngles'][:][0,:]
        else:
            globalThighAngles_Sagi = np.zeros(np.shape(jointAngles['LPelvisAngles'][:][1]))
            for i in range(np.shape(jointAngles['LPelvisAngles'][:])[1]):
                R_wp = YXZ_Euler_rotation(-jointAngles['LPelvisAngles'][:][0,i], -jointAngles['LPelvisAngles'][:][1,i], jointAngles['LPelvisAngles'][:][2,i])
                R_pt = YXZ_Euler_rotation(jointAngles['LHipAngles'][:][0,i], jointAngles['LHipAngles'][:][1,i], jointAngles['LHipAngles'][:][2,i])
                R_wt = R_wp @ R_pt
                globalThighAngles_Sagi[i], _, _ = YXZ_Euler_angles(R_wt)
        streaming_globalThighAngles_tread[subject] = globalThighAngles_Sagi

    
    with open('Streaming_data_R01/streaming_globalThighAngles_tread.pickle', 'wb') as file:
    	pickle.dump(streaming_globalThighAngles_tread, file)

def Streaming_derivedMeasurements():
    # ONLY CONSIDER THE LEFT SIDE
    # global thigh velocities and atan2
    with open('Streaming_data_R01/streaming_globalThighAngles_tread.pickle', 'rb') as file:
    	streaming_globalThighAngles_tread =  pickle.load(file)
    
    # mode = 'Tread'
    # incline = 'i0'
    dt = 1/100

    streaming_globalThighVelocities_tread = dict()
    streaming_atan2_tread = dict()
    for subject in get_subject_names():
        print("Subject: ", subject)
        # compute golabal thigh velocity with a low-pass filter
        gta_Y = streaming_globalThighAngles_tread[subject]
        v = np.diff(gta_Y) / dt
        globalThighVelocities_Sagi = butter_lowpass_filter(np.insert(v, 0, 0), 2, 1/dt, order = 1)
        
        # compute atan2 with a band-pass filter
        gt_bp = butter_bandpass_filter(gta_Y, 0.5, 2, 1/dt, order = 2)
        v_bp = np.diff(gt_bp) / dt
        gtv_bp = butter_lowpass_filter(np.insert(v_bp, 0, 0), 2, 1/dt, order = 1)
        atan2 = np.arctan2(-gtv_bp/(2*np.pi*0.8), gt_bp) # scaled
        
        # compute shifted & scaled atan2 w/ a low-pass filter


        for i in range(np.shape(atan2)[0]):
            if atan2[i] < 0:
                atan2[i] = atan2[i] + 2 * np.pi
        
        streaming_globalThighVelocities_tread[subject] = globalThighVelocities_Sagi
        streaming_atan2_tread[subject] = atan2
    
    with open('Streaming_data_R01/streaming_globalThighVelocities_tread.pickle', 'wb') as file:
    	pickle.dump(streaming_globalThighVelocities_tread, file)
    with open('Streaming_data_R01/streaming_atan2_tread.pickle', 'wb') as file:
    	pickle.dump(streaming_atan2_tread, file)

def load_Streaming_data(subject, speed):
    mode = 'Tread'
    incline = 'i0'
    # ONLY CONSIDER THE LEFT SIDE
    # Load measurements
    with open('Streaming_data_R01/streaming_globalThighAngles_tread.pickle', 'rb') as file:
    	streaming_globalThighAngles_tread =  pickle.load(file)
    with open('Streaming_data_R01/streaming_globalThighVelocities_tread.pickle', 'rb') as file:
    	streaming_globalThighVelocities_tread =  pickle.load(file)
    with open('Streaming_data_R01/streaming_atan2_tread.pickle', 'rb') as file:
    	streaming_atan2_tread =  pickle.load(file)

    # Load state variables 
    heel_strike_index = Streaming_data['Streaming'][subject][mode]['i0']['events']['LHS'][:]
    leg_length_left = Streaming_data[ Streaming_data['Streaming'][subject]['ParticipantDetails'][1,5] ][:][0,0] / 1000
    cvel = Streaming_data['Streaming'][subject][mode]['i0']['events']['VelProf']['cvel'][:][:,0]
    if speed == 'all' or speed == 'a0x2' or speed == 'a0x5':
        walking_speed = get_commanded_velocities(subject, cvel)
    elif speed == 's0x8':
        walking_speed = get_commanded_velocities(subject, 0.8)
    elif speed == 's1':
        walking_speed = get_commanded_velocities(subject, 1)
    elif speed == 's1x2':
        walking_speed = get_commanded_velocities(subject, 1.2)

    dt = 1/100
    phase = np.zeros(len(streaming_globalThighAngles_tread[subject]))
    phase_dot = np.zeros(len(streaming_globalThighAngles_tread[subject]))
    step_length = np.zeros(len(streaming_globalThighAngles_tread[subject]))
    ramp =  np.zeros(len(streaming_globalThighAngles_tread[subject])) # level-ground
    for i in range(np.size(heel_strike_index)):
        if i != np.size(heel_strike_index) - 1:
            p = int(heel_strike_index[i+1] - heel_strike_index[i])
            for k in range(p):
                phase[int(heel_strike_index[i]) + k] = k * 1/p
                phase_dot[int(heel_strike_index[i]) + k] = 1/p / dt
                if speed == 'all' or speed == 'a0x2' or speed == 'a0x5':
                    try:
                        if heel_strike_index[i+1] - heel_strike_index[i] > 200:
                            step_length[int(heel_strike_index[i]) + k] = 0
                        else:
                            step_length[int(heel_strike_index[i]) + k] = ((walking_speed[int(heel_strike_index[i])] + walking_speed[int(heel_strike_index[i+1])]) / 2 
                                                                        * p * dt / leg_length_left)
                    except:
                        continue
                else: 
                    step_length[int(heel_strike_index[i]) + k] = walking_speed * p * dt / leg_length_left 

    # Extract a particular section
    cutPoints = Streaming_data['Streaming'][subject][mode][incline]['events']['cutPoints'][:]
    if speed == 'all':
        start_idx = min(int(cutPoints[0,0]), int(cutPoints[0,1]), int(cutPoints[0,2]))
        end_idx = max(int(cutPoints[1,0]), int(cutPoints[1,1]), int(cutPoints[1,2]))
    elif speed == 's0x8':
        start_idx = int(cutPoints[0,0])
        end_idx = int(cutPoints[1,0])
    elif speed == 's1':
        start_idx = int(cutPoints[0,1])
        end_idx = int(cutPoints[1,1])
    elif speed == 's1x2':
        start_idx = int(cutPoints[0,2])
        end_idx = int(cutPoints[1,2])
    elif speed == 'a0x2':
        start_idx = int(cutPoints[0,3])
        end_idx = int(cutPoints[1,5])
    elif speed == 'a0x5':
        start_idx = int(cutPoints[0,4])
        end_idx = int(cutPoints[1,6])

    phase = phase[start_idx:end_idx]
    phase_dot = phase_dot[start_idx:end_idx]
    step_length = step_length[start_idx:end_idx]
    ramp = ramp[start_idx:end_idx]

    globalThighAngles = streaming_globalThighAngles_tread[subject][start_idx:end_idx]
    globalThighVelocities = streaming_globalThighVelocities_tread[subject][start_idx:end_idx]
    atan2 = streaming_atan2_tread[subject][start_idx:end_idx]

    jointAngles = Streaming_data['Streaming'][subject][mode]['i0']['jointAngles']
    kneeAngles = -jointAngles['LKneeAngles'][:][0, start_idx:end_idx]
    ankleAngles = -jointAngles['LAnkleAngles'][:][0, start_idx:end_idx]
    
    return (phase, phase_dot, step_length, ramp, globalThighAngles, globalThighVelocities, atan2, kneeAngles, ankleAngles)

def plot_Streaming_data(subject, speed):
    print("subject: ",  subject, "| tread-i0 | speed: ", speed)

    (phases, phase_dots, step_lengths, ramps, globalThighAngle, globalThighVelocity, atan2, kneeAngles, ankleAngles) \
                                    = load_Streaming_data(subject, speed)
    atan2_ss = Streaming_atan2_scale_shift(subject, speed, plot = False)

    m_model = model_loader('Measurement_model_012_NSL.pickle')
    Psi = load_Psi('Generic')

    globalThighAngle_pred = model_prediction(m_model.models[0], Psi['globalThighAngles'], phases, phase_dots, step_lengths, ramps)
    globalThighVelocity_pred = model_prediction(m_model.models[1], Psi['globalThighVelocities'], phases, phase_dots, step_lengths, ramps)
    
    ### HPF vs. derv + LPF #####
    globalThighAngle_hp = butter_highpass_filter(globalThighAngle, 2, 100, order = 1) * 2*np.pi * 2 # 1st order
    ############################

    atan2_pred = model_prediction(m_model.models[2], Psi['atan2'], phases, phase_dots, step_lengths, ramps) + 2*np.pi*phases
    atan2_pred = wrapTo2pi(atan2_pred)
    residuals_atan2 = atan2_ss - atan2_pred
    residuals_atan2 = np.arctan2(np.sin(residuals_atan2), np.cos(residuals_atan2))
    
    print("Cov(globalThighAngle) = ", np.cov(globalThighAngle - globalThighAngle_pred))
    print("Cov(globalThighVelocity) = ", np.cov(globalThighVelocity - globalThighVelocity_pred))
    print("Cov(atan2) = ", np.cov(residuals_atan2))

    c_model = model_loader('Control_model_NSL_B20.pickle')
    with open('Psi/Psi_kneeAngles_NSL_B20_const.pickle', 'rb') as file:#_withoutNan
        Psi_knee = pickle.load(file)
    with open('Psi/Psi_ankleAngles_NSL_B20_const.pickle', 'rb') as file:
        Psi_ankle = pickle.load(file)

    kneeAngles_pred = model_prediction(c_model.models[0], Psi_knee, phases, phase_dots, step_lengths, ramps)
    ankleAngles_pred = model_prediction(c_model.models[1], Psi_ankle, phases, phase_dots, step_lengths, ramps)

    
    plt.figure('State')
    plt.subplot(411)
    plt.plot(phases)
    plt.ylabel('Phase')
    plt.grid()
    plt.subplot(412)
    plt.plot(phase_dots)
    plt.ylim([0,1.5])
    plt.ylabel('Phase dot')
    plt.grid()
    plt.subplot(413)
    plt.plot(step_lengths)
    plt.ylim([0,2])
    plt.ylabel('Normalized step length')
    plt.grid()
    plt.subplot(414)
    plt.plot(ramps)
    plt.ylabel('Ramp')
    plt.grid()

    """
    plt.figure('atan2')
    plt.subplot(211)
    plt.plot(atan2[0:1600])
    plt.plot(atan2_pred[0:1600], '--')
    plt.legend(['atan2', 'atan2_predicted'])
    plt.subplot(212)
    a1 = atan2[0:1600] - 2*np.pi*phases[0:1600]
    for i in range(len(a1)):
        a1[i] = np.arctan2(np.sin(a1[i]), np.cos(a1[i]))
    plt.plot(a1)
    a2 = atan2_pred[0:1600] - 2*np.pi*phases[0:1600]
    for i in range(len(a2)):
        a2[i] = np.arctan2(np.sin(a2[i]), np.cos(a2[i]))
    plt.plot(a2)
    plt.legend(['atan2-phase*2pi', 'least-squares fitting', 'new'])
    """
    
    #heel_strike_index = Conti_heel_strikes(subject, trial, side) - Conti_heel_strikes(subject, trial, side)[0]
    total_step =  int(np.shape(globalThighAngle)[0] / 1)
    tt = 0.01 * np.arange(total_step)
    plt.figure('Measurements')
    plt.subplot(411)
    plt.plot(phases[0:total_step])
    plt.ylabel('Phase')
    plt.grid()
    plt.subplot(412)
    plt.plot(tt, globalThighAngle[0:total_step], 'k-')
    plt.plot(tt, globalThighAngle_pred[0:total_step],'b--')
    plt.legend(('actual', 'least squares'))
    plt.ylabel('$\\theta_{th}~(deg)$')
    plt.grid()
    plt.subplot(413)
    plt.plot(tt, globalThighVelocity[0:total_step],'k-')
    plt.plot(tt, globalThighVelocity_pred[0:total_step], 'b--')
    plt.plot(tt, globalThighAngle_hp[0:total_step], 'g--')
    plt.ylabel('$\dot{\\theta}_{Y_{2Hz}} ~(deg/s)$')
    plt.grid()
    plt.subplot(414)
    #plt.plot(tt, atan2[0:total_step],'k-')
    plt.plot(tt, atan2_ss[0:total_step],'k-')
    plt.plot(tt, atan2_pred[0:total_step], 'b--')
    plt.ylabel('$atan2~(rad)$')
    plt.xlabel('time (s)')
    plt.ylim([0, 7.5])
    plt.xlabel('time (s)')
    plt.grid()

    plt.figure()
    plt.plot(tt, phases[0:total_step] * 2 * np.pi, 'k-', linewidth = 2)
    #plt.plot(tt, atan2[0:total_step],'b-', label = 'original')
    plt.plot(tt, atan2_pred[0:total_step], 'g--', label = 'original-predicted')
    plt.plot(tt, atan2_ss[0:total_step],'m-', label = 'shifted-scaled')
    plt.ylabel('Atan$')
    plt.xlabel('time (s)')
    plt.grid()

    plt.figure('Joints')
    plt.subplot(211)
    plt.plot(tt, kneeAngles[0:total_step], 'k-')
    plt.plot(tt, kneeAngles_pred[0:total_step],'b--')
    plt.legend(('actual', 'least squares'))
    plt.ylabel('$\\theta_{knee}~(deg)$')
    plt.grid()
    plt.subplot(212)
    plt.plot(tt, ankleAngles[0:total_step], 'k-')
    plt.plot(tt, ankleAngles_pred[0:total_step],'b--')
    plt.legend(('actual', 'least squares'))
    plt.ylabel('$\\theta_{ankle}~(deg)$')
    plt.grid()

    plt.show()

def Streaming_atan2_scale_shift(subject, speed, plot = True):
    with open('Streaming_data_R01/streaming_globalThighAngles_tread.pickle', 'rb') as file:
    	streaming_globalThighAngles_tread = pickle.load(file)
    
    # Extract a particular section
    cutPoints = Streaming_data['Streaming'][subject]['Tread']['i0']['events']['cutPoints'][:]
    if speed == 'all':
        start_idx = min(int(cutPoints[0,0]), int(cutPoints[0,1]), int(cutPoints[0,2]))
        end_idx = max(int(cutPoints[1,0]), int(cutPoints[1,1]), int(cutPoints[1,2]))
    elif speed == 's0x8':
        start_idx = int(cutPoints[0,0])
        end_idx = int(cutPoints[1,0])
    elif speed == 's1':
        start_idx = int(cutPoints[0,1])
        end_idx = int(cutPoints[1,1])
    elif speed == 's1x2':
        start_idx = int(cutPoints[0,2])
        end_idx = int(cutPoints[1,2])
    elif speed == 'a0x2':
        start_idx = int(cutPoints[0,3])
        end_idx = int(cutPoints[1,5])
    elif speed == 'a0x5':
        start_idx = int(cutPoints[0,4])
        end_idx = int(cutPoints[1,6])

    dt = 1/100
    globalThighAngle = streaming_globalThighAngles_tread[subject][start_idx:end_idx]
    globalThighAngle_lp = butter_lowpass_filter(globalThighAngle, 2, 1/dt, order = 1) # 1st/2nd/3rd order

    #plt.figure()
    #plt.plot(globalThighAngle, label = 'org')
    #plt.plot(butter_lowpass_filter(globalThighAngle, 2, 1/dt, order = 1), label = '1st')
    #plt.plot(globalThighAngle_lp, label = '2nd')
    #plt.legend()
    #plt.show()
    
    # globalThighVelocity = butter_lowpass_filter(np.insert(np.diff(globalThighAngle) / dt, 0, 0), 2, 1/dt, order = 1)
    globalThighVelocity_lp = np.insert(np.diff(globalThighAngle_lp) / dt, 0, 0)
    #plt.figure()
    #plt.plot(butter_lowpass_filter(np.insert(np.diff(globalThighAngle) / dt, 0, 0), 2, 1/dt, order = 1), label = '1st')
    #plt.plot(butter_lowpass_filter(np.insert(np.diff(globalThighAngle) / dt, 0, 0), 2, 1/dt, order = 2), label = '2nd')
    #plt.legend()
    #plt.show()

    globalThighAngle_max = np.zeros(np.shape(globalThighAngle)[0])
    globalThighAngle_min = np.zeros(np.shape(globalThighAngle)[0])
    globalThighVelocity_max = 50 * np.ones(np.shape(globalThighVelocity_lp)[0])
    globalThighVelocity_min = -50 * np.ones(np.shape(globalThighVelocity_lp)[0])
    atan2 = np.ones(np.shape(globalThighVelocity_lp)[0])
    phase_y = np.ones(np.shape(globalThighVelocity_lp)[0])
    phase_x = np.ones(np.shape(globalThighVelocity_lp)[0])
    
    t_min_prev = np.ones(np.shape(globalThighVelocity_lp)[0])
    t_min = np.ones(np.shape(globalThighVelocity_lp)[0])
    t_max_prev = np.ones(np.shape(globalThighVelocity_lp)[0])
    t_max = np.ones(np.shape(globalThighVelocity_lp)[0])

    idx_min_prev = 0
    idx_min = 0
    idx_max_prev = 0
    idx_max = 0

    for i in range(np.shape(globalThighAngle)[0]):
        if i > 0:
            globalThighAngle_max[i] = np.max(globalThighAngle_lp[idx_min_prev:i])
            globalThighAngle_min[i] = np.min(globalThighAngle_lp[idx_max_prev:i])
            globalThighVelocity_max[i] = np.max(globalThighVelocity_lp[idx_min_prev:i])
            globalThighVelocity_min[i] = np.min(globalThighVelocity_lp[idx_max_prev:i])
        
        if i > idx_max + 1:
            idx_min_temp = np.argmin(globalThighAngle_lp[idx_max:i]) + idx_max
            if i > idx_min_temp + 1 and globalThighAngle_lp[idx_min_temp] < -5:
                idx_min = idx_min_temp
                idx_max_temp = np.argmax(globalThighAngle_lp[idx_min_temp:i]) + idx_min_temp
                if i > idx_max_temp + 1 and globalThighAngle_lp[idx_max_temp] > 10: # new stride
                    idx_max_prev = idx_max
                    idx_max = idx_max_temp
                    idx_min_prev = idx_min
        
        t_min_prev[i] = idx_min_prev
        t_min[i] = idx_min
        t_max_prev[i] = idx_max_prev
        t_max[i] = idx_max

        if idx_max_prev > 0 and idx_min_prev > 0:
            globalThighAngle_shift = (globalThighAngle_max[i] + globalThighAngle_min[i]) / 2
            globalThighAngle_scale = abs(globalThighVelocity_max[i] - globalThighVelocity_min[i]) / abs(globalThighAngle_max[i] - globalThighAngle_min[i])
            globalThighVelocity_shift = (globalThighVelocity_max[i] + globalThighVelocity_min[i]) / 2

            phase_y[i] = - (globalThighVelocity_lp[i] - globalThighVelocity_shift)
            phase_x[i] = globalThighAngle_scale * (globalThighAngle_lp[i] - globalThighAngle_shift)
            #phase_y[i] = - globalThighVelocity_lp[i]
            #phase_x[i] = globalThighAngle_scale * globalThighAngle_lp[i]
        else:
            phase_y[i] = - globalThighVelocity_lp[i] / (2*np.pi*0.8)
            phase_x[i] = globalThighAngle_lp[i] 
        
        atan2[i] = np.arctan2(phase_y[i], phase_x[i])
        if atan2[i] < 0:
            atan2[i] = atan2[i] + 2 * np.pi
    
    if plot == False:
        return atan2

    else:
        plt.figure()
        plt.subplot(311)
        plt.plot(globalThighAngle_lp, 'k-', linewidth = 2)
        plt.plot(globalThighAngle_max)
        plt.plot(globalThighAngle_min)
        plt.plot(t_min, np.zeros(np.shape(t_min)[0]), 'bx', label = 'min')
        plt.plot(t_min_prev, np.zeros(np.shape(t_min_prev)[0]), 'g.', label = 'min_prev')
        plt.plot(t_max, np.zeros(np.shape(t_max)[0]), 'rx', label = 'max')
        plt.plot(t_max_prev, np.zeros(np.shape(t_max_prev)[0]), 'm.', label = 'max_prev')
        plt.grid()
        plt.subplot(312)
        plt.plot(globalThighVelocity_lp, 'k-', linewidth = 2)
        plt.plot(globalThighVelocity_max, 'r-')
        plt.plot(globalThighVelocity_min, 'b-')
        #plt.plot(globalThighVelocity_max2, 'r--')
        #plt.plot(globalThighVelocity_min2, 'b--')
        plt.grid()
        plt.subplot(313)
        plt.plot(atan2)
        plt.grid()

        plt.figure("Atan2 phase plane")
        plt.plot(phase_x, phase_y, 'k-', linewidth = 2)
        theta = np.linspace(0, 2*np.pi, 100)
        #plt.plot(50 * np.cos(theta) , 50 * np.sin(theta), 'r--')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid()
        plt.show()

"""
def load_Conti_joints_angles(subject, trial, side):
    pass

def plot_Conti_joints_angles(subject, trial, side):
    phases, phase_dots, step_lengths, ramps = Conti_state_vars(subject, trial, side)
    knee_angle, ankle_angle = load_Conti_joints_angles(subject, trial, side)
    
    c_model = model_loader('Control_model_NSL_B10.pickle')

    with open('Psi_incExp/Psi_kneeAngles_NSL_B10.pickle', 'rb') as file:
        Psi_knee = pickle.load(file)
    with open('Psi_incExp/Psi_ankleAngles_NSL_B10.pickle', 'rb') as file:
        Psi_ankle = pickle.load(file)
    
    knee_angle_pred = model_prediction(c_model.models[0], Psi_knee, phases, phase_dots, step_lengths, ramps)
    ankle_angle_pred = model_prediction(c_model.models[1], Psi_ankle, phases, phase_dots, step_lengths, ramps)
    
    plt.figure("Joint Angle Control")
    start = 0
    end = 2500
    plt.subplot(211)
    plt.plot(knee_angle[start:end], 'k-')
    plt.plot(knee_angle_pred[start:end], 'b--')
    plt.ylabel('knee angle')
    plt.legend(('actual', 'pred'))
    plt.subplot(212)
    plt.plot(ankle_angle[start:end], 'k-')
    plt.plot(ankle_angle_pred[start:end], 'b--') 
    plt.ylabel('ankle angle')
    plt.show()

def detect_knee_over_extention():
    c_model = model_loader('Control_model.pickle')
    with open('Psi/Psi_kneeAngles.pickle', 'rb') as file:
        Psi_knee = pickle.load(file)
    #with open('Psi/Psi_ankleAngles.pickle', 'rb') as file:
    #    Psi_ankle = pickle.load(file)
    n = 0
    for subject in Conti_subject_names():
        for trial in Conti_trial_names(subject):
            if trial == 'subjectdetails':
                continue
            for side in ['left', 'right']:
                knee_angle, ankle_angle = load_Conti_joints_angles(subject, trial, side)
                phases, phase_dots, step_lengths, ramps = Conti_state_vars(subject, trial, side)
                knee_angle_pred = model_prediction(c_model.models[0], Psi_knee, phases, phase_dots, step_lengths, ramps)
                #ankle_angle_pred = model_prediction(c_model.models[1], Psi_ankle, phases, phase_dots, step_lengths, ramps)
                if np.count_nonzero(knee_angle_pred >= 0) > 0:
                    n += 1
                    print(subject +' / '+ trial +' / '+ side)
                    #print(np.count_nonzero(knee_angle_pred >= 0))
                    #print(np.max(knee_angle))
                    #plt.plot(knee_angle_pred)
                    #plt.plot(knee_angle)
                    #plt.legend(('pred', 'actual'))
                    #plt.show()
    print(n)
"""

if __name__ == '__main__':

    #with open('Streaming_data_R01/streaming_globalThighAngles_tread.pickle', 'rb') as file:
    #	streaming_globalThighAngles_tread =  pickle.load(file)
    #plt.figure
    #plt.plot(streaming_globalThighAngles_tread['AB01'])
    #plt.show()

    #Streaming_globalThighAngles()
    #Streaming_derivedMeasurements()
    

    #with open('Streaming_data_R01/streaming_globalThighAngles_tread.pickle', 'rb') as file:
    #	streaming_globalThighAngles_tread =  pickle.load(file)
    
    subject = 'AB03'
    mode = 'Tread'
    speed = 's0x8'
    #atan2_scale_shift(subject, speed)
    plot_Streaming_data(subject, speed)
    
    """
    jointAngles = Streaming_data['Streaming'][subject][mode]['i0']['jointAngles']
    globalThighAngles = jointAngles['LHipAngles'][:][0,:] - jointAngles['LPelvisAngles'][:][0,:]
    
    #streaming_globalThighAngles_tread['AB04'] = globalThighAngles
    #with open('Streaming_data_R01/streaming_globalThighAngles_tread.pickle', 'wb') as file:
    #	pickle.dump(streaming_globalThighAngles_tread, file)
    
    LHS = Streaming_data['Streaming'][subject][mode]['i0']['events']['LHS'][:]
    command_speed = Streaming_data['Streaming'][subject][mode]['i0']['events']['VelProf']['cvel'][:][:,0]
    #l_speed = Streaming_data['Streaming'][subject][mode]['i0']['events']['VelProf']['lvel'][:][:,0]
    cutPoints = Streaming_data['Streaming'][subject][mode]['i0']['events']['cutPoints'][:]
    startPoints = cutPoints[0,:]
    endPoints = cutPoints[1,:]

    with open('Streaming_data_R01/streaming_globalThighAngles_tread.pickle', 'rb') as file:
        streaming_globalThighAngles_tread =  pickle.load(file)
    globalThighAngles_data = streaming_globalThighAngles_tread[subject]

    with open('Streaming_data_R01/streaming_globalThighVelocities_tread.pickle', 'rb') as file:
        streaming_globalThighVelocities_tread =  pickle.load(file)
    globalThighVelocities_data = streaming_globalThighVelocities_tread[subject]

    with open('Streaming_data_R01/streaming_atan2_tread.pickle', 'rb') as file:
        streaming_atan2_tread =  pickle.load(file)
    atan2_data = streaming_atan2_tread[subject]

    plt.figure()
    plt.subplot(411)
    plt.plot(np.arange(len(command_speed))/100, command_speed.T)
    plt.plot(startPoints/100, np.zeros(len(startPoints)), 'r*')
    plt.plot(endPoints/100, np.zeros(len(endPoints)), 'b*')
    plt.xlim((0, max(len(command_speed)/100, len(globalThighAngles)/100)))
    plt.ylabel('speed command (m/s)')
    plt.grid()
    plt.subplot(412)
    plt.plot(np.arange(len(globalThighAngles))/100, globalThighAngles)
    plt.plot(np.arange(len(globalThighAngles_data))/100, globalThighAngles_data)
    plt.plot(LHS/100, np.zeros(len(LHS)), 'r*')
    plt.xlim((0, max(len(command_speed)/100, len(globalThighAngles_data)/100)))
    plt.xlabel('time stamps')
    plt.ylabel('global thigh angles (deg)')
    plt.grid()
    plt.subplot(413)
    plt.plot(np.arange(len(globalThighVelocities_data))/100, globalThighVelocities_data)
    plt.xlabel('time stamps')
    plt.ylabel('global thigh velocities (deg/s)')
    plt.xlim((0, max(len(command_speed)/100, len(globalThighVelocities_data)/100)))
    plt.grid()
    plt.subplot(414)
    plt.plot(np.arange(len(atan2_data))/100, atan2_data)
    plt.xlabel('time stamps')
    plt.ylabel('atan2')
    plt.xlim((0, max(len(command_speed)/100, len(atan2_data)/100)))
    plt.grid()

    (phase, phase_dot, step_length, ramp, globalThighAngles, globalThighVelocities, atan2, _, _) \
        = load_Streaming_data(subject, 'all')
    plt.figure("phase:" + subject + "/" + speed)
    plt.subplot(411)
    plt.plot(np.arange(len(phase))/100, phase)
    plt.ylabel('phase')
    plt.grid()
    plt.subplot(412)
    plt.plot(np.arange(len(phase_dot))/100, phase_dot)
    plt.ylabel('phase_dot')
    plt.grid()
    plt.subplot(413)
    plt.plot(np.arange(len(step_length))/100, step_length)
    plt.ylabel('step_length')
    plt.grid()
    plt.subplot(414)
    plt.plot(np.arange(len(ramp))/100, ramp)
    plt.xlabel('time (s)')
    plt.ylabel('ramp')
    plt.grid()

    plt.figure("measurement:" + subject + "/" + speed)
    plt.subplot(311)
    plt.plot(np.arange(len(globalThighAngles))/100, globalThighAngles)
    plt.ylabel('global thigh angles (deg)')
    plt.grid()
    plt.subplot(312)
    plt.plot(np.arange(len(globalThighVelocities))/100, globalThighVelocities)
    plt.ylabel('global thigh velocities (deg/s)')
    plt.grid()
    plt.subplot(313)
    plt.plot(np.arange(len(atan2))/100, atan2)
    plt.xlabel('time (s)')
    plt.ylabel('atan2')
    plt.grid()
    plt.show()

    #plt.figure
    #plt.plot(range(len(globalThighAngles)), globalThighAngles_Sagi)
    #plt.plot(range(len(globalThighAngles)), globalThighAngles)
    #plt.xlim((0, 15500))
    #plt.ylabel('speed command (m/s)')
    #plt.grid()
    """