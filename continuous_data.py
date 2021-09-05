import numpy as numpy
import h5py as hp
import pickle
from EKF import load_Psi, wrapTo2pi
from incline_experiment_utils import *
from model_framework import *
from scipy.signal import butter, lfilter, lfilter_zi

# Generate measurement data from the "Continuous" data structure

raw_walking_data = hp.File("../InclineExperiment.mat", "r")

def Conti_subject_names():
    return raw_walking_data['Continuous'].keys()

def Conti_trial_names(subject):
    return raw_walking_data['Continuous'][subject].keys()

def Conti_heel_strikes(subject, trial, side):
    return raw_walking_data['Gaitcycle'][subject][trial]['cycles'][side]['frame'][:]

def Conti_start_end(subject, trial, side):
    heel_strike_index = Conti_heel_strikes(subject, trial, side)
    start_index = heel_strike_index[0]
    end_index = heel_strike_index[np.size(heel_strike_index)-1]
    return int(start_index), int(end_index)

def Conti_globalThighAngles(subject, trial, side):
    jointangles = raw_walking_data['Continuous'][subject][trial]['kinematics']['jointangles'] #deg
    #plt.figure()
    #plt.plot(jointangles[side]['pelvis'][0,:])
    #plt.plot(jointangles[side]['hip'][0, :])
    #plt.legend(('pelvis', 'hip'))
    #plt.show()
    n_s = np.size(jointangles[side]['pelvis'][0,:]) # number of data poitns

    Y_th = np.zeros((1, n_s))
    #X_th = np.zeros((1, n_s))
    #Z_th = np.zeros((1, n_s))
    for i in range(n_s):
        R_wp = YXZ_Euler_rotation(-jointangles[side]['pelvis'][0, i], jointangles[side]['pelvis'][1, i], -jointangles[side]['pelvis'][2, i])
        R_pt = YXZ_Euler_rotation(jointangles[side]['hip'][0, i], -jointangles[side]['hip'][1, i], -jointangles[side]['hip'][2, i])
        R_wt = R_wp @ R_pt
        Y_th[0, i], _, _ = YXZ_Euler_angles(R_wt) # deg # only get the Y component
    return Y_th # only return the Y component

def Conti_reaction_wrench(subject, trial, side):
    vicon_leftbelt_offset = np.array([-768, 885])*1e-3 #[m]
    vicon_rightbelt_offset = np.array([-255, 885])*1e-3 #[m]
    if side == 'left':
        vicon_offset = vicon_leftbelt_offset
    elif side =='right':
        vicon_offset = vicon_rightbelt_offset

    forceplate = raw_walking_data['Continuous'][subject][trial]['kinetics']['forceplate'] #original unit: force[N]/moment[N*mm]/COP[mm]
    markers = raw_walking_data['Continuous'][subject][trial]['kinematics']['markers'] #original unit: markers[mm]
    
    n_s = np.size(forceplate[side]['force'][0,:]) # number of data poitns
    force_ankle_x = np.zeros((1, n_s))
    force_ankle_y = np.zeros((1, n_s))
    force_ankle_z = np.zeros((1, n_s))
    moment_ankle_x = np.zeros((1, n_s))
    moment_ankle_y = np.zeros((1, n_s))
    moment_ankle_z = np.zeros((1, n_s))
    for i in range(n_s):
        marker_list = dict()
        marker_list['toe'] = markers[side]['toe'][:,i]
        marker_list['heel'] = markers[side]['heel'][:,i]
        marker_list['knee'] = markers[side]['knee'][:,i]
        marker_list['ankle'] = markers[side]['ankle'][:,i]
        force_ankle_x[0, i], force_ankle_y[0, i], force_ankle_z[0, i], moment_ankle_x[0, i], moment_ankle_y[0, i], moment_ankle_z[0, i]\
            = wrench_ankle_conti(forceplate[side]['force'][:,i], forceplate[side]['moment'][:,i], marker_list, vicon_offset)
    
    return force_ankle_x, force_ankle_y, force_ankle_z, moment_ankle_x, moment_ankle_y, moment_ankle_z

def Conti_state_vars(subject, trial, side):
    heel_strike_index = raw_walking_data['Gaitcycle'][subject][trial]['cycles'][side]['frame'][:]
    Conti_time = raw_walking_data['Continuous'][subject][trial]['time'][:]
    dt = Conti_time[0, 1] - Conti_time[0, 0] # 0.01 s/ 100 Hz
    ptr = raw_walking_data['Continuous'][subject][trial]['description'][1][0]
    walking_speed = raw_walking_data[ptr][:][0, 0]
    ptr = raw_walking_data['Continuous'][subject][trial]['description'][1][1]
    incline = raw_walking_data[ptr][:][0, 0]
    
    if side == 'left':
        ptr_sl = raw_walking_data['Gaitcycle'][subject]['subjectdetails'][1][4]
    elif side == 'right':
        ptr_sl = raw_walking_data['Gaitcycle'][subject]['subjectdetails'][1][5]
    leg_length = raw_walking_data[ptr_sl] # mm

    phase = np.zeros((np.size(Conti_time)))
    phase_dot = np.zeros((np.size(Conti_time)))
    step_length = np.zeros((np.size(Conti_time)))
    ramp = incline * np.ones((np.size(Conti_time)))

    for i in range(np.size(heel_strike_index)):
        if i != np.size(heel_strike_index) - 1:
            stride_steps = int(heel_strike_index[i+1] - heel_strike_index[i])
            for k in range(stride_steps):
                phase[int(heel_strike_index[i]) + k] = k * 1/stride_steps
                phase_dot[int(heel_strike_index[i]) + k] = 1/stride_steps / dt
                step_length[int(heel_strike_index[i]) + k] = walking_speed * stride_steps * dt / leg_length * 1000

    # truncate the signal s.t. it starts and ends at heel strikes
    start_index, end_index = Conti_start_end(subject, trial, side)
    phase = phase[start_index:end_index]
    phase_dot = phase_dot[start_index:end_index]
    step_length = step_length[start_index:end_index]
    ramp = ramp[start_index:end_index]
    
    return phase, phase_dot, step_length, ramp

def load_Conti_measurement_data(subject, trial, side):
    with open('Continuous_data/Continuous_measurement_data.pickle', 'rb') as file:
        Continuous_measurement_data = pickle.load(file)

    with open('Continuous_data/globalFootAngle_offset.pickle', 'rb') as file:
        offset_dict = pickle.load(file)

    start_index, end_index = Conti_start_end(subject, trial, side)
    globalThighAngle = Continuous_measurement_data[subject][trial][side]['globalThighAngles'][0, start_index:end_index]
    globalThighVelocity = Continuous_measurement_data[subject][trial][side]['globalThighVelocity'][start_index:end_index]
    atan2 = Continuous_measurement_data[subject][trial][side]['atan2_s'][start_index:end_index]
    globalFootAngle = -raw_walking_data['Continuous'][subject][trial]['kinematics']['jointangles'][side]['foot'][0,start_index:end_index] - 90
    try:
        globalFootAngle -= offset_dict[subject][trial][side]
    except:
        pass
    ankleMoment = raw_walking_data['Continuous'][subject][trial]['kinetics']['jointmoment'][side]['ankle'][0, start_index:end_index] / 1000 # N-mm to N-m
    ankleMoment = butter_lowpass_filter(ankleMoment, 7, 100, order = 1)
    tibiaForce = raw_walking_data['Continuous'][subject][trial]['kinetics']['jointforce'][side]['knee'][2, start_index:end_index]
    
    return globalThighAngle, globalThighVelocity, atan2, globalFootAngle, ankleMoment, tibiaForce

def plot_Conti_measurement_data(subject, trial, side):
    print("subject: ",  subject, "| trial: ",  trial, " | side: ", side)

    phases, phase_dots, step_lengths, ramps = Conti_state_vars(subject, trial, side)
    globalThighAngle, globalThighVelocity, atan2, globalFootAngle, ankleMoment, tibiaForce = load_Conti_measurement_data(subject, trial, side)
    
    m_model = model_loader('Measurement_model_012345_NSL.pickle')
    Psi = load_Psi('Generic')

    globalThighAngle_pred = model_prediction(m_model.models[0], Psi['globalThighAngles'], phases, phase_dots, step_lengths, ramps)
    globalThighVelocity_pred = model_prediction(m_model.models[1], Psi['globalThighVelocities'], phases, phase_dots, step_lengths, ramps)
    
    ankleMoment_pred = model_prediction(m_model.models[4], Psi['ankleMoment'], phases, phase_dots, step_lengths, ramps)
    tibiaForce_pred = model_prediction(m_model.models[5], Psi['tibiaForce'], phases, phase_dots, step_lengths, ramps)
    
    atan2_pred = model_prediction(m_model.models[2], Psi['atan2'], phases, phase_dots, step_lengths, ramps) + 2*np.pi*phases
    atan2_pred = wrapTo2pi(atan2_pred)
    residuals_atan2 = atan2 - atan2_pred
    residuals_atan2 = np.arctan2(np.sin(residuals_atan2), np.cos(residuals_atan2))
    
    globalFootAngle_pred = model_prediction(m_model.models[3], Psi['globalFootAngles'], phases, phase_dots, step_lengths, ramps)
    
    L_cop = np.zeros(len(tibiaForce))
    for i in range(len(tibiaForce)):
        if tibiaForce[i] < -3:
            L_cop[i] = -ankleMoment[i] / tibiaForce[i]
    
    print("Cov(globalThighAngle) = ", np.cov(globalThighAngle - globalThighAngle_pred))
    print("Cov(globalThighVelocity) = ", np.cov(globalThighVelocity - globalThighVelocity_pred))
    print("Cov(ankleMoment) = ", np.cov(ankleMoment - ankleMoment_pred))
    print("Cov(tibiaForce) = ", np.cov(tibiaForce - tibiaForce_pred))
    print("Cov(atan2) = ", np.cov(residuals_atan2))
    print("Cov(globalFootAngle) = ", np.cov(globalFootAngle - globalFootAngle_pred))

    plt.figure('State')
    plt.subplot(411)
    plt.plot(phases)
    plt.ylabel('Phase')
    plt.subplot(412)
    plt.plot(phase_dots)
    plt.ylabel('Phase dot')
    plt.subplot(413)
    plt.plot(step_lengths)
    plt.ylabel('Normalized step length')
    plt.subplot(414)
    plt.plot(ramps)
    plt.ylabel('Ramp')

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
    
    #heel_strike_index = Conti_heel_strikes(subject, trial, side) - Conti_heel_strikes(subject, trial, side)[0]
    total_step =  int(np.shape(globalThighAngle)[0] / 1)
    tt = 0.01 * np.arange(total_step)
    plt.figure('Measurements')
    plt.subplot(711)
    plt.plot(phases[0:total_step])
    plt.ylabel('Phase')
    plt.subplot(712)
    plt.plot(tt, globalThighAngle[0:total_step], 'k-')
    plt.plot(tt, globalThighAngle_pred[0:total_step],'b--')
    #plt.xlim([0, 13.6])
    plt.legend(('actual', 'least squares'))
    #plt.legend(('actual', 'least squares'), bbox_to_anchor=(1, 1.05))
    plt.ylabel('$\\theta_{th}~(deg)$')

    plt.subplot(713)
    plt.plot(tt, globalThighVelocity[0:total_step],'k-')
    plt.plot(tt, globalThighVelocity_pred[0:total_step], 'b--')
    plt.ylabel('$\dot{\\theta}_{Y_{2Hz}} ~(deg/s)$')
    #plt.xlim([0, 13.6])

    plt.subplot(714)
    plt.plot(tt, ankleMoment[0:total_step], 'k-')
    plt.plot(tt, ankleMoment_pred[0:total_step], 'b--')
    plt.ylabel('$m_Y~(N \cdot m)$')
    #plt.xlim([0, 13.6])
    plt.xlabel('time (s)')
    
    plt.subplot(715)
    plt.plot(tt, tibiaForce[0:total_step], 'k-')
    plt.plot(tt, tibiaForce_pred[0:total_step], 'b--')
    plt.ylabel('$f_Z~(N \cdot m)$')
    #plt.xlim([0, 13.6])
    plt.xlabel('time (s)')

    plt.subplot(716)
    plt.plot(tt, atan2[0:total_step],'k-')
    plt.plot(tt, atan2_pred[0:total_step], 'b--')
    plt.ylabel('$atan2~(rad)$')
    plt.xlabel('time (s)')
    plt.ylim([0, 7.5])
    #plt.xlim([0, 13.6])

    plt.subplot(717)
    plt.plot(tt, globalFootAngle[0:total_step], 'k-')
    plt.plot(tt, globalFootAngle_pred[0:total_step], 'b--')
    plt.ylabel('$\\theta_{f}~(deg)$')
    plt.xlabel('time (s)')

    plt.figure('L COP')
    plt.subplot(411)
    plt.plot(tt, ankleMoment[0:total_step], 'k-')
    plt.plot(tt, ankleMoment_pred[0:total_step], 'b--')
    plt.ylabel('$m_Y~(N \cdot m)$')
    plt.subplot(412)
    plt.plot(tt, tibiaForce[0:total_step], 'k-')
    plt.plot(tt, tibiaForce_pred[0:total_step], 'b--')
    plt.ylabel('$f_Z~(N \cdot m)$')
    plt.subplot(413)
    plt.plot(tt, L_cop[0:total_step], 'k-')
    plt.ylabel('L cop (m)')
    plt.subplot(414)
    plt.plot(tt, globalFootAngle[0:total_step], 'k-')
    plt.plot(tt, globalFootAngle_pred[0:total_step], 'b--')
    plt.ylabel('$\\theta_{f}~(deg)$')
    plt.xlabel('time (s)')

    plt.show()

def Conti_maxmin(plot = True):
    #for subject in Conti_subject_names():
    phase_dots_sup = np.zeros((9,1)) # 9 different ramp angles
    phase_dots_inf = np.zeros((9,1))
    phase_dots_mean = np.zeros((9,1))
    step_lengths_sup = np.zeros((9,1))
    step_lengths_inf = np.zeros((9,1))
    step_lengths_mean = np.zeros((9,1))
    ramp_code = ['d10', 'd7x5', 'd5', 'd2x5', 'i0', 'i2x5', 'i5', 'i7x5', 'i10']
    ramp_angles = [-10, -7.5, -5, -2.5, 0, 2.5, 5, 7.5, 10]

    for r in range(9): # LOOP THROUGH ALL ANGLES!!
        phase_dots_max = -1000000
        phase_dots_min = 1000000
        step_lengths_max = -1000000
        step_lengths_min = 1000000
        for subject in Conti_subject_names():
            for trial in raw_walking_data['Continuous'][subject].keys():
                if str(trial)[-3:] == ramp_code[r] or str(trial)[-4:] == ramp_code[r] or str(trial)[-2:] == ramp_code[r]:
                    for side in ['left', 'right']:
                        _, phase_dots, step_lengths, _ = Conti_state_vars(subject, trial, side)

                        phase_dots_mean[r] += np.mean(phase_dots)

                        if np.max(phase_dots) > phase_dots_max:
                            phase_dots_max = np.max(phase_dots)
                        if np.min(phase_dots) < phase_dots_min:
                            phase_dots_min = np.min(phase_dots)

                        step_lengths_mean[r] += np.mean(step_lengths)
                            
                        if np.max(step_lengths) > step_lengths_max:
                            step_lengths_max = np.max(step_lengths)
                        if np.min(step_lengths) < step_lengths_min:
                            step_lengths_min = np.min(step_lengths)
                            
        phase_dots_mean[r] = phase_dots_mean[r]/60
        phase_dots_sup[r] = phase_dots_max
        phase_dots_inf[r] = phase_dots_min
        step_lengths_mean[r] = step_lengths_mean[r]/60
        step_lengths_sup[r] = step_lengths_max
        step_lengths_inf[r] = step_lengths_min
    
    saturation_range =np.array([np.max(phase_dots_sup), np.min(phase_dots_inf), np.max(step_lengths_sup), np.min(step_lengths_inf)])

    #print("phases_max =", phases_max)
    #print("phases_min =", phases_min)
    #print("phase_dots_max =", saturation_range[0])
    #print("phase_dots_min =", saturation_range[1])
    #print("step_lengths_max =", saturation_range[2])
    #print("step_lengths_min =", saturation_range[3])
    #print("ramps_max =", ramps_max)
    #print("ramps_min =", ramps_min)

    if plot:
        plt.figure("Phase_dots Extrema")
        plt.plot(ramp_angles, phase_dots_sup, 'r-')
        plt.plot(ramp_angles, phase_dots_mean, 'g-')
        plt.plot(ramp_angles, phase_dots_inf, 'b-')
        plt.legend(("phase_dots_max", "phase_dots_mean",  "phase_dots_min"))
        plt.xlabel("ramp angles")
        plt.ylabel("Phase_dot")

        plt.figure("Step_lengths Extrema")
        plt.plot(ramp_angles, step_lengths_sup, 'r-')
        plt.plot(ramp_angles, step_lengths_mean, 'g-')
        plt.plot(ramp_angles, step_lengths_inf, 'b-')
        plt.legend(("step_lengths_max", "step_lengths_mean", "step_lengths_min"))
        plt.xlabel("ramp angles")
        plt.ylabel("step_lengths")
        plt.show()
    
    return saturation_range

def detect_nan_in_measurements():
    nan_dict = dict()
    for subject in Conti_subject_names():
        nan_dict[subject] = dict()
        for trial in Conti_trial_names(subject):
            if trial == 'subjectdetails':
                continue
            nan_dict[subject][trial] = dict()
            for side in ['left', 'right']:
                nan_dict[subject][trial][side] = True
                globalThighAngle, _, _, globalFootAngle, ankleMoment, tibiaForce= load_Conti_measurement_data(subject, trial, side)
                for i in range(3, len(globalThighAngle)):
                    if globalThighAngle[i] == 0 and globalThighAngle[i-1] == 0 and globalThighAngle[i-2] == 0 and globalThighAngle[i-3] == 0:
                        nan_dict[subject][trial][side] = False
                        print(subject + "/"+ trial + "/"+ side, ": globalThighAngle")
                        break  
                    if globalFootAngle[i] == -90 and globalFootAngle[i-1] == -90 and globalFootAngle[i-2] == -90:
                        nan_dict[subject][trial][side] = False
                        print(subject + "/"+ trial + "/"+ side, ": globalFootAngle")
                        break
                    if ankleMoment[i] == 0 and ankleMoment[i-1] == 0 and ankleMoment[i-2] == 0 and ankleMoment[i-3] == 0:
                        nan_dict[subject][trial][side] = False
                        print(subject + "/"+ trial + "/"+ side, ": ankleMoment")
                        break
                    if tibiaForce[i] == 0 and tibiaForce[i-1] == 0 and tibiaForce[i-2] == 0 and tibiaForce[i-3] == 0:
                        nan_dict[subject][trial][side] = False
                        print(subject + "/"+ trial + "/"+ side, ": tibiaForce")
                        break
    with open('Continuous_data/Measurements_with_Nan.pickle', 'wb') as file:
    	pickle.dump(nan_dict, file)

def load_Conti_joints_angles(subject, trial, side):
    with open('Continuous_data/Continuous_joint_data.pickle', 'rb') as file:
        Continuous_joint_data = pickle.load(file)

    start_index, end_index = Conti_start_end(subject, trial, side)
    knee_angle = Continuous_joint_data[subject][trial][side]['knee'][start_index:end_index]
    ankle_angle = Continuous_joint_data[subject][trial][side]['ankle'][start_index:end_index]

    return knee_angle, ankle_angle

def plot_Conti_joints_angles(subject, trial, side):
    phases, phase_dots, step_lengths, ramps = Conti_state_vars(subject, trial, side)
    knee_angle, ankle_angle = load_Conti_joints_angles(subject, trial, side)
    
    c_model = model_loader('Control_model.pickle')

    with open('Psi/Psi_kneeAngles_NSL_B3.pickle', 'rb') as file:
        Psi_knee = pickle.load(file)
    with open('Psi/Psi_ankleAngles_NSL_B3.pickle', 'rb') as file:
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

def detect_nan_in_joints():
    nan_dict = dict()
    
    n_a = 0
    n_k = 0
    n_b = 0

    for subject in Conti_subject_names():
        nan_dict[subject] = dict()
        for trial in Conti_trial_names(subject):
            if trial == 'subjectdetails':
                continue
            nan_dict[subject][trial] = dict()
            for side in ['left', 'right']:
                
                flag_k = True
                flag_a = True

                nan_dict[subject][trial][side] = True
                knee_angle, ankle_angle = load_Conti_joints_angles(subject, trial, side)
                for i in range(3, len(knee_angle)):
                    if knee_angle[i] == 0 and knee_angle[i-1] == 0 and knee_angle[i-2] == 0\
                        and knee_angle[i-3] == 0:
                        nan_dict[subject][trial][side] = False
                        n_k += 1
                        flag_k = False 
                        #print("Nan in knee angle: " + subject + "/"+ trial + "/"+ side)
                        break

                for i in range(3, len(ankle_angle)):
                    if ankle_angle[i] == 0 and ankle_angle[i-1] == 0 and ankle_angle[i-2] == 0\
                        and ankle_angle[i-3] == 0:
                        nan_dict[subject][trial][side] = False
                        n_a += 1
                        flag_a = False 
                        #print("Nan in ankle angle: " + subject + "/"+ trial + "/"+ side)
                        break
                
                if flag_k == False and flag_a == False:
                    n_b += 1
                
    print("Numbers of trials with nan in the knee angles: ", n_k)    
    print("Numbers of trials with nan in the ankle angles: ", n_a)
    print("Numbers of trials with nan in both knee and ankle angles: ", n_b)

    with open('Continuous_data/KneeAngles_with_Nan.pickle', 'wb') as file:
    	pickle.dump(nan_dict, file)

def plot_Conti_kinetics_data(subject, trial, side):

    ptr = raw_walking_data['Gaitcycle'][subject]['subjectdetails'][1][3]
    subject_weight = raw_walking_data[ptr] # kg
    kneeForce = raw_walking_data['Continuous'][subject][trial]['kinetics']['jointforce'][side]['knee'][:, :] #* subject_weight
    ankleForce = raw_walking_data['Continuous'][subject][trial]['kinetics']['jointforce'][side]['ankle'][:, :] #* subject_weight
    ankleMoment = raw_walking_data['Continuous'][subject][trial]['kinetics']['jointmoment'][side]['ankle'][:, :]  / 1000 #* subject_weight
    #kneeMoment = raw_walking_data['Continuous'][subject][trial]['kinetics']['jointmoment'][side]['knee'][:, :]  / 1000 #* subject_weight

    start = 1000
    end = 2000
    plt.figure()
    plt.subplot(211)
    #plt.plot(range(np.shape(kneeForce)[1])[start:end], kneeForce[0, start:end])
    #plt.plot(range(np.shape(kneeForce)[1])[start:end], kneeForce[1, start:end])
    plt.plot(range(np.shape(kneeForce)[1])[start:end], kneeForce[2, start:end], 'b-')
    plt.plot(range(np.shape(kneeForce)[1])[start:end], -ankleForce[0, start:end], 'r-')

    plt.legend(('tibia', 'ankle'))
    plt.xlabel('samples')
    plt.ylabel('Force Z (N)')
    
    plt.subplot(212)
    plt.plot(range(np.shape(ankleMoment)[1])[start:end], ankleMoment[0, start:end])
    #plt.plot(range(np.shape(ankleMoment)[1])[start:end], kneeMoment[0, start:end])
    #plt.plot(range(np.shape(ankleMoment)[1])[start:end], ankleMoment[2, start:end])
    plt.legend(('ankleMoment', 'kneeMoment', '1','2'))
    plt.xlabel('samples')
    plt.ylabel('Ankle Moment (N-m)')
    
    #plt.figure()
    #plt.plot(range(np.shape(ankleMoment)[1])[start:end], footAngles[0, start:end])
    #plt.plot(range(np.shape(ankleMoment)[1])[start:end], footAngles[1, start:end])
    #plt.plot(range(np.shape(ankleMoment)[1])[start:end], footAngles[2, start:end])

    plt.show()

def globalFootAngle_offset():
    with open('Continuous_data/Measurements_with_Nan.pickle', 'rb') as file:
        nan_dict = pickle.load(file)
    
    dt = 1/100
    tibiaForce_threshold = -1.2

    #speeds = [0.8, 1, 1.2]
    #ramp_angles = [-10, -7.5, -5, -2.5, 0, 2.5, 5, 7.5, 10]

    #offset = np.zeros((3, 9))
    offset_dict = dict()
    for subject in Conti_subject_names():
        offset_dict[subject] = dict()
        for trial in raw_walking_data['Gaitcycle']['AB01'].keys():
            if trial == 'subjectdetails':
                continue
            offset_dict[subject][trial] = dict()

            """
            ptr = raw_walking_data['Continuous'][subject][trial]['description'][1][0]
            speed = raw_walking_data[ptr][:][0, 0]
            s, = np.where(speeds == speed)
            s = s[0]
            """
            ptr = raw_walking_data['Continuous'][subject][trial]['description'][1][1]
            ramp = raw_walking_data[ptr][:][0, 0]
            """
            r, = np.where(ramp_angles == ramp)
            r = r[0]

            offset_side = 0
            k = 0
            """
            for side in ['left', 'right']:
                if nan_dict[subject][trial][side] == False:
                    print(subject + "/"+ trial + "/"+ side+ ": Trial skipped!")
                    continue
                start_index, end_index = Conti_start_end(subject, trial, side)
                globalFootAngle = -raw_walking_data['Continuous'][subject][trial]['kinematics']['jointangles'][side]\
                                    ['foot'][0,start_index:end_index] - 90
                globalFootAngle_Vel = np.insert(np.diff(globalFootAngle)/dt, 0, 0)
                globalFootAngle_Vel = butter_lowpass_filter(globalFootAngle_Vel, 5, 1/dt, order = 1)
                
                tibiaForce = raw_walking_data['Continuous'][subject][trial]['kinetics']['jointforce'][side]['knee'][2, start_index:end_index]
                    
                directRamp = np.zeros(np.size(globalFootAngle))
                stance = False
                stance_prev = False
                stance_idxs = 0
                stance_idx1 = 0
                stance_idx2 = 0
                for i in range(len(globalFootAngle)):
                    if tibiaForce[i] <= tibiaForce_threshold: 
                        if stance_prev == False:
                            stance_idxs = i
                        stance = True
                        stance_prev = stance
                    else:
                        if stance_prev == True:
                            stance_idx2 = i
                            stance_idx1 = stance_idxs
                        stance = False
                        stance_prev = stance
                        
                    if stance_idx1 < stance_idx2:
                        idx = np.argmin(abs(globalFootAngle_Vel[stance_idx1:stance_idx2]))
                        directRamp[i] = globalFootAngle[stance_idx1 + idx]
                    else:
                        directRamp[i] = 1e-4
                
                offset_dict[subject][trial][side] = np.mean(directRamp[200:]) - ramp
                
                #offset_side += np.mean(directRamp[200:]) - ramp
                #k += 1
                
                """
                plt.figure(subject + "/"+ trial + "/"+ side)
                plt.subplot(211)
                plt.plot(directRamp)
                plt.plot(globalFootAngle)
                plt.grid()
                plt.subplot(212)
                plt.plot(globalFootAngle_Vel)
                plt.grid()
                plt.show()
                """
            """
            if k > 0:
                offset[s, r] = offset_side / k
            else:
                offset[s, r] = numpy.NAN
            """

        #plt.figure(subject)
        #plt.plot(np.linspace(-10, 10, 9), offset.T)
        #plt.grid()
        #plt.show()
        
    with open('Continuous_data/globalFootAngle_offset.pickle', 'wb') as file:
    	pickle.dump(offset_dict, file)


if __name__ == '__main__':
    #detect_nan_in_measurements()
    globalFootAngle_offset()
    """
    with open('Continuous_data/Continuous_measurement_data.pickle', 'rb') as file:
        Continuous_measurement_data = pickle.load(file)
    for subject in Conti_subject_names():
        for trial in raw_walking_data['Continuous'][subject].keys():
            if trial == 'subjectdetails':
                continue
            for side in ['left', 'right']:
                Continuous_measurement_data[subject][trial][side]['globalThighAngles'] = Continuous_measurement_data[subject][trial][side].pop('global_thigh_angle_Y')
                Continuous_measurement_data[subject][trial][side]['globalThighVelocity'] = Continuous_measurement_data[subject][trial][side].pop('global_thigh_angVel_2hz')
    with open('Continuous_data/Continuous_measurement_data.pickle', 'wb') as file:
    	pickle.dump(Continuous_measurement_data, file)
    """

    """
    Continuous_joint_data = dict()
    for subject in Conti_subject_names():
        Continuous_joint_data[subject] = dict()
        for trial in raw_walking_data['Continuous'][subject].keys():
            if trial == 'subjectdetails':
                continue
            Continuous_joint_data[subject][trial] = dict()
            for side in ['left', 'right']:
                jointangles = raw_walking_data['Continuous'][subject][trial]['kinematics']['jointangles'][side]
                Continuous_joint_data[subject][trial][side] = dict()
                Continuous_joint_data[subject][trial][side]['knee'] = -jointangles['knee'][0, :]
                Continuous_joint_data[subject][trial][side]['ankle'] = -jointangles['ankle'][0, :]
    with open('Continuous_joint_data.pickle', 'wb') as file:
    	pickle.dump(Continuous_joint_data, file)
    """
    """
    Continuous_measurement_data = dict()
    for subject in Conti_subject_names():
    #for subject in ['AB01']:
        Continuous_measurement_data[subject] = dict()
        for trial in raw_walking_data['Continuous'][subject].keys():
        #for trial in ['s0x8i0']:
            if trial == 'subjectdetails':
                continue
            Continuous_measurement_data[subject][trial] = dict()
            for side in ['left', 'right']:
                Continuous_measurement_data[subject][trial][side] = dict()
                Continuous_measurement_data[subject][trial][side]['globalThighAngles'] = Conti_globalThighAngles(subject, trial, side)
            
                force_ankle_x, force_ankle_y, force_ankle_z, moment_ankle_x, moment_ankle_y, moment_ankle_z \
                    = Conti_reaction_wrench(subject, trial, side)

                Continuous_measurement_data[subject][trial][side]['force_ankle_x'] = force_ankle_x
                Continuous_measurement_data[subject][trial][side]['force_ankle_y'] = force_ankle_y
                Continuous_measurement_data[subject][trial][side]['force_ankle_z'] = force_ankle_z
                Continuous_measurement_data[subject][trial][side]['moment_ankle_x'] = moment_ankle_x
                Continuous_measurement_data[subject][trial][side]['moment_ankle_y'] = moment_ankle_y
                Continuous_measurement_data[subject][trial][side]['moment_ankle_z'] = moment_ankle_z
    
    with open('Continuous_measurement_data.pickle', 'wb') as file:
    	pickle.dump(Continuous_measurement_data, file)
    """
    
    # APPEND NEW DATA
    """
    with open('Continuous_measurement_data_unscaled.pickle', 'rb') as file:
    	Continuous_measurement_data = pickle.load(file)

    dt = 1/100
    for subject in Conti_subject_names():
        for trial in raw_walking_data['Continuous'][subject].keys():
            if trial == 'subjectdetails':
                continue
            for side in ['left', 'right']:
                # APPEND NEW DATA: Global_thigh_angVel_Y
                gt_Y = Continuous_measurement_data[subject][trial][side]['globalThighAngles'][0, :]
                #v = np.diff(gt_Y) / dt
                #global_thigh_angVel_5hz = butter_lowpass_filter(np.insert(v, 0, 0), 5, 1/dt, order = 1)
                #global_thigh_angVel_2x5hz = butter_lowpass_filter(np.insert(v, 0, 0), 2.5, 1/dt, order = 1)
                #globalThighVelocity = butter_lowpass_filter(np.insert(v, 0, 0), 2, 1/dt, order = 1)

                # compute atan2 w/ band-passed signals
                gt_bp = butter_bandpass_filter(gt_Y, 0.5, 2, 1/dt, order = 2)
                v_bp = np.diff(gt_bp) / dt
                gtv_bp = butter_lowpass_filter(np.insert(v_bp, 0, 0), 2, 1/dt, order = 1)
                atan2 = np.arctan2(-gtv_bp/(2*np.pi*0.8), gt_bp) # scaled
                for i in range(np.shape(atan2)[0]):
                    if atan2[i] < 0:
                        atan2[i] = atan2[i] + 2 * np.pi
                
                #Continuous_measurement_data[subject][trial][side]['global_thigh_angle_bp'] = gt_Y_bp
                #Continuous_measurement_data[subject][trial][side]['global_thigh_angVel_5hz'] = global_thigh_angVel_5hz
                #Continuous_measurement_data[subject][trial][side]['global_thigh_angVel_2x5hz'] = global_thigh_angVel_2x5hz
                #Continuous_measurement_data[subject][trial][side]['globalThighVelocity'] = globalThighVelocity
                Continuous_measurement_data[subject][trial][side]['atan2_s'] = atan2
    
    with open('Continuous_measurement_data.pickle', 'wb') as file:
    	pickle.dump(Continuous_measurement_data, file)
    """
    
    ##### Find the saturation range for all subjects ###############
    """
    phase_dots_max = np.zeros((10,1))
    phase_dots_min = np.zeros((10,1))
    step_lengths_max = np.zeros((10,1))
    step_lengths_min = np.zeros((10,1))
    s = 0
    for subject in Conti_subject_names():
        saturation_range = Conti_maxmin(subject, plot = False)
        #print(saturation_range)
        phase_dots_max[s] = saturation_range[0]
        phase_dots_min[s] = saturation_range[1]
        step_lengths_max[s] = saturation_range[2]
        step_lengths_min[s] = saturation_range[3]
        s += 1
    print("phase_dots_max = ", np.max(phase_dots_max))
    print("phase_dots_min = ", np.min(phase_dots_min))
    print("step_lengths_max = ", np.max(step_lengths_max))
    print("step_lengths_min = ", np.min(step_lengths_min))
    """
    ##################################################################
    
    subject = 'AB10'
    trial = 's1i0'
    side = 'right'
    
    #footAngles = raw_walking_data['Continuous'][subject][trial]['kinematics']['jointangles'][side]['foot'][0,:]
    #plt.figure()
    #plt.plot(-footAngles-90)
    #plt.show()
    #plot_Conti_kinetics_data(subject, trial, side)
    #plot_Conti_joints_angles(subject, trial, side)
    #plot_Conti_measurement_data(subject, trial, side)

    ######## test real0time filters #############
    """
    globalThighAngles, _, _, _, _, _, globalThighVelocity, atan2 = load_Conti_measurement_data(subject, trial, side)
    
    # configure low-pass filter (1-order)
    nyq = 0.5 * 100
    normal_cutoff = 2 / nyq
    b_lp, a_lp = butter(1, normal_cutoff, btype='low', analog=False)
    z_lp_1 = lfilter_zi(b_lp,  a_lp)
    z_lp_2 = lfilter_zi(b_lp,  a_lp)
    # configure band-pass filter (2-order)
    nyq = 0.5 * 100
    normal_lowcut = 0.5 / nyq
    normal_highcut = 2 / nyq
    b_bp, a_bp = butter(2, [normal_lowcut, normal_highcut], btype='band', analog=False)
    z_bp = lfilter_zi(b_bp,  a_bp)

    dt = 1/100
    global_thigh_angle_vel_lp = np.zeros((len(globalThighAngles), 1))
    Atan2 = np.zeros((len(globalThighAngles), 1))
    for i in range(len(globalThighAngles)):
        if i == 0:
            global_thigh_angle_vel_lp[i] = 0 
        else:
            global_thigh_angle_vel = (globalThighAngles[i] - global_thigh_angle_0) / dt
            # low-pass filtering
            vel_lp, z_lp_1 = lfilter(b_lp, a_lp, [global_thigh_angle_vel], zi = z_lp_1)
            global_thigh_angle_vel_lp[i] = vel_lp[0]

        global_thigh_angle_0 = globalThighAngles[i]

        # Compute atan2
        ang_bp, z_bp = lfilter(b_bp, a_bp, [globalThighAngles[i]], zi = z_bp) 
        global_thigh_angle_bp = ang_bp[0]
        if i == 0:
            global_thigh_angle_vel_blp = 0
        else:
            global_thigh_angle_vel_bp = (global_thigh_angle_bp - global_thigh_angle_bp_0) / dt
            # low-pass filtering
            vel_blp, z_lp_2 = lfilter(b_lp, a_lp, [global_thigh_angle_vel_bp], zi = z_lp_2)
            global_thigh_angle_vel_blp = vel_blp[0]

        global_thigh_angle_bp_0 = global_thigh_angle_bp

        Atan2[i] = np.arctan2(-global_thigh_angle_vel_blp / (2*np.pi*0.8), global_thigh_angle_bp)
        if Atan2[i] < 0:
            Atan2[i] = Atan2[i] + 2 * np.pi

    plt.figure()
    plt.plot(global_thigh_angle_vel_lp, 'r-')
    plt.plot(globalThighVelocity, 'k--')
    plt.figure()
    plt.plot(Atan2, 'r-')
    plt.plot(atan2, 'k--')
    plt.show()
    """
    #############################################

    """
    dt = get_time_step(subject)
    with open('Gait_cycle_data/Global_thigh_angle.npz', 'rb') as file:
        gt_Y = np.load(file)
        gt = np.zeros(np.shape(gt_Y[subject][0]))
        gt_bp = np.zeros(np.shape(gt_Y[subject][0]))
        gt_bp2 = np.zeros(np.shape(gt_Y[subject][0]))
        atan2 = np.zeros(np.shape(gt_Y[subject][0]))
        atan22 = np.zeros(np.shape(gt_Y[subject][0]))
        atan2v = np.zeros(np.shape(gt_Y[subject][0]))
        gtv_2hz = np.zeros(np.shape(gt_Y[subject][0]))
        gtv_2hz2 = np.zeros(np.shape(gt_Y[subject][0]))

        plt.figure('Phase portrait')
        for i in range(15):
            gt[i, :] = gt_Y[subject][0][i, :]
            gt_bp[i, :] = butter_bandpass_filter(gt_Y[subject][0][i, :], 0.5, 2, 1/dt[i, 0], order = 1)
            
            gt_rep = np.array([gt_Y[subject][0][i, :], gt_Y[subject][0][i, :], gt_Y[subject][0][i, :],\
                                   gt_Y[subject][0][i, :], gt_Y[subject][0][i, :]]).reshape(-1)
            gbp = butter_bandpass_filter(gt_rep, 0.5, 2, 1/dt[i, 0], order = 2)
            gt_bp2[i, :] = gbp[2*len(gt_Y[subject][0][i, :]): 3*len(gt_Y[subject][0][i, :])]
            
            v_bp = np.diff(gt_bp2[i, :]) / dt[i, 0]
            gtv_bp = np.insert(v_bp, 0, 0)
            gtv_stack = np.array([gtv_bp, gtv_bp, gtv_bp, gtv_bp, gtv_bp]).reshape(-1)
            gtv_blp = butter_lowpass_filter(gtv_stack, 2, 1/dt[i, 0], order = 1)[2*len(gt_Y[subject][0][i, :]): 3*len(gt_Y[subject][0][i, :])]

            atan2[i, :] = np.arctan2(-gtv_blp/(2*np.pi*0.8), gt_bp2[i, :])
            plt.plot(gt_bp2[i, :], -gtv_blp/(2*np.pi*0.8), 'r-')

            for j in range(np.shape(atan2[i, :])[0]):
                if atan2[i, j] < 0:
                    atan2[i, j] = atan2[i, j] + 2 * np.pi
            
            #######
            gbp = butter_bandpass_filter(gt_rep, 0.5, 2, 1/dt[i, 0], order = 1)
            gt_bp2[i, :] = gbp[2*len(gt_Y[subject][0][i, :]): 3*len(gt_Y[subject][0][i, :])]
            
            v_bp = np.diff(gt_bp2[i, :]) / dt[i, 0]
            gtv_bp = np.insert(v_bp, 0, 0)
            gtv_stack = np.array([gtv_bp, gtv_bp, gtv_bp, gtv_bp, gtv_bp]).reshape(-1)
            gtv_blp = butter_lowpass_filter(gtv_stack, 2, 1/dt[i, 0], order = 1)[2*len(gt_Y[subject][0][i, :]): 3*len(gt_Y[subject][0][i, :])]

            atan22[i, :] = np.arctan2(-gtv_blp, gt_bp2[i, :])
            for j in range(np.shape(atan22[i, :])[0]):
                if atan22[i, j] < 0:
                    atan22[i, j] = atan22[i, j] + 2 * np.pi
            
            
            v = np.diff(gt_Y[subject][0][i, :]) / dt[i, 0]
            gtv = np.insert(v, 0, 0)
            gtv_stack = np.array([gtv, gtv, gtv, gtv, gtv]).reshape(-1)
            gtv_2hz[i, :] = butter_lowpass_filter(gtv, 2, 1/dt[i, 0], order = 1)
            gtv2 = butter_lowpass_filter(gtv_stack, 2, 1/dt[i, 0], order = 2)
            gtv_2hz2[i, :] = gtv2[3 * len(gt_Y[subject][0][i, :]): 4 * len(gt_Y[subject][0][i, :])]


    plt.figure()
    plt.plot(gt[0:10,:].ravel(), 'b-')
    plt.plot(gt_bp[0:10,:].ravel(), 'k-')
    plt.plot(gt_bp2[0:10,:].ravel(), 'r-')
    plt.legend(('original thigh angle', 'band-pass filtered', 'new band-pass filtered'))
    
    plt.figure()
    plt.plot(gtv_2hz[0:15,:].ravel(), 'b-')
    plt.plot(gtv_2hz2[0:15,:].ravel(), 'r--')
    
    plt.figure()
    plt.plot(atan2[0:15,:].ravel(), 'b-')
    plt.plot(atan22[0:15,:].ravel(), 'r-')

    #plt.figure()
    #plt.plot(atan2v[0:15,:].ravel(), 'b-')

    plt.show()
    """