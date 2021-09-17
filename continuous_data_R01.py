import numpy as numpy
import h5py
import pickle
from EKF import load_Psi, wrapTo2pi
from incline_experiment_utils import *
from model_framework import *
from scipy.signal import butter, lfilter, lfilter_zi


dataset_location = '../Reznick_Dataset/'
Streaming_data = h5py.File(dataset_location + 'Streaming.mat', 'r')

def get_subject_names():
    return Streaming_data['Streaming'].keys()

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
    
    m_model = model_loader('Measurement_model_012_NSL.pickle')
    Psi = load_Psi('Generic')

    globalThighAngle_pred = model_prediction(m_model.models[0], Psi['globalThighAngles'], phases, phase_dots, step_lengths, ramps)
    globalThighVelocity_pred = model_prediction(m_model.models[1], Psi['globalThighVelocities'], phases, phase_dots, step_lengths, ramps)
    
    #ankleMoment_pred = model_prediction(m_model.models[4], Psi['ankleMoment'], phases, phase_dots, step_lengths, ramps)
    #tibiaForce_pred = model_prediction(m_model.models[5], Psi['tibiaForce'], phases, phase_dots, step_lengths, ramps)
    
    atan2_pred = model_prediction(m_model.models[2], Psi['atan2'], phases, phase_dots, step_lengths, ramps) + 2*np.pi*phases
    atan2_pred = wrapTo2pi(atan2_pred)
    residuals_atan2 = atan2 - atan2_pred
    residuals_atan2 = np.arctan2(np.sin(residuals_atan2), np.cos(residuals_atan2))
    
    #globalFootAngle_pred = model_prediction(m_model.models[3], Psi['globalFootAngles'], phases, phase_dots, step_lengths, ramps)
    
    L_cop = np.zeros(len(tibiaForce))
    for i in range(len(tibiaForce)):
        if tibiaForce[i] < -3:
            L_cop[i] = -ankleMoment[i] / tibiaForce[i]
    
    print("Cov(globalThighAngle) = ", np.cov(globalThighAngle - globalThighAngle_pred))
    print("Cov(globalThighVelocity) = ", np.cov(globalThighVelocity - globalThighVelocity_pred))
    #print("Cov(ankleMoment) = ", np.cov(ankleMoment - ankleMoment_pred))
    #print("Cov(tibiaForce) = ", np.cov(tibiaForce - tibiaForce_pred))
    print("Cov(atan2) = ", np.cov(residuals_atan2))
    #print("Cov(globalFootAngle) = ", np.cov(globalFootAngle - globalFootAngle_pred))

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
    #plt.plot(tt, ankleMoment_pred[0:total_step], 'b--')
    plt.ylabel('$m_Y~(N \cdot m)$')
    #plt.xlim([0, 13.6])
    plt.xlabel('time (s)')
    
    plt.subplot(715)
    plt.plot(tt, tibiaForce[0:total_step], 'k-')
    #plt.plot(tt, tibiaForce_pred[0:total_step], 'b--')
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
    #plt.plot(tt, globalFootAngle_pred[0:total_step], 'b--')
    plt.ylabel('$\\theta_{f}~(deg)$')
    plt.xlabel('time (s)')

    plt.figure('L COP')
    plt.subplot(411)
    plt.plot(tt, ankleMoment[0:total_step], 'k-')
    #plt.plot(tt, ankleMoment_pred[0:total_step], 'b--')
    plt.ylabel('$m_Y~(N \cdot m)$')
    plt.subplot(412)
    plt.plot(tt, tibiaForce[0:total_step], 'k-')
    #plt.plot(tt, tibiaForce_pred[0:total_step], 'b--')
    plt.ylabel('$f_Z~(N \cdot m)$')
    plt.subplot(413)
    plt.plot(tt, L_cop[0:total_step], 'k-')
    plt.ylabel('L cop (m)')
    plt.subplot(414)
    plt.plot(tt, globalFootAngle[0:total_step], 'k-')
    #plt.plot(tt, globalFootAngle_pred[0:total_step], 'b--')
    plt.ylabel('$\\theta_{f}~(deg)$')
    plt.xlabel('time (s)')

    plt.show()


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


if __name__ == '__main__':
    subject = 'AB08'
    mode = 'Tread'
    speed = 'a0x2'
    
    jointAngles = Streaming_data['Streaming'][subject][mode]['i0']['jointAngles']
    globalThighAngles = jointAngles['LHipAngles'][:][0,:] - jointAngles['LPelvisAngles'][:][0,:]
    
    command_speed = Streaming_data['Streaming'][subject][mode]['i0']['events']['VelProf']['cvel'][:][:,0]

    plt.figure
    plt.subplot(211)
    plt.plot(range(len(command_speed)), command_speed.T)
    plt.xlim((0, 15500))
    plt.ylabel('speed command (m/s)')
    plt.grid()
    plt.subplot(212)
    plt.plot(range(len(globalThighAngles)), globalThighAngles)
    plt.xlim((0, 15500))
    plt.xlabel('time stamps')
    plt.ylabel('global thigh angles (deg)')
    plt.grid()
    plt.show()