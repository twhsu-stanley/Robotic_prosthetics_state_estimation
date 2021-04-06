import numpy as numpy
import h5py as hp
import pickle
import math
from incline_experiment_utils import *
from model_framework import *
from model_fit import *

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

def Conti_global_thigh_angle_Y(subject, trial, side):
    jointangles = raw_walking_data['Continuous'][subject][trial]['kinematics']['jointangles'] #deg
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
                step_length[int(heel_strike_index[i]) + k] = walking_speed * stride_steps * dt

    # truncate the signal s.t. it starts and ends at heel strikes
    start_index, end_index = Conti_start_end(subject, trial, side)
    phase = phase[start_index:end_index]
    phase_dot = phase_dot[start_index:end_index]
    step_length = step_length[start_index:end_index]
    ramp = ramp[start_index:end_index]
    
    return phase, phase_dot, step_length, ramp

def load_Conti_measurement_data(subject, trial, side):
    with open('Continuous_measurement_data_2.pickle', 'rb') as file:
        Continuous_measurement_data = pickle.load(file)

    start_index, end_index = Conti_start_end(subject, trial, side)

    global_thigh_angle_Y = Continuous_measurement_data[subject][trial][side]['global_thigh_angle_Y'][0, start_index:end_index]
    force_z_ankle = Continuous_measurement_data[subject][trial][side]['force_ankle_z'][0, start_index:end_index]
    force_x_ankle = Continuous_measurement_data[subject][trial][side]['force_ankle_x'][0, start_index:end_index]
    moment_y_ankle = Continuous_measurement_data[subject][trial][side]['moment_ankle_y'][0, start_index:end_index]
    global_thigh_angVel_Y1 = Continuous_measurement_data[subject][trial][side]['global_thigh_angVel_Y1'][start_index:end_index]
    global_thigh_angVel_Y2 = Continuous_measurement_data[subject][trial][side]['global_thigh_angVel_Y2'][start_index:end_index]
    global_thigh_angVel_Y3 = Continuous_measurement_data[subject][trial][side]['global_thigh_angVel_Y3'][start_index:end_index]
    #atan2 = Continuous_measurement_data[subject][trial][side]['atan2'][start_index:end_index]

    return global_thigh_angle_Y, force_z_ankle, force_x_ankle, moment_y_ankle,\
           global_thigh_angVel_Y1, global_thigh_angVel_Y2, global_thigh_angVel_Y3 #, atan2

def plot_Conti_data(subject, trial, side):
    phases, phase_dots, step_lengths, ramps = Conti_state_vars(subject, trial, side)
    global_thigh_angle_Y, force_z_ankle, force_x_ankle, moment_y_ankle,\
                                         global_thigh_angVel_Y1, global_thigh_angVel_Y2, global_thigh_angVel_Y3\
                                         = load_Conti_measurement_data(subject, trial, side)
    m_model = model_loader('Measurement_model_2.pickle')
    Psi = load_Psi(subject)

    global_thigh_angle_Y_pred = model_prediction(m_model.models[0], Psi[0], phases, phase_dots, step_lengths, ramps)
    force_z_ankle_pred = model_prediction(m_model.models[1], Psi[1], phases, phase_dots, step_lengths, ramps)
    force_x_ankle_pred = model_prediction(m_model.models[2], Psi[2], phases, phase_dots, step_lengths, ramps)
    moment_y_ankle_pred = model_prediction(m_model.models[3],Psi[3], phases, phase_dots, step_lengths, ramps)
    global_thigh_angVel_Y1_pred = model_prediction(m_model.models[4], Psi[4], phases, phase_dots, step_lengths, ramps)
    global_thigh_angVel_Y2_pred = model_prediction(m_model.models[5], Psi[5], phases, phase_dots, step_lengths, ramps)
    global_thigh_angVel_Y3_pred = model_prediction(m_model.models[6], Psi[6], phases, phase_dots, step_lengths, ramps)
    #atan2_pred = model_prediction(m_model.models[5], Psi[5], phases, phase_dots, step_lengths, ramps)

    #dt = 1/100
    #v = np.diff(global_thigh_angle_Y) / dt
    #global_thigh_angVel_Y1 = butter_lowpass_filter(np.insert(v, 0, 0), 5, 1/dt, order = 1)     
    #global_thigh_angVel_Y2 = butter_lowpass_filter(np.insert(v, 0, 0), 2, 1/dt, order = 1)
    #global_thigh_angVel_Y3 = butter_lowpass_filter(np.insert(v, 0, 0), 1.5, 1/dt, order = 1)

    #plt.figure()
    #plt.subplot(211)
    #plt.plot(global_thigh_angle_Y[0:800])
    #plt.subplot(212)
    #plt.plot(np.insert(v, 0, 0)[0:800])
    #plt.plot(global_thigh_angVel_Y1[0:800])
    #plt.plot(global_thigh_angVel_Y2[0:800])
    #plt.plot(global_thigh_angVel_Y3[0:800])

    #atan2 = np.arctan2(-global_thigh_angVel_Y, gt_Y_filt) # negate y to ensure counter-clockwise phase angles
    #for i in range(len(atan2)):
        #if atan2[i] < 0:
            #atan2[i] = 2 * math.pi + atan2[i]
    
    plt.figure('measurement')
    plt.subplot(711)
    plt.plot(global_thigh_angle_Y, 'b-')
    plt.plot(global_thigh_angle_Y_pred,'k--')
    plt.legend(['actual','predicted'])
    plt.ylabel('global_thigh_angle_Y')
    plt.subplot(712)
    plt.plot(force_z_ankle, 'b-')
    plt.plot(force_z_ankle_pred, 'k--')
    plt.legend(['actual','predicted'])
    plt.ylabel('force_z_ankle')
    plt.subplot(713)
    plt.plot(force_x_ankle, 'b-')
    plt.plot(force_x_ankle_pred, 'k--')
    plt.legend(['actual','predicted'])
    plt.ylabel('force_x_ankle')
    plt.subplot(714)
    plt.plot(moment_y_ankle, 'b-')
    plt.plot(moment_y_ankle_pred, 'k--')
    plt.legend(['actual','predicted'])
    plt.ylabel('moment_y_ankle')
    plt.subplot(715)
    plt.plot(global_thigh_angVel_Y1, 'b-')
    plt.plot(global_thigh_angVel_Y1_pred,'k--')
    plt.legend(['actual','predicted'])
    plt.ylabel('global_thigh_angVel_Y1')
    plt.subplot(716)
    plt.plot(global_thigh_angVel_Y2, 'b-')
    plt.plot(global_thigh_angVel_Y2_pred,'k--')
    plt.legend(['actual','predicted'])
    plt.ylabel('global_thigh_angVel_Y2')
    plt.subplot(717)
    plt.plot(global_thigh_angVel_Y3, 'b-')
    plt.plot(global_thigh_angVel_Y3_pred,'k--')
    plt.legend(['actual','predicted'])
    plt.ylabel('global_thigh_angVel_Y3')
    #plt.subplot(616)
    #plt.plot(atan2, 'b-')
    #plt.plot(atan2_pred,'k--')
    #plt.legend(['actual','predicted'])
    #plt.ylabel('atan2')

    plt.figure('state')
    plt.subplot(411)
    plt.plot(phases)
    plt.ylabel('phase')
    plt.subplot(412)
    plt.plot(phase_dots)
    plt.ylabel('phase dot')
    plt.subplot(413)
    plt.plot(step_lengths)
    plt.ylabel('step length')
    plt.subplot(414)
    plt.plot(ramps)
    plt.ylabel('ramp')
    
    plt.figure()
    plt.plot(global_thigh_angVel_Y1[1200:2000],'k-')
    plt.plot(global_thigh_angVel_Y1_pred[1200:2000], 'k--')
    plt.plot(global_thigh_angVel_Y2[1200:2000],'r-')
    plt.plot(global_thigh_angVel_Y2_pred[1200:2000], 'r--')
    plt.plot(global_thigh_angVel_Y3[1200:2000],'b-')
    plt.plot(global_thigh_angVel_Y3_pred[1200:2000], 'b--')
    plt.ylabel('global_thigh_angVel_Y')
    #plt.subplot(414)
    #plt.plot(atan2[0:600], 'b-')
    #plt.plot(atan2_pred[0:600],'k--')
    #plt.ylabel('atan2')

    #plt.figure("phase portrait")
    #plt.plot(gt_Y_filt, global_thigh_angVel_Y, 'b-')
    #plt.plot(global_thigh_angle_Y_pred, global_thigh_angVel_Y_pred, 'k--')
    #plt.xlabel('global_thigh_angle_Y')
    #plt.ylabel('global_thigh_angVel_Y')
    #plt.legend(['actual','predicted'])

    plt.show()

if __name__ == '__main__':
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
                Continuous_measurement_data[subject][trial][side]['global_thigh_angle_Y'] = Conti_global_thigh_angle_Y(subject, trial, side)
            
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
    """
    # APPEND NEW DATA: Global_thigh_angVel_Y
    with open('Continuous_measurement_data.pickle', 'rb') as file:
    	Continuous_measurement_data = pickle.load(file)

    dt = 1/100
    for subject in Conti_subject_names():
        for trial in raw_walking_data['Continuous'][subject].keys():
            if trial == 'subjectdetails':
                continue
            for side in ['left', 'right']:
                # APPEND NEW DATA: Global_thigh_angVel_Y
                gt_Y = Continuous_measurement_data[subject][trial][side]['global_thigh_angle_Y'][0, :]
                v = np.diff(gt_Y) / dt
                global_thigh_angVel_Y1 = butter_lowpass_filter(np.insert(v, 0, 0), 5, 1/dt, order = 1)
                global_thigh_angVel_Y2 = butter_lowpass_filter(np.insert(v, 0, 0), 2, 1/dt, order = 1)
                global_thigh_angVel_Y3 = butter_lowpass_filter(np.insert(v, 0, 0), 1.5, 1/dt, order = 1)
                
                Continuous_measurement_data[subject][trial][side]['global_thigh_angVel_Y1'] = global_thigh_angVel_Y1
                Continuous_measurement_data[subject][trial][side]['global_thigh_angVel_Y2'] = global_thigh_angVel_Y2
                Continuous_measurement_data[subject][trial][side]['global_thigh_angVel_Y3'] = global_thigh_angVel_Y3
    
    with open('Continuous_measurement_data_2.pickle', 'wb') as file:
    	pickle.dump(Continuous_measurement_data, file)
    """
    
    # Test plot
    subject = 'AB02'
    trial = 's0x8i10'
    side = 'left'
    #print(np.diff(Conti_heel_strikes('AB02', 's0x8d10', 'left').reshape(-1)))
    plot_Conti_data(subject, trial, side)
    #R = measurement_error_cov('AB09')
    #plot_Conti_data('AB01', 's0x8i0', 'left')
