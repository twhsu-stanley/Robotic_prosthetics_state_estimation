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
    with open('Continuous_measurement_data_new_s.pickle', 'rb') as file:
        Continuous_measurement_data = pickle.load(file)

    start_index, end_index = Conti_start_end(subject, trial, side)
    global_thigh_angle_Y = Continuous_measurement_data[subject][trial][side]['global_thigh_angle_Y'][0, start_index:end_index]
    #global_thigh_angle_bp = Continuous_measurement_data[subject][trial][side]['global_thigh_angle_bp'][start_index:end_index] # bp
    force_z_ankle = Continuous_measurement_data[subject][trial][side]['force_ankle_z'][0, start_index:end_index]
    force_x_ankle = Continuous_measurement_data[subject][trial][side]['force_ankle_x'][0, start_index:end_index]
    moment_y_ankle = Continuous_measurement_data[subject][trial][side]['moment_ankle_y'][0, start_index:end_index]
    global_thigh_angVel_5hz = Continuous_measurement_data[subject][trial][side]['global_thigh_angVel_5hz'][start_index:end_index]
    global_thigh_angVel_2x5hz = Continuous_measurement_data[subject][trial][side]['global_thigh_angVel_2x5hz'][start_index:end_index]
    global_thigh_angVel_2hz = Continuous_measurement_data[subject][trial][side]['global_thigh_angVel_2hz'][start_index:end_index]
    atan2 = Continuous_measurement_data[subject][trial][side]['atan2_s'][start_index:end_index]

    return global_thigh_angle_Y, force_z_ankle, force_x_ankle, moment_y_ankle,\
           global_thigh_angVel_5hz, global_thigh_angVel_2x5hz, global_thigh_angVel_2hz, atan2

def plot_Conti_data(subject, trial, side):
    phases, phase_dots, step_lengths, ramps = Conti_state_vars(subject, trial, side)
    global_thigh_angle_Y, force_z_ankle, force_x_ankle, moment_y_ankle,\
                                         global_thigh_angVel_5hz, global_thigh_angVel_2x5hz, global_thigh_angVel_2hz, atan2\
                                         = load_Conti_measurement_data(subject, trial, side)
    m_model = model_loader('Measurement_model_8.pickle')
    Psi = load_Psi(subject)

    global_thigh_angle_Y_pred = model_prediction(m_model.models[0], Psi[0], phases, phase_dots, step_lengths, ramps)
    force_z_ankle_pred = model_prediction(m_model.models[1], Psi[1], phases, phase_dots, step_lengths, ramps)
    force_x_ankle_pred = model_prediction(m_model.models[2], Psi[2], phases, phase_dots, step_lengths, ramps)
    moment_y_ankle_pred = model_prediction(m_model.models[3],Psi[3], phases, phase_dots, step_lengths, ramps)
    global_thigh_angVel_5hz_pred = model_prediction(m_model.models[4], Psi[4], phases, phase_dots, step_lengths, ramps)
    global_thigh_angVel_2x5hz_pred = model_prediction(m_model.models[5], Psi[5], phases, phase_dots, step_lengths, ramps)
    global_thigh_angVel_2hz_pred = model_prediction(m_model.models[6], Psi[6], phases, phase_dots, step_lengths, ramps)
    atan2_pred = model_prediction(m_model.models[7], Psi[7], phases, phase_dots, step_lengths, ramps) + 2*np.pi*phases

 
    # compute rmse
    print("subject: ",  subject)
    print("trial: ",  trial)

    err_gthY = global_thigh_angle_Y - global_thigh_angle_Y_pred
    print("mean g_th_Y ", np.mean(err_gthY))
    print("std g_th_Y ", np.std(err_gthY))
    print("RMSE g_th_Y ", np.sqrt(np.square(err_gthY).mean()))
    print('________________________________')
    err_fz = force_z_ankle - force_z_ankle_pred
    print("mean f_z ", np.mean(err_fz))
    print("std f_z ", np.std(err_fz))
    print("RMSE f_z ", np.sqrt(np.square(err_fz).mean()))
    print('________________________________')
    err_fx = force_x_ankle - force_x_ankle_pred
    print("mean f_x ", np.mean(err_fx))
    print("std f_x ", np.std(err_fx))
    print("RMSE f_x ", np.sqrt(np.square(err_fx).mean()))
    print('________________________________')
    err_my = moment_y_ankle - moment_y_ankle_pred
    print("mean m_y ", np.mean(err_my))
    print("std m_y ", np.std(err_my))
    print("RMSE m_y ", np.sqrt(np.square(err_my).mean()))
    print('________________________________')
    err_gtv_5hz = global_thigh_angVel_5hz - global_thigh_angVel_5hz_pred
    print("mean gtv_5hz ", np.mean(err_gtv_5hz))
    print("std gtv_5hz ", np.std(err_gtv_5hz))
    print("RMSE gtv_5hz ", np.sqrt(np.square(err_gtv_5hz).mean()))
    print('________________________________')
    err_gtv_2x5hz = global_thigh_angVel_2x5hz - global_thigh_angVel_2x5hz_pred
    print("mean gtv_2x5hz ", np.mean(err_gtv_2x5hz))
    print("std gtv_2x5hz ", np.std(err_gtv_2x5hz))
    print("RMSE gtv_2x5hz ", np.sqrt(np.square(err_gtv_2x5hz).mean()))
    print('________________________________')
    err_gtv_2hz = global_thigh_angVel_2hz - global_thigh_angVel_2hz_pred
    print("mean gtv_2hz ", np.mean(err_gtv_2hz))
    print("std gtv_2hz ", np.std(err_gtv_2hz))
    print("RMSE gtv_2hz ", np.sqrt(np.square(err_gtv_2hz).mean()))
    print('________________________________')
    err_atan2 = atan2 - atan2_pred
    err_atan2 = np.arctan2(np.sin(err_atan2), np.cos(err_atan2)) # wrap to pi
    print("mean atan2 ", np.mean(err_atan2))
    print("std atan2 ", np.std(err_atan2))
    print("RMSE atan2 ", np.sqrt(np.square(err_atan2).mean()))

    plt.figure('atan2')
    plt.subplot(211)
    plt.plot(atan2[0:1600])

    at2 = atan2_pred + 0
    for i in range(len(at2)):
        if at2[i] > 2*np.pi:
           at2[i] -= 2*np.pi
    plt.plot(at2[0:1600], '--')
    #plt.plot(phases[0:1600]*2*np.pi, 'r')
    plt.legend(['atan2', 'atan2_predicted'])#, 'phase*2pi'
    
    plt.subplot(212)
    a = atan2[0:1600] - 2*np.pi*phases[0:1600]
    for i in range(len(a)):
        a[i] = np.arctan2(np.sin(a[i]), np.cos(a[i])) 
    plt.plot(a)
    plt.plot(atan2_pred[0:1600] - 2*np.pi*phases[0:1600], '--')
    plt.legend(['atan2-phase*2pi', 'least-squares fitting'])
    

    heel_strike_index = Conti_heel_strikes(subject, trial, side) - Conti_heel_strikes(subject, trial, side)[0]
    total_step =  int(heel_strike_index[10, 0])+1 #np.shape(z)[1]
    tt = 0.01 * np.arange(total_step)
    plt.figure('Original Measurements')
    plt.subplot(411)
    plt.plot(tt, global_thigh_angle_Y[0:total_step], 'k-')
    plt.plot(tt, global_thigh_angle_Y_pred[0:total_step],'b--')
    plt.ylim([-25, 105])
    plt.xlim([0, 13.6])
    plt.legend(('actual', 'least squares'))
    #plt.legend(('actual', 'least squares'), bbox_to_anchor=(1, 1.05))
    plt.ylabel('$\\theta_Y~(deg)$')
    plt.subplot(412)
    plt.plot(tt, force_z_ankle[0:total_step], 'k-')
    plt.plot(tt, force_z_ankle_pred[0:total_step], 'b--')
    plt.ylabel('$f_Z~(N)$')
    plt.xlim([0, 13.6])
    plt.subplot(413)
    plt.plot(tt, force_x_ankle[0:total_step], 'k-')
    plt.plot(tt, force_x_ankle_pred[0:total_step], 'b--')
    plt.ylabel('$f_X~(N)$')
    plt.xlim([0, 13.6])
    plt.subplot(414)
    plt.plot(tt, moment_y_ankle[0:total_step], 'k-')
    plt.plot(tt, moment_y_ankle_pred[0:total_step], 'b--')
    plt.ylabel('$m_Y~(N \cdot m)$')
    plt.xlim([0, 13.6])
    plt.xlabel('time (s)')


    plt.figure('State')
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
    
    
    plt.figure('Fictitious Sensors: thigh angle vel')
    plt.subplot(311)
    plt.plot(tt, global_thigh_angVel_5hz[0:total_step],'k-')
    plt.plot(tt, global_thigh_angVel_5hz_pred[0:total_step], 'b--')
    plt.ylabel('$\dot{\\theta}_{Y_{5Hz}} ~(deg/s)$')
    plt.ylim([-200, 400])
    plt.xlim([0, 13.6])
    plt.legend(('actual', 'least squares'))
    #plt.legend(('actual', 'least squares'), bbox_to_anchor=(1, 1.05))
    plt.subplot(312)
    plt.plot(tt, global_thigh_angVel_2x5hz[0:total_step],'k-')
    plt.plot(tt, global_thigh_angVel_2x5hz_pred[0:total_step], 'b--')
    plt.ylabel('$\dot{\\theta}_{Y_{2.5Hz}} ~(deg/s)$')
    plt.xlim([0, 13.6])
    plt.subplot(313)
    plt.plot(tt, global_thigh_angVel_2hz[0:total_step],'k-')
    plt.plot(tt, global_thigh_angVel_2hz_pred[0:total_step], 'b--')
    plt.ylabel('$\dot{\\theta}_{Y_{2Hz}} ~(deg/s)$')
    plt.xlim([0, 13.6])
    plt.xlabel('time (s)')
    
    plt.figure('Fictitious Sensors: atan2')
    plt.plot(tt, atan2[0:total_step],'k-')
    plt.plot(tt, at2[0:total_step], 'b--')
    plt.ylabel('$atan2~(rad)$')
    plt.xlabel('time (s)')
    plt.ylim([0, 7.5])
    plt.xlim([0, 13.6])
    plt.legend(('actual', 'least squares'))
    
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
    
    # APPEND NEW DATA
    """
    with open('Continuous_measurement_data_new.pickle', 'rb') as file:
    	Continuous_measurement_data = pickle.load(file)

    dt = 1/100
    for subject in Conti_subject_names():
        for trial in raw_walking_data['Continuous'][subject].keys():
            if trial == 'subjectdetails':
                continue
            for side in ['left', 'right']:
                # APPEND NEW DATA: Global_thigh_angVel_Y
                gt_Y = Continuous_measurement_data[subject][trial][side]['global_thigh_angle_Y'][0, :]
                #v = np.diff(gt_Y) / dt
                #global_thigh_angVel_5hz = butter_lowpass_filter(np.insert(v, 0, 0), 5, 1/dt, order = 1)
                #global_thigh_angVel_2x5hz = butter_lowpass_filter(np.insert(v, 0, 0), 2.5, 1/dt, order = 1)
                #global_thigh_angVel_2hz = butter_lowpass_filter(np.insert(v, 0, 0), 2, 1/dt, order = 1)

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
                #Continuous_measurement_data[subject][trial][side]['global_thigh_angVel_2hz'] = global_thigh_angVel_2hz
                Continuous_measurement_data[subject][trial][side]['atan2_s'] = atan2
    
    with open('Continuous_measurement_data_new_s.pickle', 'wb') as file:
    	pickle.dump(Continuous_measurement_data, file)

    """
    subject = 'AB03'
    trial = 's1i10'
    side = 'left'
    #plot_Conti_data(subject, trial, side)
    
    #####################
    
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
    