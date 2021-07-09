import numpy as numpy
import h5py as hp
import pickle
from EKF import load_Psi
from incline_experiment_utils import *
from model_framework import *
from model_fit import *
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

def Conti_global_thigh_angle_Y(subject, trial, side):
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
    with open('Continuous_data/Continuous_measurement_data.pickle', 'rb') as file:
        Continuous_measurement_data = pickle.load(file)

    start_index, end_index = Conti_start_end(subject, trial, side)
    global_thigh_angle_Y = Continuous_measurement_data[subject][trial][side]['global_thigh_angle_Y'][0, start_index:end_index]
    force_z_ankle = Continuous_measurement_data[subject][trial][side]['force_ankle_z'][0, start_index:end_index]
    force_x_ankle = Continuous_measurement_data[subject][trial][side]['force_ankle_x'][0, start_index:end_index]
    moment_y_ankle = Continuous_measurement_data[subject][trial][side]['moment_ankle_y'][0, start_index:end_index]
    #global_thigh_angVel_5hz = Continuous_measurement_data[subject][trial][side]['global_thigh_angVel_5hz'][start_index:end_index]
    #global_thigh_angVel_2x5hz = Continuous_measurement_data[subject][trial][side]['global_thigh_angVel_2x5hz'][start_index:end_index]
    global_thigh_angVel_2hz = Continuous_measurement_data[subject][trial][side]['global_thigh_angVel_2hz'][start_index:end_index]
    atan2 = Continuous_measurement_data[subject][trial][side]['atan2_s'][start_index:end_index]

    return global_thigh_angle_Y, force_z_ankle, force_x_ankle, moment_y_ankle,\
           global_thigh_angVel_2hz, atan2

def plot_Conti_measurement_data(subject, trial, side):
    phases, phase_dots, step_lengths, ramps = Conti_state_vars(subject, trial, side)
    global_thigh_angle_Y, force_z_ankle, force_x_ankle, moment_y_ankle, global_thigh_angVel_2hz, atan2\
                                         = load_Conti_measurement_data(subject, trial, side)
    m_model = model_loader('Measurement_model_6.pickle') # load new model w/ linear phase_dot
    Psi = load_Psi('Generic')

    global_thigh_angle_Y_pred = model_prediction(m_model.models[0], Psi['global_thigh_angle'], phases, phase_dots, step_lengths, ramps)
    force_z_ankle_pred = model_prediction(m_model.models[1], Psi['force_Z'], phases, phase_dots, step_lengths, ramps)
    force_x_ankle_pred = model_prediction(m_model.models[2], Psi['force_X'], phases, phase_dots, step_lengths, ramps)
    moment_y_ankle_pred = model_prediction(m_model.models[3],Psi['moment_Y'], phases, phase_dots, step_lengths, ramps)
    #global_thigh_angVel_5hz_pred = model_prediction(m_model.models[4], Psi[4], phases, phase_dots, step_lengths, ramps)
    #global_thigh_angVel_2x5hz_pred = model_prediction(m_model.models[5], Psi[5], phases, phase_dots, step_lengths, ramps)
    global_thigh_angVel_2hz_pred = model_prediction(m_model.models[4], Psi['global_thigh_angle_vel'], phases, phase_dots, step_lengths, ramps)
    atan2_pred = model_prediction(m_model.models[5], Psi['atan2'], phases, phase_dots, step_lengths, ramps) + 2*np.pi*phases
 
    # compute rmse
    print("subject: ",  subject)
    print("trial: ",  trial)
    """
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
    """
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
    total_step =  np.shape(global_thigh_angle_Y)[0]#int(heel_strike_index[20, 0])+1 #
    tt = 0.01 * np.arange(total_step)
    plt.figure('Original Measurements')
    plt.subplot(411)
    plt.plot(tt, global_thigh_angle_Y[0:total_step], 'k-')
    plt.plot(tt, global_thigh_angle_Y_pred[0:total_step],'b--')
    plt.ylim([-25, 105])
    #plt.xlim([0, 13.6])
    plt.legend(('actual', 'least squares'))
    #plt.legend(('actual', 'least squares'), bbox_to_anchor=(1, 1.05))
    plt.ylabel('$\\theta_Y~(deg)$')
    plt.subplot(412)
    plt.plot(tt, force_z_ankle[0:total_step], 'k-')
    plt.plot(tt, force_z_ankle_pred[0:total_step], 'b--')
    plt.ylabel('$f_Z~(N)$')
    #plt.xlim([0, 13.6])
    plt.subplot(413)
    plt.plot(tt, force_x_ankle[0:total_step], 'k-')
    plt.plot(tt, force_x_ankle_pred[0:total_step], 'b--')
    plt.ylabel('$f_X~(N)$')
    #plt.xlim([0, 13.6])
    plt.subplot(414)
    plt.plot(tt, moment_y_ankle[0:total_step], 'k-')
    plt.plot(tt, moment_y_ankle_pred[0:total_step], 'b--')
    plt.ylabel('$m_Y~(N \cdot m)$')
    #plt.xlim([0, 13.6])
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
    
    plt.figure('Fictitious Sensors')
    plt.subplot(211)
    plt.plot(tt, global_thigh_angVel_2hz[0:total_step],'k-')
    plt.plot(tt, global_thigh_angVel_2hz_pred[0:total_step], 'b--')
    plt.ylabel('$\dot{\\theta}_{Y_{2Hz}} ~(deg/s)$')
    #plt.xlim([0, 13.6])
    plt.subplot(212)
    plt.plot(tt, atan2[0:total_step],'k-')
    plt.plot(tt, at2[0:total_step], 'b--')
    plt.ylabel('$atan2~(rad)$')
    plt.xlabel('time (s)')
    plt.ylim([0, 7.5])
    #plt.xlim([0, 13.6])
    plt.legend(('actual', 'least squares'))
    
    plt.show()

def Conti_maxmin(subject, plot = True):
    #for subject in Conti_subject_names():
    phase_dots_sup = np.zeros((9,1)) # 9 different ramp angles
    phase_dots_inf = np.zeros((9,1))
    phase_dots_mean = np.zeros((9,1))
    step_lengths_sup = np.zeros((9,1))
    step_lengths_inf = np.zeros((9,1))
    step_lengths_mean = np.zeros((9,1))
    ramp_code = ['d10', 'd7x5', 'd5', 'd2x5', 'i0', 'i2x5', 'i5', 'i7x5', 'i10']
    ramp_angles = [-10, -7.5, -5, -2.5, 0, 2.5, 5, 7.5, 10]
    #for subject in Conti_subject_names():
    #for subject in ['AB03']:
    for r in range(9): # LOOP THROUGH ALL ANGLES!!
        #phases_max = -1000000
        #phases_min = 1000000
        phase_dots_max = -1000000
        phase_dots_min = 1000000
        step_lengths_max = -1000000
        step_lengths_min = 1000000
        #ramps_max = -1000000
        #ramps_min = 1000000
        for trial in raw_walking_data['Continuous'][subject].keys():
            if str(trial)[-3:] == ramp_code[r] or str(trial)[-4:] == ramp_code[r] or str(trial)[-2:] == ramp_code[r]:
            #if trial == 'subjectdetails':
            #    continue
                for side in ['left', 'right']:
                    phases, phase_dots, step_lengths, ramps = Conti_state_vars(subject, trial, side)
                    #if np.max(phases) > phases_max:
                    #    phases_max = np.max(phases)
                    #if np.min(phases) < phases_min:
                    #    phases_min = np.min(phases)

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
                        
                    #if np.max(ramps) > ramps_max:
                    #    ramps_max = np.max(ramps)
                    #if np.min(ramps) < ramps_min:
            
                    #    ramps_min = np.min(ramps)
        phase_dots_mean[r] = phase_dots_mean[r]/6
        phase_dots_sup[r] = phase_dots_max
        phase_dots_inf[r] = phase_dots_min
        step_lengths_mean[r] = step_lengths_mean[r]/6
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
    #for subject in ['AB03']:
        nan_dict[subject] = dict()
        for trial in Conti_trial_names(subject):
            if trial == 'subjectdetails':
                continue
            nan_dict[subject][trial] = dict()
            for side in ['left', 'right']:
                nan_dict[subject][trial][side] = True
                global_thigh_angle_Y, _, _, _, _, _, _, _ = load_Conti_measurement_data(subject, trial, side)
                for i in range(3, len(global_thigh_angle_Y)):
                    if global_thigh_angle_Y[i] == 0 and global_thigh_angle_Y[i-1] == 0 and global_thigh_angle_Y[i-2] == 0\
                        and global_thigh_angle_Y[i-3] == 0:
                        nan_dict[subject][trial][side] = False
                        print(subject + "/"+ trial + "/"+ side)
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

    with open('Psi/Psi_knee_G.pickle', 'rb') as file:
        Psi_knee = pickle.load(file)
    with open('Psi/Psi_ankle_G.pickle', 'rb') as file:
        Psi_ankle = pickle.load(file)
    
    knee_angle_pred = model_prediction(c_model.models[0], Psi_knee, phases, phase_dots, step_lengths, ramps)
    ankle_angle_pred = model_prediction(c_model.models[1], Psi_ankle, phases, phase_dots, step_lengths, ramps)

    plt.figure("Joint Angle Control")
    start = 500
    end = 2500
    plt.subplot(211)
    plt.plot(knee_angle[start:end], 'k-')
    plt.plot(knee_angle_pred[start:end], 'b-')
    plt.ylabel('knee angle')
    plt.legend(('actual', 'least squares'))
    plt.subplot(212)
    plt.plot(ankle_angle[start:end], 'k-')
    plt.plot(ankle_angle_pred[start:end], 'b-') 
    plt.ylabel('ankle angle')
    plt.show()

def detect_knee_over_extention():
    c_model = model_loader('Control_model.pickle')
    with open('Psi/Psi_knee_G.pickle', 'rb') as file:
        Psi_knee = pickle.load(file)
    with open('Psi/Psi_ankle_G.pickle', 'rb') as file:
        Psi_ankle = pickle.load(file)
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

if __name__ == '__main__':
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
    with open('Continuous_measurement_data_unscaled.pickle', 'rb') as file:
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
    trial = 's0x8i0'
    side = 'right'

    #detect_knee_over_extention()
    #detect_nan_in_measurements()
    #detect_nan_in_joints()

    #jointangles = raw_walking_data['Continuous'][subject][trial]['kinematics']['jointangles'][side]
    #k_Y = -jointangles['knee'][0, :]
    #k_X = -jointangles['knee'][1, :]
    #k_Z = -jointangles['knee'][2, :]
    #plt.plot(k_Y)
    #plt.plot(k_X)
    #plt.plot(k_Z)
    #plt.legend(('Y', 'X', 'Z'))
    #plt.ylabel('knee angle')
    #plt.show()

    plot_Conti_joints_angles(subject, trial, side)
    #Conti_global_thigh_angle_Y(subject, trial, side)
    #plt.show()
    #plot_Conti_measurement_data(subject, trial, side)
    #Conti_maxmin('AB01', plot = True)

    ######## test real0time filters #############
    """
    global_thigh_angle_Y, _, _, _, _, _, global_thigh_angVel_2hz, atan2 = load_Conti_measurement_data(subject, trial, side)
    
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
    global_thigh_angle_vel_lp = np.zeros((len(global_thigh_angle_Y), 1))
    Atan2 = np.zeros((len(global_thigh_angle_Y), 1))
    for i in range(len(global_thigh_angle_Y)):
        if i == 0:
            global_thigh_angle_vel_lp[i] = 0 
        else:
            global_thigh_angle_vel = (global_thigh_angle_Y[i] - global_thigh_angle_0) / dt
            # low-pass filtering
            vel_lp, z_lp_1 = lfilter(b_lp, a_lp, [global_thigh_angle_vel], zi = z_lp_1)
            global_thigh_angle_vel_lp[i] = vel_lp[0]

        global_thigh_angle_0 = global_thigh_angle_Y[i]

        # Compute atan2
        ang_bp, z_bp = lfilter(b_bp, a_bp, [global_thigh_angle_Y[i]], zi = z_bp) 
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
    plt.plot(global_thigh_angVel_2hz, 'k--')
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