import numpy as numpy
import h5py as hp
import pickle
from wrapping import wrapTo2pi
from incline_experiment_utils import *
from model_framework import *
from scipy.signal import butter, lfilter, lfilter_zi
from load_Psi import load_Psi 

## Process, store, and generate continuous (streaming) data for the incExp dataset

raw_walking_data = hp.File("../InclineExperiment.mat", "r")

def get_Continuous_subject_names():
    return raw_walking_data['Continuous'].keys()

def get_Continuous_trial_names(subject):
    return raw_walking_data['Continuous'][subject].keys()

def get_Continuous_heel_strikes(subject, trial, side):
    return raw_walking_data['Gaitcycle'][subject][trial]['cycles'][side]['frame'][:][:,0]

def get_Continuous_start_end(subject, trial, side):
    heel_strike_index = get_Continuous_heel_strikes(subject, trial, side)
    start_index = heel_strike_index[0]
    end_index = heel_strike_index[np.size(heel_strike_index)-1]
    return int(start_index), int(end_index)

def get_Continuous_globalThighAngles(subject, trial, side):
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

def store_Continuous_globalThighAngles():
    Continuous_globalThighAngles_data = dict()
    for subject in get_Continuous_subject_names():
        print("subject: ",  subject)
        Continuous_globalThighAngles_data[subject] = dict()
        for trial in raw_walking_data['Continuous'][subject].keys():
            if trial == 'subjectdetails':
                continue
            print("   trial: ",  trial)
            Continuous_globalThighAngles_data[subject][trial] = dict()
            for side in ['left', 'right']:
                print("      side: ", side)
                Continuous_globalThighAngles_data[subject][trial][side] = dict()
                Continuous_globalThighAngles_data[subject][trial][side] = get_Continuous_globalThighAngles(subject, trial, side)
            
    with open('Continuous_data_incExp/Continuous_globalThighAngles_data.pickle', 'wb') as file:
    	pickle.dump(Continuous_globalThighAngles_data, file)

def get_Continuous_reaction_wrench(subject, trial, side):
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

def get_Continuous_state_vars(subject, trial, side):
    heel_strike_index = raw_walking_data['Gaitcycle'][subject][trial]['cycles'][side]['frame'][:]
    Continuous_time = raw_walking_data['Continuous'][subject][trial]['time'][:]
    dt = Continuous_time[0, 1] - Continuous_time[0, 0] # 0.01 s/ 100 Hz
    ptr = raw_walking_data['Continuous'][subject][trial]['description'][1][0]
    walking_speed = raw_walking_data[ptr][:][0, 0]
    ptr = raw_walking_data['Continuous'][subject][trial]['description'][1][1]
    incline = raw_walking_data[ptr][:][0, 0]
    
    if side == 'left':
        ptr_sl = raw_walking_data['Gaitcycle'][subject]['subjectdetails'][1][4]
    elif side == 'right':
        ptr_sl = raw_walking_data['Gaitcycle'][subject]['subjectdetails'][1][5]
    leg_length = raw_walking_data[ptr_sl] # mm

    phase = np.zeros((np.size(Continuous_time)))
    phase_dot = np.zeros((np.size(Continuous_time)))
    step_length = np.zeros((np.size(Continuous_time)))
    ramp = incline * np.ones((np.size(Continuous_time)))

    for i in range(np.size(heel_strike_index)):
        if i != np.size(heel_strike_index) - 1:
            stride_steps = int(heel_strike_index[i+1] - heel_strike_index[i])
            for k in range(stride_steps):
                phase[int(heel_strike_index[i]) + k] = k * 1/stride_steps
                phase_dot[int(heel_strike_index[i]) + k] = 1/stride_steps / dt
                step_length[int(heel_strike_index[i]) + k] = walking_speed * stride_steps * dt / leg_length * 1000

    # truncate the signal s.t. it starts and ends at heel strikes
    start_index, end_index = get_Continuous_start_end(subject, trial, side)
    phase = phase[start_index:end_index]
    phase_dot = phase_dot[start_index:end_index]
    step_length = step_length[start_index:end_index]
    ramp = ramp[start_index:end_index]
    
    return phase, phase_dot, step_length, ramp

def get_Continuous_measurement_data(subject, trial, side):
    start_index, end_index = get_Continuous_start_end(subject, trial, side)

    # Global thigh angles
    with open('Continuous_data_incExp/Continuous_globalThighAngles_data.pickle', 'rb') as file:
        Continuous_globalThighAngles_data = pickle.load(file)
    globalThighAngle = Continuous_globalThighAngles_data[subject][trial][side][0, start_index:end_index]
    #globalThighAngle = get_Continuous_globalThighAngles(subject, trial, side)[0, start_index:end_index]

    # Global thigh angular velocity
    dt = 1/100
    v = np.diff(globalThighAngle) / dt
    gtv = np.insert(v, 0, 0)
    globalThighVelocity = butter_lowpass_filter(gtv, 2, 1/dt, order = 1)

    # Atan2
    gt_bp = butter_bandpass_filter(globalThighAngle, 0.5, 2, 1/dt, order = 2)
    v_bp = np.diff(gt_bp) / dt
    gtv_bp = butter_lowpass_filter(np.insert(v_bp, 0, 0), 2, 1/dt, order = 1)
    atan2 = np.arctan2(-gtv_bp/(2*np.pi*0.8), gt_bp) # scaled
    for i in range(np.shape(atan2)[0]):
        if atan2[i] < 0:
            atan2[i] = atan2[i] + 2 * np.pi
    
    # Global foot angles
    with open('Gait_training_data_incExp/globalFootAngles_offset.pickle', 'rb') as file:
        globalFootAngles_offset = pickle.load(file)

    globalFootAngle = -raw_walking_data['Continuous'][subject][trial]['kinematics']['jointangles'][side]['foot'][0,start_index:end_index] - 90
    globalFootAngle -= globalFootAngles_offset[trial][subject][side]
    
    # Kinetic measurements
    ankleMoment = raw_walking_data['Continuous'][subject][trial]['kinetics']['jointmoment'][side]['ankle'][0, start_index:end_index] / 1000 # N-mm to N-m
    ankleMoment = butter_lowpass_filter(ankleMoment, 7, 100, order = 1)
    tibiaForce = raw_walking_data['Continuous'][subject][trial]['kinetics']['jointforce'][side]['knee'][2, start_index:end_index]
    
    return globalThighAngle, globalThighVelocity, atan2, globalFootAngle, ankleMoment, tibiaForce

def get_Continuous_atan2_scale_shift(subject, trial, side, plot = True):  
    dt = 1/100
    start_index, end_index = get_Continuous_start_end(subject, trial, side)

    # Global thigh angles
    with open('Continuous_data_incExp/Continuous_globalThighAngles_data.pickle', 'rb') as file:
        Continuous_globalThighAngles_data = pickle.load(file)
    globalThighAngle = Continuous_globalThighAngles_data[subject][trial][side][0, start_index:end_index]
    #globalThighAngle = get_Continuous_globalThighAngles(subject, trial, side)[0, start_index:end_index]

    globalThighAngle_lp = butter_lowpass_filter(globalThighAngle, 2, 1/dt, order = 1) # 1st/2nd/3rd order
    globalThighVelocity_lp = np.insert(np.diff(globalThighAngle_lp) / dt, 0, 0)

    globalThighAngle_max = np.zeros(np.shape(globalThighAngle)[0])
    globalThighAngle_min = np.zeros(np.shape(globalThighAngle)[0])
    globalThighVelocity_max = 50 * np.ones(np.shape(globalThighVelocity_lp)[0])
    globalThighVelocity_min = -50 * np.ones(np.shape(globalThighVelocity_lp)[0])
    atan2 = np.ones(np.shape(globalThighVelocity_lp)[0])
    phase_y = np.ones(np.shape(globalThighVelocity_lp)[0])
    phase_x = np.ones(np.shape(globalThighVelocity_lp)[0])
    
    ####
    heel_strike_index = get_Continuous_heel_strikes(subject, trial, side)
    heel_strike_index  = heel_strike_index - start_index

    for i in range(np.shape(globalThighAngle)[0]):
        h1 = int(heel_strike_index[np.where(heel_strike_index <= i)[0][-1]])
        h2 = int(heel_strike_index[np.where(heel_strike_index > i)[0][0]])
        globalThighAngle_max[i] = max(globalThighAngle_lp[h1:h2])
        globalThighAngle_min[i] = min(globalThighAngle_lp[h1:h2])
        globalThighVelocity_max[i] = max(globalThighVelocity_lp[h1:h2])
        globalThighVelocity_min[i] = min(globalThighVelocity_lp[h1:h2])

        gta_shift = (globalThighAngle_max[i] + globalThighAngle_min[i]) / 2
        gta_scale = abs(globalThighVelocity_max[i] - globalThighVelocity_min[i]) / abs(globalThighAngle_max[i]- globalThighAngle_min[i])
        gtv_shift = (globalThighVelocity_max[i] + globalThighVelocity_min[i]) / 2

        phase_y[i] = - (globalThighVelocity_lp[i] - gtv_shift)
        phase_x[i] = gta_scale * (globalThighAngle_lp[i] - gta_shift)

        atan2[i] = np.arctan2(phase_y[i], phase_x[i])
        if atan2[i] < 0:
            atan2[i] = atan2[i] + 2 * np.pi
    
    if plot == False:
        return atan2

    else:
        phases, phase_dots, step_lengths, ramps = get_Continuous_state_vars(subject, trial, side)
        m_model = model_loader('Measurement_model_globalThighAngles_globalThighVelocities_atan2_globalFootAngles.pickle')
        Psi = load_Psi()
        atan2_pred = model_prediction(m_model.models[2], Psi['atan2'], phases, phase_dots, step_lengths,ramps) + 2*np.pi*phases
        atan2_pred = wrapTo2pi(atan2_pred)

        plt.figure()
        plt.subplot(311)
        plt.plot(globalThighAngle_lp, 'k-', linewidth = 2)
        plt.plot(globalThighAngle_max)
        plt.plot(globalThighAngle_min)
        plt.grid()
        plt.subplot(312)
        plt.plot(globalThighVelocity_lp, 'k-', linewidth = 2)
        plt.plot(globalThighVelocity_max, 'r-')
        plt.plot(globalThighVelocity_min, 'b-')
        plt.grid()
        plt.subplot(313)
        plt.plot(atan2)
        plt.grid()

        plt.figure("Atan2 phase plane")
        plt.plot(phase_x[2000:3000], phase_y[2000:3000], linewidth = 2)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid()

        phases, _, _, _ = get_Continuous_state_vars(subject, trial, side)
        idx_start = int(heel_strike_index[5])
        idx_end = int(heel_strike_index[10]) + 1
        tt = 0.01 * np.arange(idx_end - idx_start)
        fig, ax1 = plt.subplots()
        color = 'tab:blue'
        ax1.set_xlabel('time (s)')
        ax1.set_ylabel('atan2')
        ax1.plot(tt, atan2[idx_start:idx_end], color=color)
        ax1.plot(tt, atan2_pred[idx_start:idx_end], 'g--', alpha = 0.5)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True)
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'k'
        ax2.set_ylabel('phase', color=color)  # we already handled the x-label with ax1
        ax2.plot(tt, phases[idx_start:idx_end], color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()

def plot_Continuous_measurement_data(subject, trial, side):
    print("subject: ",  subject, "| trial: ",  trial, " | side: ", side)
    
    phases, phase_dots, step_lengths, ramps = get_Continuous_state_vars(subject, trial, side)
    globalThighAngle, globalThighVelocity, atan2, globalFootAngle, ankleMoment, tibiaForce = get_Continuous_measurement_data(subject, trial, side)
    
    atan2_ss = get_Continuous_atan2_scale_shift(subject, trial, side, plot = False) # use the shifted & scalsed version
    
    m_model = model_loader('Measurement_model_globalThighAngles_globalThighVelocities_atan2_globalFootAngles.pickle')
    Psi = load_Psi()

    globalThighAngle_pred = model_prediction(m_model.models[0], Psi['globalThighAngles'], phases, phase_dots, step_lengths,ramps)
    globalThighVelocity_pred = model_prediction(m_model.models[1], Psi['globalThighVelocities'], phases, phase_dots, step_lengths,ramps)
    
    #ankleMoment_pred = model_prediction(m_model.models[4], Psi['ankleMoment'], phases, phase_dots, step_lengths, ramps)
    #tibiaForce_pred = model_prediction(m_model.models[5], Psi['tibiaForce'], phases, phase_dots, step_lengths, ramps)
    
    atan2_pred = model_prediction(m_model.models[2], Psi['atan2'], phases, phase_dots, step_lengths,ramps) + 2*np.pi*phases
    atan2_pred = wrapTo2pi(atan2_pred)
    residuals_atan2 = atan2_ss - atan2_pred
    residuals_atan2 = np.arctan2(np.sin(residuals_atan2), np.cos(residuals_atan2))
    
    globalFootAngle_pred = model_prediction(m_model.models[3], Psi['globalFootAngles'], phases, phase_dots, step_lengths, ramps)
    
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
    plt.plot(atan2[0:1600], '-.')
    plt.plot(atan2_ss[0:1600])
    plt.plot(atan2_pred[0:1600], '--')
    plt.legend(['atan2', 'atan2_shifted_scaled', 'atan2_predicted'])
    plt.subplot(212)
    a1 = atan2_ss[0:1600] - 2*np.pi*phases[0:1600]
    for i in range(len(a1)):
        a1[i] = np.arctan2(np.sin(a1[i]), np.cos(a1[i]))
    plt.plot(a1)
    a2 = atan2_pred[0:1600] - 2*np.pi*phases[0:1600]
    for i in range(len(a2)):
        a2[i] = np.arctan2(np.sin(a2[i]), np.cos(a2[i]))
    plt.plot(a2)
    plt.legend(['atan2-phase*2pi', 'least-squares fitting', 'new'])
    
    #heel_strike_index = get_Continuous_heel_strikes(subject, trial, side) - get_Continuous_heel_strikes(subject, trial, side)[0]
    total_step =  int(np.shape(globalThighAngle)[0] / 1)
    tt = 0.01 * np.arange(total_step)
    plt.figure('Thigh Measurements')
    plt.subplot(211)
    plt.plot(tt, globalThighAngle[0:total_step], 'k-')
    plt.plot(tt, globalThighAngle_pred[0:total_step],'b--')
    plt.grid()
    #plt.xlim([0, 13.6])
    plt.legend(('actual', 'least squares'))
    #plt.legend(('actual', 'least squares'), bbox_to_anchor=(1, 1.05))
    plt.ylabel('$\\theta_{th}~(deg)$')

    plt.subplot(212)
    plt.plot(tt, globalThighVelocity[0:total_step],'k-')
    plt.plot(tt, globalThighVelocity_pred[0:total_step], 'b--')
    plt.grid()
    plt.ylabel('$\dot{\\theta}_{th} ~(deg/s)$')
    #plt.xlim([0, 13.6])

    plt.figure('Kinetic Measurements')
    plt.subplot(211)
    plt.plot(tt, ankleMoment[0:total_step], 'k-')
    #plt.plot(tt, ankleMoment_pred[0:total_step], 'b--')
    plt.grid()
    plt.ylabel('$m_Y~(N \cdot m)$')
    #plt.xlim([0, 13.6])
    plt.xlabel('time (s)')
    
    plt.subplot(212)
    plt.plot(tt, tibiaForce[0:total_step], 'k-')
    #plt.plot(tt, tibiaForce_pred[0:total_step], 'b--')
    plt.grid()
    plt.ylabel('$f_Z~(N \cdot m)$')
    #plt.xlim([0, 13.6])
    plt.xlabel('time (s)')

    plt.figure('Foot Angle Measurements')
    plt.plot(tt, globalFootAngle[0:total_step], 'k-')
    plt.plot(tt, globalFootAngle_pred[0:total_step], 'b--')
    plt.grid()
    plt.legend(('globalFootAngle', 'predicted globalFootAngle'))
    plt.ylabel('$\\theta_{f}~(deg)$')
    plt.xlabel('time (s)')

    plt.figure('L COP')
    plt.subplot(311)
    plt.plot(tt, ankleMoment[0:total_step], 'k-')
    #plt.plot(tt, ankleMoment_pred[0:total_step], 'b--')
    plt.ylabel('$m_Y~(N \cdot m)$')
    plt.subplot(312)
    plt.plot(tt, tibiaForce[0:total_step], 'k-')
    #plt.plot(tt, tibiaForce_pred[0:total_step], 'b--')
    plt.ylabel('$f_Z~(N \cdot m)$')
    plt.subplot(313)
    plt.plot(tt, L_cop[0:total_step], 'k-')
    plt.ylabel('L cop (m)')

    plt.show()

def get_Continuous_maxmin(plot = True):
    #for subject in get_Continuous_subject_names():
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
        for subject in get_Continuous_subject_names():
            for trial in raw_walking_data['Continuous'][subject].keys():
                if str(trial)[-3:] == ramp_code[r] or str(trial)[-4:] == ramp_code[r] or str(trial)[-2:] == ramp_code[r]:
                    for side in ['left', 'right']:
                        _, phase_dots, step_lengths, _ = get_Continuous_state_vars(subject, trial, side)

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
    
    saturation_range = np.array([np.max(phase_dots_sup), np.min(phase_dots_inf), np.max(step_lengths_sup), np.min(step_lengths_inf)])

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
    for subject in get_Continuous_subject_names():
        nan_dict[subject] = dict()
        for trial in get_Continuous_trial_names(subject):
            if trial == 'subjectdetails':
                continue
            nan_dict[subject][trial] = dict()
            for side in ['left', 'right']:
                nan_dict[subject][trial][side] = True
                globalThighAngle, _, _, globalFootAngle, ankleMoment, tibiaForce= get_Continuous_measurement_data(subject, trial, side)
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
    with open('Continuous_data_incExp/Measurements_with_Nan.pickle', 'wb') as file:
        pickle.dump(nan_dict, file)

def get_Continuous_joints_angles(subject, trial, side):
    jointangles = raw_walking_data['Continuous'][subject][trial]['kinematics']['jointangles'][side]
    start_index, end_index = get_Continuous_start_end(subject, trial, side)
    knee_angle = -jointangles['knee'][0, start_index:end_index]
    ankle_angle = -jointangles['ankle'][0, start_index:end_index]
    return knee_angle, ankle_angle

def plot_Continuous_joints_angles(subject, trial, side):
    phases, phase_dots, step_lengths, ramps = get_Continuous_state_vars(subject, trial, side)
    knee_angle, ankle_angle = get_Continuous_joints_angles(subject, trial, side)
    
    c_model = model_loader('Control_model_kneeAngles_ankleAngles.pickle')

    with open('Psi/Psi_kneeAngles', 'rb') as file:
        Psi_knee = pickle.load(file)
    with open('Psi/Psi_ankleAngles', 'rb') as file:
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
    for subject in get_Continuous_subject_names():
        for trial in get_Continuous_trial_names(subject):
            if trial == 'subjectdetails':
                continue
            for side in ['left', 'right']:
                knee_angle, ankle_angle = get_Continuous_joints_angles(subject, trial, side)
                phases, phase_dots, step_lengths, ramps = get_Continuous_state_vars(subject, trial, side)
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

    for subject in get_Continuous_subject_names():
        nan_dict[subject] = dict()
        for trial in get_Continuous_trial_names(subject):
            if trial == 'subjectdetails':
                continue
            nan_dict[subject][trial] = dict()
            for side in ['left', 'right']:
                
                flag_k = True
                flag_a = True

                nan_dict[subject][trial][side] = True
                knee_angle, ankle_angle = get_Continuous_joints_angles(subject, trial, side)
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

    with open('Continuous_data_incExp/KneeAngles_with_Nan.pickle', 'wb') as file:
        pickle.dump(nan_dict, file)

def plot_Continuous_kinetics_data(subject, trial, side):

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

if __name__ == '__main__':
    #detect_nan_in_measurements()
    #get_globalFootAngle_offset()
    #store_Continuous_globalThighAngles()

    subject = 'AB10'
    trial = 's1x2i0'
    side = 'left'

    #get_Continuous_measurement_data(subject, trial, side)
    #plot_Continuous_measurement_data(subject, trial, side)

    get_Continuous_atan2_scale_shift(subject, trial, side, plot = True)
    #plot_Continuous_kinetics_data(subject, trial, side)
    #plot_Continuous_joints_angles(subject, trial, side)
