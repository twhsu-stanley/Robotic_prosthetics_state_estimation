import numpy as numpy
import h5py as hp
import pickle
from incline_experiment_utils import *
from model_framework import *

# Generate measurement data from the "Continuous" data structure

raw_walking_data = hp.File("../InclineExperiment.mat", "r")

def Conti_subject_names():
    return raw_walking_data['Continuous'].keys()

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
    
    return phase, phase_dot, step_length, ramp

def Conti_start_end(subject, trial, side):
    heel_strike_index = raw_walking_data['Gaitcycle'][subject][trial]['cycles'][side]['frame'][:]
    start_index = heel_strike_index[0]
    end_index = heel_strike_index[np.size(heel_strike_index)-1]
    return int(start_index), int(end_index)

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

    # Test plot
    subject = 'AB09'
    trial= 's1x2i7x5'
    side = 'left'

    with open('Continuous_measurement_data.pickle', 'rb') as file:
    	Continuous_measurement_data = pickle.load(file)

    global_thigh_angle_Y = Continuous_measurement_data[subject][trial][side]['global_thigh_angle_Y'][0,:]
    force_z_ankle = Continuous_measurement_data[subject][trial][side]['force_ankle_z'][0,:]
    force_x_ankle = Continuous_measurement_data[subject][trial][side]['force_ankle_x'][0,:]
    moment_y_ankle = Continuous_measurement_data[subject][trial][side]['moment_ankle_y'][0,:]

    phases, phase_dots, step_lengths, ramps = Conti_state_vars(subject, trial, side)

    m_model = model_loader('Measurement_model.pickle')

    with open('Measurement_model_coeff.npz', 'rb') as file:
        Measurement_model_coeff = np.load(file, allow_pickle = True)
        psi_thigh_Y = Measurement_model_coeff['global_thigh_angle_Y']
        psi_force_z = Measurement_model_coeff['reaction_force_z_ankle']
        psi_force_x = Measurement_model_coeff['reaction_force_x_ankle']
        psi_moment_y = Measurement_model_coeff['reaction_moment_y_ankle']

    Conti_time = raw_walking_data['Continuous'][subject][trial]['time'][:]
    n_s = np.size(Conti_time) # number of data poitns

    global_thigh_angle_Y_pred = model_prediction(m_model.models[0], psi_thigh_Y.item()[subject], phases, phase_dots, step_lengths, ramps)
    force_z_ankle_pred = model_prediction(m_model.models[1], psi_force_z.item()[subject], phases, phase_dots, step_lengths, ramps)
    force_x_ankle_pred = model_prediction(m_model.models[2], psi_force_x.item()[subject], phases, phase_dots, step_lengths, ramps)
    moment_y_ankle_pred = model_prediction(m_model.models[3], psi_moment_y.item()[subject], phases, phase_dots, step_lengths, ramps)

    plt.figure()
    plt.subplot(411)
    plt.plot(global_thigh_angle_Y, 'b-')
    plt.plot(global_thigh_angle_Y_pred,'k--')
    plt.legend(['actual','predicted'])
    plt.ylabel('global_thigh_angle_Y')

    plt.subplot(412)
    plt.plot(force_z_ankle, 'b-')
    plt.plot(force_z_ankle_pred, 'k--')
    plt.legend(['actual','predicted'])
    plt.ylabel('force_z_ankle')

    plt.subplot(413)
    plt.plot(force_x_ankle, 'b-')
    plt.plot(force_x_ankle_pred, 'k--')
    plt.legend(['actual','predicted'])
    plt.ylabel('force_x_ankle')

    plt.subplot(414)
    plt.plot(moment_y_ankle, 'b-')
    plt.plot(moment_y_ankle_pred, 'k--')
    plt.legend(['actual','predicted'])
    plt.ylabel('moment_y_ankle')
    

    """
    plt.figure(4)
    plt.title("Ground reaction force in ankle frame: left")
    plt.plot(range(n_s), Continuous_measurement_data[subject][trial][side]['force_ankle_x'][0,:],\
             range(n_s), Continuous_measurement_data[subject][trial][side]['force_ankle_y'][0,:],\
             range(n_s), Continuous_measurement_data[subject][trial][side]['force_ankle_z'][0,:])
    plt.ylabel("force [N]")
    plt.legend(('X_ankle','Y_ankle','Z_ankle'))

    plt.figure(5)
    plt.title("Ground reaction moment in ankle frame: left")
    plt.plot(range(n_s), Continuous_measurement_data[subject][trial][side]['moment_ankle_x'][0,:],\
             range(n_s), Continuous_measurement_data[subject][trial][side]['moment_ankle_y'][0,:],\
             range(n_s), Continuous_measurement_data[subject][trial][side]['moment_ankle_z'][0,:])
    plt.ylabel("Moment [N-m]")
    plt.legend(('X_ankle','Y_ankle','Z_ankle'))
    
    plt.figure(8)
    plt.title("Global thigh angles: Y")
    plt.plot(range(n_s), Continuous_measurement_data[subject][trial][side]['global_thigh_angle_Y'] [0,:], range(n_s), phase[0,:], '--')
    plt.ylabel("angles [deg]")
    
    plt.figure()
    plt.plot(range(n_s), phase[0,:], range(n_s), phase_dot[0,:], range(n_s), step_length[0,:], range(n_s), ramp[0,:])
    """
    plt.show()