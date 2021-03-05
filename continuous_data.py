import numpy as numpy
from incline_experiment_utils import *
import h5py as hp
import pickle

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

#def Conti_phase_var(subject, trial, side):
    

#def Conti_task_var(subject, trial, side):


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
    with open('Continuous_measurement_data.pickle', 'rb') as file:
    	Continuous_measurement_data = pickle.load(file)

    subject = 'AB07'
    trial= 's0x8i0'
    side = 'right'
    forceplate = raw_walking_data['Continuous'][subject][trial]['kinetics']['forceplate']
    n_s = np.size(forceplate['left']['force'][0,:]) # number of data poitns

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
    plt.plot(range(n_s), Continuous_measurement_data[subject][trial][side]['global_thigh_angle_Y'] [0,:])
    plt.ylabel("angles [deg]")

    plt.show()

