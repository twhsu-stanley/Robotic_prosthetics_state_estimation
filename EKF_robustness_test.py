import numpy as np
from EKF import *

subject = 'AB01'
trial= 's1x2i5'
side = 'right'

with open('Measurement_error_cov.pickle', 'rb') as file:
        R = pickle.load(file)

m_model = model_loader('Measurement_model.pickle')

def robustness_test(subject, trial, side, sys, disturbance):
    dt = 1/100
    Psi = load_Psi(subject)
    phases, phase_dots, step_lengths, ramps = Conti_state_vars(subject, trial, side)
    global_thigh_angle_Y, force_z_ankle, force_x_ankle, moment_y_ankle = load_Conti_measurement_data(subject, trial, side)
    
    sys.R = R[subject]

    # initialize the state
    init = myStruct()
    init.x = init_x + disturbance
    init.Sigma = np.diag([1e-14, 1e-14, 1e-14, 1e-14])

    ekf = extended_kalman_filter(sys, init)

    z = np.array([[global_thigh_angle_Y],\
                  [force_z_ankle], \
                  [force_x_ankle],\
                  [moment_y_ankle]])
    z = np.squeeze(z)

    x = []  # state estimate
    #x.append(init.x)
    
    for i in range(np.shape(z)[1]):
        ekf.prediction(dt)
        ekf.correction(z[:, i])
        x.append(ekf.x)
    x = np.array(x).squeeze()

    # evaluate robustness
    # compare x and ground truth:
    track = True
    heel_strike_index = Conti_heel_strikes(subject, trial, side) - Conti_heel_strikes(subject, trial, side)[0]
    for i in range(np.size(heel_strike_index)):
        if i != np.size(heel_strike_index) - 1:
            start = int(heel_strike_index[i]) + 20
            end = int(heel_strike_index[i+1]) - 20
            track = track and all(abs(phase[start:end] - x[start:end, 0]) < 0.1)
    
    return track

if __name__ == '__main__':
    # build the system
    sys = myStruct()
    sys.f = process_model
    sys.A = A
    sys.h = m_model
    
    # iterate through Q

    for Q_phase_dot in [1e-7]:
        for Q_step_length in [1e-8]:
            for Q_ramp in [1e-4, 1e-8, 1e-14]:
                sys.Q = np.diag([0, Q_phase_dot, Q_step_length, Q_ramp]) # process model noise covariance
                track_count = 0
                for subject in Conti_subject_names():
                    for trial in Conti_trial_names(subject):
                        for side in ['left', 'right']:
                            if robustness_test(subject, trial, side, ekf, disturb) == True:
                                track_count = track_count + 1
                
                
                

    
    