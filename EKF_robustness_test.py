import numpy as np
from EKF import *

subject = 'AB01'
trial= 's1x2i5'
side = 'right'

with open('Measurement_error_cov.pickle', 'rb') as file:
        R = pickle.load(file)

def kidnap_test(initial_estimate):
    
    dt = 1/100
    phases, phase_dots, step_lengths, ramps = Conti_state_vars(subject, trial, side)
    global_thigh_angle_Y, force_z_ankle, force_x_ankle, moment_y_ankle = load_Conti_measurement_data(subject, trial, side)
    m_model = model_loader('Measurement_model.pickle')
    Psi = load_Psi(subject)

    # build the system
    sys = myStruct()
    sys.f = process_model
    sys.A = A
    sys.h = m_model
    sys.Q = np.diag([0, 1e-7, 1e-8, 1e-14]) # process model noise covariance
    sys.R = R[subject] # measurement noise covariance

    # initialize the state
    init = myStruct()
    init.x = np.array([[phases[0]], [phase_dots[0]], [step_lengths[0]], [ramps[0]]])
    init.Sigma = np.diag([1e-14, 1e-14, 1e-14, 1e-14])

    ekf = extended_kalman_filter(sys, init)

    z = np.array([[global_thigh_angle_Y],\
                  [force_z_ankle], \
                  [force_x_ankle],\
                  [moment_y_ankle]])
    z = np.squeeze(z)

    x = []  # state estimate
    x.append(init.x)
    
    for i in range(np.shape(z)[1]):
        ekf.prediction(dt)
        ekf.correction(z[:, i])
        x.append(ekf.x)
        
    x = np.array(x).squeeze()

    print("Sigma at final step: \n", ekf.Sigma)

    # plot results
    plt.figure()
    plt.subplot(411)
    plt.plot(phases)
    plt.plot(x[:, 0], '--')
    plt.ylabel('phase')

    plt.subplot(412)
    plt.plot(phase_dots)
    plt.plot(x[:, 1], '--')
    plt.ylabel('phase dot')
    #plt.ylim(phase_dots.min()-0.2, phase_dots.max() +0.2)

    plt.subplot(413)
    plt.plot(step_lengths)
    plt.plot(x[:, 2], '--')
    #plt.ylim(step_lengths.min()-0.02, step_lengths.max() +0.02)
    plt.ylabel('step length')

    plt.subplot(414)
    plt.plot(ramps)
    plt.plot(x[:, 3], '--')
    plt.ylabel('ramp')
    #plt.ylim(ramps.min()-1, ramps.max()+1)

    
    
    plt.show()