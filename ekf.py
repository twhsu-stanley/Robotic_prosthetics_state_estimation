import numpy as np
from model_framework import *
from data_generators import *
from continuous_data import *
from model_fit import *
import matplotlib.pyplot as plt

def warpToOne(phase):
    phase_wrap = np.remainder(phase, 1)
    while np.abs(phase_wrap) > 1:
        phase_wrap = phase_wrap - np.sign(phase_wrap)
    return phase_wrap

class myStruct:
    pass

class extended_kalman_filter:
    def __init__(self, system, init):
        # EKF Construct an instance of this class
        # Inputs:
        #   system: system and noise models
        #   init:   initial state mean and covariance
        self.f = system.f  # process model
        self.A = system.A  # system matrix Jacobian
        self.Q = system.Q  # process model noise covariance

        self.h = system.h  # measurement model
        self.R = system.R  # measurement noise covariance
        
        self.x = init.x  # state mean
        self.Sigma = init.Sigma  # state covariance

    def prediction(self, dt):
        # EKF propagation (prediction) step
        self.x_pred = self.f(self.x, dt)  # predicted state
        self.x_pred[0, 0] = warpToOne(self.x_pred[0, 0]) # wrap to be between 0 and 1
        self.Sigma_pred = self.A(dt) @ self.Sigma @ self.A(dt).T + self.Q  # predicted state covariance

    def correction(self, z, Psi):
        # EKF correction step
        # Inputs:
        #   z:  measurement

        # evaluate measurement Jacobian at current operating point
        H = self.h.evaluate_dh_func(Psi, self.x_pred[0,0], self.x_pred[1,0], self.x_pred[2,0], self.x_pred[3,0])

        # predicted measurements
        z_hat = self.h.evaluate_h_func(Psi, self.x_pred[0,0], self.x_pred[1,0], self.x_pred[2,0], self.x_pred[3,0])
        
        # innovation
        z = np.array([[z[0]], [z[1]], [z[2]], [z[3]]])
        self.v = z - z_hat
        #print("innov: \n", self.v)

        # innovation covariance
        S = H @ self.Sigma_pred @ H.T + self.R  

        # filter gain
        K = self.Sigma_pred @ H.T @ np.linalg.inv(S)
        #print("K: \n", K)

        # correct the predicted state statistics
        self.x = self.x_pred + K @ self.v
        self.x[0, 0] = warpToOne(self.x[0, 0])

        I = np.eye(np.shape(self.x)[0])
        self.Sigma = (I - K @ H) @ self.Sigma_pred

def A(dt):
    return np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

def process_model(x, dt):
    #dt = 0.01 # data sampling rate: 100 Hz
    return A(dt) @ x
    
if __name__ == '__main__':
    subject = 'AB02'
    trial= 's1x2i5'
    side = 'right'

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
    sys.Q = np.diag([0, 1e-7, 1e-8, 1e-10]) # process model noise covariance
    
    with open('Measurement_error_cov.pickle', 'rb') as file:
        R = pickle.load(file)
    
    sys.R = R[subject] # measurement noise covariance

    # initialize the state
    init = myStruct()
    init.x = np.array([[phases[0]+0.1], [phase_dots[0]], [step_lengths[0]], [ramps[0]]])
    init.Sigma = np.diag([0, 1e-14, 1e-14, 10])

    ekf = extended_kalman_filter(sys, init)

    z = np.array([[global_thigh_angle_Y],\
                  [force_z_ankle], \
                  [force_x_ankle],\
                  [moment_y_ankle]])
    z = np.squeeze(z)

    x = []  # state estimate
    #x.append(init.x)

    #Sigma = []
    #Sigma.append(np.diag(init.Sigma))
    
    for i in range(np.shape(z)[1]):
        ekf.prediction(dt)
        ekf.correction(z[:, i], Psi)

        x.append(ekf.x)
        #Sigma.append(np.diag(ekf.Sigma))
        
    x = np.array(x).squeeze()
    #Sigma = np.array(Sigma)

    # evaluate robustness
    # compare x and ground truth:
    track = True
    track_tol = 0.05
    heel_strike_index = Conti_heel_strikes(subject, trial, side) - Conti_heel_strikes(subject, trial, side)[0]
    # start checking tracking after the 3rd stride
    for i in range(3, np.size(heel_strike_index)):
        if i != np.size(heel_strike_index) - 1:
            start = int(heel_strike_index[i]) + 10
            end = int(heel_strike_index[i+1]) - 10
            print(max(abs(phases[start:end] - x[start:end, 0])))
            track = track and all(abs(phases[start:end] - x[start:end, 0]) < track_tol)
    
    print("track? ",track)

    print("Sigma at final step: \n", ekf.Sigma)

    #time_step = 100
    #track = all(e < 0.01 for e in abs(x[time_step:, 0] - phases[time_step:]))
    plt.figure()
    plt.plot(phases)
    plt.plot(x[:, 0], '--')
    plt.plot(heel_strike_index[3:], np.zeros(np.size(heel_strike_index[3:])), 'rx')
    plt.show()

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

    """
    plt.figure()
    plt.subplot(411)
    plt.plot(Sigma[:,0])
    plt.subplot(412)
    plt.plot(Sigma[:,1])
    plt.subplot(413)
    plt.plot(Sigma[:,2])
    plt.subplot(414)
    plt.plot(Sigma[:,3])
    """
    #plot_Conti_data(subject, trial, side)

    plt.show()