import numpy as np
from model_framework import *
from data_generators import *
import matplotlib.pyplot as plt
import h5py

# load data
subject = 'AB01'
stride_id = 50

m_model = model_loader('Measurement_model.pickle')

with open('Measurement_model_coeff.npz', 'rb') as file:
    Measurement_model_coeff = np.load(file, allow_pickle = True)
    psi_thigh_Y = Measurement_model_coeff['global_thigh_angle_Y']
    psi_force_z = Measurement_model_coeff['reaction_force_z_ankle']
    psi_force_x = Measurement_model_coeff['reaction_force_x_ankle']
    psi_moment_y = Measurement_model_coeff['reaction_moment_y_ankle']
Psi = np.array([psi_thigh_Y.item()[subject],\
                psi_force_z.item()[subject],\
                psi_force_x.item()[subject],\
                psi_moment_y.item()[subject]]) # Psi: 4 x 336

with open('Global_thigh_angle.npz', 'rb') as file:
    g_t = np.load(file)
    global_thigh_angle_Y = g_t[subject][0]

with open('Reaction_wrench.npz', 'rb') as file:
    r_w = np.load(file)
    force_x_ankle = r_w[subject][0]
    force_z_ankle = r_w[subject][2]
    moment_y_ankle = r_w[subject][4]

phases = get_phase(global_thigh_angle_Y)
phase_dots = get_phase_dot(subject)
step_lengths = get_step_length(subject)
ramps = get_ramp(subject)

n = 20
global_thigh_angle_Y = global_thigh_angle_Y[stride_id: stride_id + n,:].ravel()
force_z_ankle = force_z_ankle[stride_id: stride_id + n,:].ravel()
force_x_ankle = force_x_ankle[stride_id: stride_id + n,:].ravel()
moment_y_ankle = moment_y_ankle[stride_id: stride_id + n,:].ravel()

phases = phases[stride_id: stride_id + n,:].ravel()
phase_dots = phase_dots[stride_id: stride_id + n,:].ravel()
step_lengths = step_lengths[stride_id: stride_id + n,:].ravel()
ramps = ramps[stride_id: stride_id + n,:].ravel()

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

    def prediction(self):
        # EKF propagation (prediction) step
        self.x_pred = self.f(self.x)  # predicted state
        self.x_pred[0, 0] = warpToOne(self.x_pred[0, 0]) # wrap to be between 0 and 1
        self.Sigma_pred = self.A @ self.Sigma @ self.A.T + self.Q  # predicted state covariance

    def correction(self, z):
        # EKF correction step
        # Inputs:
        #   z:  measurement

        # evaluate measurement Jacobian at current operating point
        H = self.h.evaluate_dh_func(Psi, self.x_pred[0,0], self.x_pred[1,0], self.x_pred[2,0], self.x_pred[3,0])

        # predicted measurements
        z_hat = self.h.evaluate_h_func(Psi, self.x_pred[0,0], self.x_pred[1,0], self.x_pred[2,0], self.x_pred[3,0])
        
        # TEST H nad h
        #h0 = self.h.evaluate_h_func(Psi, self.x[0,0], self.x[1,0], self.x[2,0], self.x[3,0])
        #H0 = self.h.evaluate_dh_func(Psi, self.x[0,0], self.x[1,0], self.x[2,0], self.x[3,0])
        #print("h': ", z_hat)
        #print("dx: ", self.x_pred - self.x[0,0])
        #print("h0 + H0 dx: ", h0 + H0 @ (self.x_pred - self.x))
        #print("linearization error= ", h0 + H0 @ (self.x_pred - self.x) - z_hat)

        # innovation
        z = np.array([[z[0]], [z[1]], [z[2]], [z[3]]])
        self.v = z - z_hat

        # innovation covariance
        S = H @ self.Sigma_pred @ H.T + self.R  

        # filter gain
        K = self.Sigma_pred @ H.T @ np.linalg.inv(S)

        # correct the predicted state statistics
        self.x = self.x_pred + K @ self.v
        self.x[0, 0] = warpToOne(self.x[0, 0])

        I = np.eye(np.shape(self.x)[0])
        self.Sigma = (I - K @ H) @ self.Sigma_pred

def process_model(x):
    dt = 0.01 # data sampling rate: 100 Hz
    A = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    return A @ x

def measurement_data_cov(plot = False):
    global_thigh_angle_Y_pred = model_prediction(m_model.models[0], Psi[0], phases, phase_dots, step_lengths, ramps)
    force_z_ankle_pred = model_prediction(m_model.models[1], Psi[1], phases, phase_dots, step_lengths, ramps)
    force_x_ankle_pred = model_prediction(m_model.models[2], Psi[2], phases, phase_dots, step_lengths, ramps)
    moment_y_ankle_pred = model_prediction(m_model.models[3], Psi[3], phases, phase_dots, step_lengths, ramps)
    err_gthY = global_thigh_angle_Y - global_thigh_angle_Y_pred
    err_fz = force_z_ankle - force_z_ankle_pred
    err_fx = force_x_ankle - force_x_ankle_pred
    err_my = moment_y_ankle - moment_y_ankle_pred
    err = np.stack((err_gthY, err_fz, err_fx, err_my))
    R_data = np.cov(err)
    
    if plot:
        plt.figure('Measurement Data vs. Prediction')
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
    return R_data

if __name__ == '__main__':
    dt = 0.01 # data sampling rate: 100 Hz
    A = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    
    # build the system
    sys = myStruct()
    sys.f = process_model
    sys.A = A
    
    sys.h = m_model
    sys.Q = 1 * np.diag([1e-14, 1e-10, 1e-14, 1e-14]) # process model noise covariance
    #sys.R = measurement_data_cov()
    sys.R = np.diag([10, 1400, 350, 2100]) # measurement noise covariance

    # initialize the state
    init = myStruct()
    init.x = np.array([[phases[0]], [phase_dots[0]], [step_lengths[0]], [ramps[0]]])
    init.Sigma = 1e-14 * np.diag([1, 1, 1, 1])

    ekf = extended_kalman_filter(sys, init)

    z = np.array([[global_thigh_angle_Y],\
                  [force_z_ankle], \
                  [force_x_ankle],\
                  [moment_y_ankle]])
    z = np.squeeze(z)

    x = []  # state
    x.append(init.x)

    v = []

    for i in range(np.shape(z)[1]):
        ekf.prediction()
        ekf.correction(z[:, i])
        x.append(ekf.x)

    x = np.array(x).squeeze()
    print(np.shape(x))

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
    plt.ylim(phase_dots.min()-0.2, phase_dots.max() +0.2)

    plt.subplot(413)
    plt.plot(step_lengths)
    plt.plot(x[:, 2], '--')
    plt.ylim(step_lengths.min()-0.02, step_lengths.max() +0.02)
    plt.ylabel('step length')

    plt.subplot(414)
    plt.plot(ramps)
    plt.plot(x[:, 3], '--')
    plt.ylabel('ramp')
    plt.ylim(ramps.min()-0.5, ramps.max() + 0.5)
    plt.show()
    

    
