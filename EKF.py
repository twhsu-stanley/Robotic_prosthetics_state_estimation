import numpy as np
from model_framework import *

# Process model for the EKF
def A(dt):
    return np.array([[1, dt, 0], [0, 1, 0], [0, 0, 1]])

def process_model(x, dt):
    return A(dt) @ x.T

## Load control model & coefficients (for OSL implementation)
c_model = model_loader('Control_model.pickle')
with open('Psi_3states/Psi_kneeAngles', 'rb') as file:#_withoutNan
    Psi_knee = pickle.load(file)
with open('Psi_3states/Psi_ankleAngles', 'rb') as file:
    Psi_ankle = pickle.load(file)

## Load model coefficients
def load_Psi():
    with open('Psi_3states/Psi_globalThighAngles', 'rb') as file:
        Psi_globalThighAngles = pickle.load(file)
    with open('Psi_3states/Psi_globalThighVelocities', 'rb') as file:
        Psi_globalThighVelocities = pickle.load(file)
    with open('Psi_3states/Psi_atan2ss', 'rb') as file:
        Psi_atan2 = pickle.load(file)
    """
    with open('Psi/Psi_globalThighAngles_NSL_B10_const.pickle', 'rb') as file:
        Psi_globalThighAngles = pickle.load(file)
        
    with open('Psi/Psi_globalThighVelocities_NSL_B10_const.pickle', 'rb') as file:
        Psi_globalThighVelocities = pickle.load(file)
        
    #with open('Psi_incExp/Psi_ankleMoment_NSL_B33.pickle', 'rb') as file:
    #    Psi_ankleMoment = pickle.load(file)
        
    #with open('Psi_incExp/Psi_tibiaForce_NSL_B33.pickle', 'rb') as file:
    #    Psi_tibiaForce = pickle.load(file)
        
    with open('Psi/Psi_atan2ss_NSL.pickle', 'rb') as file:
        Psi_atan2 = pickle.load(file)
    """
    Psi = {'globalThighAngles': Psi_globalThighAngles, 'globalThighVelocities': Psi_globalThighVelocities,
           'atan2': Psi_atan2}
    return Psi

def warpToOne(phase):
    phase_wrap = np.remainder(phase, 1)
    while np.abs(phase_wrap) > 1:
        phase_wrap = phase_wrap - np.sign(phase_wrap)
    return phase_wrap

def wrapTo2pi(ang):
    ang = ang % (2*np.pi)
    return ang

def phase_error(phase_est, phase_truth):
    if len(phase_est) == len(phase_truth):
        data_len = len(phase_est)
    else:
        exit("Error in phase_error(phase_est, phase_truth): lengths of input data did not match.")
    phase_error = np.zeros(data_len)
    for i in range(data_len):
        # measure error between estimated and ground-truth phase
        if abs(phase_est[i] - phase_truth[i]) < 0.5:
            #return abs(phase_est - phase_truth)
            phase_error[i] = abs(phase_est[i] - phase_truth[i])
        else:
            #return 1 - abs(phase_est - phase_truth)
            phase_error[i] = 1 - abs(phase_est[i] - phase_truth[i])
    return phase_error

def joints_control(phases, phase_dots, step_lengths):
    joint_angles = c_model.evaluate_h_func([Psi_knee, Psi_ankle], phases, phase_dots, step_lengths)
    return joint_angles
    
class myStruct:
    pass

class extended_kalman_filter:
    def __init__(self, system, init):
        # Constructor
        # Inputs:
        #   system: system and noise models
        #   init:   initial state mean and covariance
        self.f = system.f  # process model
        self.A = system.A  # system matrix Jacobian
        self.Q = system.Q
        self.Q_init = np.copy(system.Q)

        self.h = system.h  # measurement model
        self.Psi = system.Psi
        self.R = system.R  # measurement noise covariance

        self.saturation = system.saturation
        self.saturation_range = system.saturation_range
        
        self.adapt_Q = system.adapt_Q
        fc = 0.2
        self.a = 2 * np.pi * 0.01 * fc / (2 * np.pi * 0.01 * fc + 1) # forgetting factor for averaging

        self.x = init.x  # state mean
        self.Sigma = init.Sigma  # state covariance

        self.MD = 0

    def prediction(self, dt):
        # EKF propagation (prediction) step
        self.x = self.f(self.x, dt)  # predicted state
        self.x[0] = warpToOne(self.x[0]) # wrap to [0,1)
        self.Sigma = self.A(dt) @ self.Sigma @ self.A(dt).T + self.Q  # predicted state covariance
        if self.saturation == True:
            self.state_saturation(self.saturation_range)

    def correction(self, z, using_atan2 = False):
        # EKF correction step
        # Inputs:
        #   z:  measurement

        # evaluate measurement Jacobian at current operating point
        H = self.h.evaluate_dh_func(self.Psi, self.x[0], self.x[1], self.x[2])
        
        # predicted measurements
        self.z_hat = self.h.evaluate_h_func(self.Psi, self.x[0], self.x[1], self.x[2])[:,0]

        if using_atan2:
            H[2, 0] += 2*np.pi
            self.z_hat[2] += self.x[0] * 2 * np.pi
            #self.z_hat[2] = wrapTo2pi(self.z_hat[2])
        
        # innovation
        self.v = z - self.z_hat
        if using_atan2:
            # wrap to pi
            self.v[2] = np.arctan2(np.sin(self.v[2]), np.cos(self.v[2]))

        # innovation covariance
        S = H @ self.Sigma @ H.T + self.R

        # filter gain
        K = self.Sigma @ H.T @ np.linalg.inv(S)

        # correct the predicted state statistics
        self.x = self.x + K @ self.v
        self.x[0] = warpToOne(self.x[0])
        self.Sigma = (np.eye(3) - K @ H) @ self.Sigma

        d = K @ self.v[:,None]
        self.MD = np.sqrt(d.T @ np.linalg.pinv(self.Q) @ d)[0,0]
        if self.adapt_Q == True:
            self.Q = self.a * d @ d.T + (1 - self.a) * self.Q
            #r =  z - self.h.evaluate_h_func(self.Psi, self.x[0], self.x[1], self.x[2])[:,0]
            #self.R = self.a * self.R + (1-self.a) * (r[:,None] @ r[:,None].T + H @ self.Sigma @ H.T)        
        
        if self.saturation == True:
            self.state_saturation(self.saturation_range)

        # Compute MD using residuals
        """
        self.z_pred = self.h.evaluate_h_func(Psi, self.x[0,0], self.x[1,0], self.x[2,0], self.x[3,0])
        #if direct_ramp != False:
        #    self.z_pred = np.vstack((self.z_pred, np.array([self.x[3,0]])))
        if using_atan2:
            self.z_pred[2] += self.x[0,0] * 2 * np.pi
            self.z_pred[2] = wrapTo2pi(self.z_pred[2])
        residual = z - self.z_pred
        if using_atan2:
            residual[2] = np.arctan2(np.sin(residual[2]), np.cos(residual[2]))
        #self.MD_residual = 0.2 * np.sqrt(residual.T @ np.linalg.inv(self.R) @ residual) + (1-0.2) * np.copy(self.MD_residual)
        self.MD_residual = np.sqrt(residual[2] * 1/self.R[2,2] * residual[2])
        """

    def state_saturation(self, saturation_range):
        phase_dots_max = saturation_range[0]
        phase_dots_min = saturation_range[1]
        step_lengths_max = saturation_range[2]
        step_lengths_min = saturation_range[3]
        
        if self.x[1] > phase_dots_max:
            self.x[1] = phase_dots_max
        elif self.x[1] < phase_dots_min:
            self.x[1] = phase_dots_min

        if self.x[2] > step_lengths_max:
            self.x[2] = step_lengths_max
        elif self.x[2] < step_lengths_min:
            self.x[2] = step_lengths_min
    