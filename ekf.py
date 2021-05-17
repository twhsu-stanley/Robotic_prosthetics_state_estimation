import numpy as np
from model_framework import *
from continuous_data import *
from model_fit import *


def warpToOne(phase):
    phase_wrap = np.remainder(phase, 1)
    while np.abs(phase_wrap) > 1:
        phase_wrap = phase_wrap - np.sign(phase_wrap)
    return phase_wrap

def wrapTo2pi(ang):
    while ang > 2*np.pi:
        ang -= 2*np.pi
    #while ang < 0:
    #    ang += 2*np.pi
    return ang


def phase_error(phase_est, phase_truth):
    # measure error between estimated and ground-truth phase
    if abs(phase_est - phase_truth) < 0.5:
        return abs(phase_est - phase_truth)
    else:
        return 1 - abs(phase_est - phase_truth)
    
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

    def correction(self, z, Psi, arctan2 = False):
        # EKF correction step
        # Inputs:
        #   z:  measurement

        # evaluate measurement Jacobian at current operating point
        H = self.h.evaluate_dh_func(Psi, self.x_pred[0,0], self.x_pred[1,0], self.x_pred[2,0], self.x_pred[3,0])

        # predicted measurements
        self.z_hat = self.h.evaluate_h_func(Psi, self.x_pred[0,0], self.x_pred[1,0], self.x_pred[2,0], self.x_pred[3,0])

        if arctan2:
            H[-1, 0] += 2*np.pi
            self.z_hat[-1] += self.x_pred[0,0] * 2 * np.pi
            # wrap to 2pi
            self.z_hat[-1] = wrapTo2pi(self.z_hat[-1])
                    
        # innovation
        z = np.array([z]).T
        self.v = z - self.z_hat
        
        if arctan2:
            # wrap to pi
            self.v[-1] = np.arctan2(np.sin(self.v[-1]), np.cos(self.v[-1])) 

        # Adjust R dynamically according to errors
        R = self.R
        #MD = np.sqrt(self.v.T @ np.linalg.inv(self.R) @ self.v) # Mahalanobis distance
        #if MD > np.sqrt(26.13): # 8-DOF Chi-square test
            # scale R of thigh angle vel
            #U = np.diag([1, 1, 1, 1, 1/2, 1/2, 1/2, 1])
            #R = U @ R @ U.T

        # innovation covariance
        S = H @ self.Sigma_pred @ H.T + R

        # filter gain
        K = self.Sigma_pred @ H.T @ np.linalg.inv(S)

        # correct the predicted state statistics
        self.x = self.x_pred + K @ self.v
        self.x[0, 0] = warpToOne(self.x[0, 0])

        I = np.eye(np.shape(self.x)[0])
        self.Sigma = (I - K @ H) @ self.Sigma_pred
        