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
    ang = ang % (2*np.pi)
    #while ang > 2*np.pi:
    #    ang -= 2*np.pi
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
        
        # state saturation
        #self.state_saturation()

    def correction(self, z, Psi, arctan2 = False):
        # EKF correction step
        # Inputs:
        #   z:  measurement

        # evaluate measurement Jacobian at current operating point
        H = self.h.evaluate_dh_func(Psi, self.x_pred[0,0], self.x_pred[1,0], self.x_pred[2,0], self.x_pred[3,0])

        # predicted measurements
        self.z_hat = self.h.evaluate_h_func(Psi, self.x_pred[0,0], self.x_pred[1,0], self.x_pred[2,0], self.x_pred[3,0])

        ### Jacobian test#########################################################
        #print("HPH=",  H @ self.Sigma_pred @ H.T)
        #print("R=", self.R)
        #z2 = self.h.evaluate_h_func(Psi, self.x_pred[0,0]-0.01, self.x_pred[1,0]-0.01, self.x_pred[2,0]+0.01, self.x_pred[3,0]-0.01)
        #print(z2 - self.z_hat)
        #print(H @ np.array([[-0.01], [-0.01], [0.01], [-0.01]]))
        ###########################################################################

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

        R = self.R
        lost = False
        # Detect kidnapping event
        self.MD = np.sqrt(self.v.T @ np.linalg.inv(self.R) @ self.v) # Mahalanobis distance
        if self.MD > np.sqrt(22.458): # 6-DOF Chi-square test np.sqrt(22.458)
            lost = True
            # scale R of thigh angle vel
            #U = np.diag([1, 1, 1, 1, 1, 1/2])
            #R = U @ R @ U.T
            #print("kd!: ", MD)
            self.Sigma_pred += np.diag([2e-5, 2e-4, 4e-3, 4])

        # innovation covariance
        S = H @ self.Sigma_pred @ H.T + R
        # check singularity/invertibility/ of S

        # filter gain
        K = self.Sigma_pred @ H.T @ np.linalg.inv(S)

        # correct the predicted state statistics
        self.x = self.x_pred + K @ self.v
        self.x[0, 0] = warpToOne(self.x[0, 0])

        # set phase according to atan2
        #if lost and arctan2:
            #print(K @ self.v)
            #self.x[0, 0] = z[-1] / (2 * np.pi)

        I = np.eye(np.shape(self.x)[0])
        self.Sigma = (I - K @ H) @ self.Sigma_pred
        
    def state_saturation(self, saturation_range):
        phase_dots_max = saturation_range[0]
        phase_dots_min = saturation_range[1]
        step_lengths_max = saturation_range[2]
        step_lengths_min = saturation_range[3]
        #if self.x_pred[0, 0] > 1:
        #    self.x_pred[0, 0] = 1
        #elif self.x_pred[0, 0] < 0:
        #    self.x_pred[0, 0] = 0
        
        if self.x_pred[1, 0] > phase_dots_max:
            self.x_pred[1, 0] = phase_dots_max
        elif self.x_pred[1, 0] < phase_dots_min:
            self.x_pred[1, 0] = phase_dots_min

        if self.x_pred[2, 0] > step_lengths_max:
            self.x_pred[2, 0] = step_lengths_max
        elif self.x_pred[2, 0] < step_lengths_min:
            self.x_pred[2, 0] = step_lengths_min

        if self.x_pred[3, 0] > 10:
            self.x_pred[3, 0] = 10
        elif self.x_pred[3, 0] < -10:
            self.x_pred[3, 0] = -10