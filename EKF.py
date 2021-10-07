import numpy as np
from model_framework import *

# Process model for the EKF
def A(dt):
    return np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

def process_model(x, dt):
    #dt = 0.01 # data sampling rate: 100 Hz
    return A(dt) @ x

## Load control model & coefficients (for OSL implementation)
c_model = model_loader('Control_model_NSL_B20.pickle')
with open('Psi/Psi_kneeAngles_NSL_B20_const.pickle', 'rb') as file:#_withoutNan
    Psi_knee = pickle.load(file)
with open('Psi/Psi_ankleAngles_NSL_B20_const.pickle', 'rb') as file:
    Psi_ankle = pickle.load(file)

## Load model coefficients
def load_Psi(subject = 'Generic'):
    if subject == 'Generic':
        with open('Psi/Psi_globalThighAngles_NSL_B10_const.pickle', 'rb') as file:
            Psi_globalThighAngles = pickle.load(file)
        
        with open('Psi/Psi_globalThighVelocities_NSL_B10_const.pickle', 'rb') as file:
            Psi_globalThighVelocities = pickle.load(file)
        
        #with open('Psi_incExp/Psi_ankleMoment_NSL_B33.pickle', 'rb') as file:
        #    Psi_ankleMoment = pickle.load(file)
        
        #with open('Psi_incExp/Psi_tibiaForce_NSL_B33.pickle', 'rb') as file:
        #    Psi_tibiaForce = pickle.load(file)
        
        with open('Psi/Psi_atan2_NSL.pickle', 'rb') as file:
            Psi_atan2 = pickle.load(file)

    else:
        print("Subject-specific model is not available at this time.")
        """
        with open('Psi/Psi_thigh_Y.pickle', 'rb') as file:
            p = pickle.load(file)
            Psi_globalThighAngles = p[subject]
        with open('Psi/Psi_force_Z.pickle', 'rb') as file:
            p = pickle.load(file)
            Psi_force_Z = p[subject]
        with open('Psi/Psi_force_X.pickle', 'rb') as file:
            p = pickle.load(file)
            Psi_force_X = p[subject]
        with open('Psi/Psi_moment_Y.pickle', 'rb') as file:
            p = pickle.load(file)
            Psi_ankleMoment = p[subject]
        with open('Psi/Psi_thighVel_2hz.pickle', 'rb') as file:
            p = pickle.load(file)
            Psi_globalThighVelocities = p[subject]
        with open('Psi/Psi_atan2.pickle', 'rb') as file:
            p = pickle.load(file)
            Psi_atan2 = p[subject]
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

def joints_control(phases, phase_dots, step_lengths, ramps):
    joint_angles = c_model.evaluate_h_func([Psi_knee, Psi_ankle], phases, phase_dots, step_lengths, ramps)
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
        self.Q_static = system.Q
        self.Q = self.Q_static    # process model noise covariance

        self.h = system.h  # measurement model
        self.R_static = system.R  # measurement noise covariance
        self.R = self.R_static

        self.x = init.x  # state mean
        self.Sigma = init.Sigma  # state covariance

    def prediction(self, dt):
        # EKF propagation (prediction) step
        self.x = self.f(self.x, dt)  # predicted state
        self.x[0, 0] = warpToOne(self.x[0, 0]) # wrap to be between 0 and 1
        self.Sigma = self.A(dt) @ self.Sigma @ self.A(dt).T + self.Q  # predicted state covariance

    def correction(self, z, Psi, using_atan2 = False, steady_state_walking = False, direct_ramp = False):
        # EKF correction step
        # Inputs:
        #   z:  measurement

        # evaluate measurement Jacobian at current operating point
        H = self.h.evaluate_dh_func(Psi, self.x[0,0], self.x[1,0], self.x[2,0], self.x[3,0])
        
        # predicted measurements
        self.z_hat = self.h.evaluate_h_func(Psi, self.x[0,0], self.x[1,0], self.x[2,0], self.x[3,0])

        ### Jacobian test #########################################################
        #print("HPH=",  H @ self.Sigma @ H.T)
        #print("R=", self.R)
        #z2 = self.h.evaluate_h_func(Psi, self.x[0,0]-0.01, self.x[1,0]-0.01, self.x[2,0]+0.01, self.x[3,0]-0.01)
        #print(z2 - self.z_hat)
        #print(H @ np.array([[-0.01], [-0.01], [0.01], [-0.01]]))
        ###########################################################################

        if direct_ramp != False:
            # add the direct ramp as the last element of the measurement vector
            H = np.vstack((H, np.array([0, 0, 0, 1])))
            self.z_hat = np.vstack((self.z_hat, np.array([self.x[3,0]])))
            z = np.concatenate((z, direct_ramp))

        if using_atan2:
            H[2, 0] += 2*np.pi
            self.z_hat[2] += self.x[0,0] * 2 * np.pi
            # wrap to 2pi
            self.z_hat[2] = wrapTo2pi(self.z_hat[2])
        
        # innovation
        z = np.array([z]).T
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
        self.x[0, 0] = warpToOne(self.x[0, 0])

        # Compute MD using residuals
        """
        z_pred = self.h.evaluate_h_func(Psi, self.x[0,0], self.x[1,0], self.x[2,0], self.x[3,0])
        if direct_ramp != False:
            z_pred = np.vstack((z_pred, np.array([self.x[3,0]])))
        if using_atan2:
            z_pred[2] += self.x[0,0] * 2 * np.pi
            z_pred[2] = wrapTo2pi(z_pred[2])
        self.residual = z - z_pred
        if using_atan2:
            self.residual[2] = np.arctan2(np.sin(self.residual[2]), np.cos(self.residual[2]))
        self.MD_residual = np.sqrt(self.residual.T @ np.linalg.inv(self.R) @ self.residual) # Mahalanobis distance
        """

        """
        if steady_state_walking and self.MD_residual > np.sqrt(18.5):
            #self.Q = self.Q_static + self.Q_static * 0.2
            self.R = np.diag([2, 1, 2, 1]) @ self.R_static @ np.diag([2, 1, 2, 1]).T
        else:
            #self.Q = self.Q_static
            self.R = self.R_static
        """

        # Adaptive Q and R
        #alpha = 0.3
        #self.Q = alpha * self.Q + (1-alpha)*(K @ self.v @ self.v.T @ K.T)
        #self.R = alpha * self.R + (1-alpha)*(self.residual @ self.residual.T + H @ self.Sigma @ H.T)

        I = np.eye(np.shape(self.x)[0])
        self.Sigma = (I - K @ H) @ self.Sigma

    def state_saturation(self, saturation_range):
        phase_dots_max = saturation_range[0]
        phase_dots_min = saturation_range[1]
        step_lengths_max = saturation_range[2]
        step_lengths_min = saturation_range[3]
        
        if self.x[1, 0] > phase_dots_max:
            self.x[1, 0] = phase_dots_max
        elif self.x[1, 0] < phase_dots_min:
            self.x[1, 0] = phase_dots_min

        if self.x[2, 0] > step_lengths_max:
            self.x[2, 0] = step_lengths_max
        elif self.x[2, 0] < step_lengths_min:
            self.x[2, 0] = step_lengths_min

        if self.x[3, 0] > 10:
            self.x[3, 0] = 10
        elif self.x[3, 0] < -10:
            self.x[3, 0] = -10
    