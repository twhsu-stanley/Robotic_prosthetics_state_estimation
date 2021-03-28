import numpy as np
from numpy import matlib
from numpy.random import randn, rand
from scipy.stats import multivariate_normal
from model_framework import *
import math

def warpToOne(phase):
    phase_wrap = np.remainder(phase, 1)
    while np.abs(phase_wrap) > 1:
        phase_wrap = phase_wrap - np.sign(phase_wrap)
    return phase_wrap

def phase_error(phase_est, phase_truth):
    # measure error between estimated and ground-truth phase
    if abs(phase_est - phase_truth) < 0.5:
        return abs(phase_est - phase_truth)
    else:
        return 1 - abs(phase_est - phase_truth)

class myStruct():
    def __init__(self, n):
        self.x = np.zeros((n, 4)) # particles
        self.w = np.zeros(n) # associated weights

class particle_filter:
    def __init__(self, system, init):
        # Constructor
        # Input:
        #   system: system and noise models
        #   init:   initialization parameters
        self.f = system.f  # process model
        self.Q = system.Q  # process model noise covariance
        self.LQ = np.linalg.cholesky(self.Q)  # Cholesky factor of Q
        self.h = system.h  # measurement model
        self.R = system.R  # measurement noise covariance
        self.n = init.n  # dynamic number of particles
        self.pn = init.n # number of particles

        self.p = myStruct(self.n)  # particles
        self.mu = init.mu # particles mean
        self.Sigma = init.Sigma # particles covariance

        #wu = 1 / self.n  # uniform weights
        L_init = np.linalg.cholesky(init.Sigma)
        for i in range(self.n):
            self.p.x[i, :] = (np.dot(L_init, randn(4, 1)) + init.mu).T
            self.p.w[i] = 1 / self.n  # uniform weights

    def particles_propagation(self, dt):
        for i in range(self.n):
            pn = np.dot(self.LQ, randn(4, 1)) # process noise
            self.p.x[i, :] = self.f(np.array([self.p.x[i, :]]).T, dt, pn).T #reshape(-1)
        
    def importance_measurement(self, z, Psi):
        # Inputs:
        #   z: measurement
        z = np.array([z]).T
        w = np.zeros(self.n)  # importance weights
        count = 0
        for i in range(self.n):
            z_hat = self.h.evaluate_h_func(Psi, warpToOne(self.p.x[i,0]), self.p.x[i,1], self.p.x[i,2], self.p.x[i,3])
            w[i] = multivariate_normal.pdf((z - z_hat).T, np.array([0, 0, 0, 0]), self.R)
            if w[i] > 1e-15:
                count += 1

        self.p.w = np.multiply(self.p.w, w) # update weights

        self.mean_cov() # compute mean and covariance of estimate

        if count < self.pn / 30: # might be kidnapped
            self.Neff = 0
            self.n = self.pn * 50 # increase number of particles by "100" times
            self.resampling(mode = "uniform")
        else:
            self.p.w = self.p.w / np.sum(self.p.w) # normalize weights
            self.Neff = 1 / np.sum(np.power(self.p.w, 2))  # effective number of particles
            if self.Neff < self.n / 5:
                self.n = self.pn # set number of pparticles to the origin value
                self.resampling(mode = "low_variance")
        
    def resampling(self, mode):
        if mode == "uniform":
            # generate uniformly distributed state
            #print("resamping: uniform")
            mu_temp = np.mean(self.p.x[:, 0])
            self.p.x = np.zeros((self.n, 4))
            self.p.w = np.zeros(self.n)
            for m in range(self.n):
                phase_kidnap = np.random.uniform(0, 1)
                phase_dot_kidnap = np.random.uniform(0, 5)
                step_length_kidnap = np.random.uniform(0, 2)
                ramp_kidnap = np.random.uniform(-45, 45)

                self.p.x[m, :] = np.array([math.floor(mu_temp) + phase_kidnap, phase_dot_kidnap, step_length_kidnap, ramp_kidnap])
                self.p.w[m] = 1 / self.n

        elif mode == "low_variance":
            # low variance resampling
            #print("resamping: low_variance")
            W = np.cumsum(self.p.w)
            r = rand(1) / self.n
            j = 0
            x_temp = self.p.x
            self.p.x = np.zeros((self.n, 4)) 
            self.p.w = np.zeros(self.n)
            for m in range(self.n):
                u = r + m / self.n
                while u > W[j]:
                    j = j + 1
            
                self.p.x[m, :] = x_temp[j, :]
                self.p.w[m] = 1 / self.n

    def mean_cov(self):
        wtot = np.sum(self.p.w)
        if wtot > 0:
            self.mu[0] = np.sum(self.p.x[:, 0] * self.p.w) / wtot
            self.mu[1] = np.sum(self.p.x[:, 1] * self.p.w) / wtot
            self.mu[2] = np.sum(self.p.x[:, 2] * self.p.w) / wtot
            self.mu[3] = np.sum(self.p.x[:, 3] * self.p.w) / wtot
            
            #sum = 0
            #for i in range(self.n):
            #    dev = self.p.x[i, :].T - self.mu
            #   sum += dev * dev.T * self.p.w[i]
            #self.Sigma = sum / wtot

            self.mu[0] = warpToOne(self.mu[0])

        else:
            #print('\033[91mWarning: Total weight is zero or nan!\033[0m')
            self.mu[0] = np.mean(self.p.x[:, 0]) #np.nan
            self.mu[0] = warpToOne(self.mu[0])
            self.mu[1] = np.mean(self.p.x[:, 1]) #np.nan
            self.mu[2] = np.mean(self.p.x[:, 2]) #np.nan
            self.mu[3] = np.mean(self.p.x[:, 3]) #np.nan

    def kidnap(self, state_kidnap):
        delta = state_kidnap - self.mu
        self.p.x = self.p.x + matlib.repmat(delta.T, self.n, 1)
