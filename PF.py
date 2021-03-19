import numpy as np
from numpy import matlib
from numpy.random import randn, rand
from scipy.stats import multivariate_normal
from model_framework import *

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
    def __init__(self):
        self.x = [] # particles
        self.w = [] # associated weights

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
        self.n = init.n  # number of particles

        self.p = myStruct()  # particles
        self.mu = init.mu # particles mean
        self.Sigma = init.Sigma # particles covariance

        wu = 1 / self.n  # uniform weights
        L_init = np.linalg.cholesky(init.Sigma)
        for i in range(self.n):
            self.p.x.append(np.dot(L_init, randn(len(init.mu), 1)) + init.mu)
            self.p.w.append(wu)
        self.p.x = np.array(self.p.x).reshape(-1, len(init.mu))
        self.p.w = np.array(self.p.w).reshape(-1, 1)

    def sample_motion(self, dt):
        for i in range(self.n):
            # process noise
            pn = np.dot(self.LQ, randn(4, 1))
            # propagate the particles
            self.p.x[i, :] = self.f(np.array([self.p.x[i, :]]).T, dt, pn).reshape(-1)

    def importance_measurement(self, z, Psi):
        # Inputs:
        #   z: measurement
        z = np.array([z]).T
        w = np.zeros([self.n, 1])  # importance weights
        for i in range(self.n):
            z_hat = self.h.evaluate_h_func(Psi, warpToOne(self.p.x[i,0]), self.p.x[i,1], self.p.x[i,2], self.p.x[i,3])
            v = z - z_hat
            w[i] = multivariate_normal.pdf(v.reshape(-1), np.array([0, 0, 0, 0]), self.R)

        # update and normalize weights
        self.p.w = np.multiply(self.p.w, w)  # since we used motion model to sample
        self.p.w = self.p.w / np.sum(self.p.w)
        # compute effective number of particles
        self.Neff = 1 / np.sum(np.power(self.p.w, 2))  # effective number of particles
        if self.Neff < self.n / 5:
            self.resampling()
        
        # compute mean and covariance of estimate
        self.mean_cov()

    def resampling(self):
        #print("resampling!")
        # low variance resampling
        W = np.cumsum(self.p.w)
        r = rand(1) / self.n
        # r = 0.5 / self.n
        j = 1
        for i in range(self.n):
            u = r + (i - 1) / self.n
            while u > W[j]:
                j = j + 1
            self.p.x[i, :] = self.p.x[j, :]
            self.p.w[i] = 1 / self.n

    def mean_cov(self):
        wtot = np.sum(self.p.w)
        if wtot > 0:
            self.mu[0] = np.sum(self.p.x[:, 0] * self.p.w.reshape(-1)) / wtot
            self.mu[1] = np.sum(self.p.x[:, 1] * self.p.w.reshape(-1)) / wtot
            self.mu[2] = np.sum(self.p.x[:, 2] * self.p.w.reshape(-1)) / wtot
            self.mu[3] = np.sum(self.p.x[:, 3] * self.p.w.reshape(-1)) / wtot
            
            sum = 0
            for i in range(self.n):
                dev = self.p.x[i, :].T - self.mu
                sum += dev * dev.T * self.p.w[i]
            self.Sigma = sum / wtot

            self.mu[0] = warpToOne(self.mu[0])

        else:
            print('\033[91mWarning: Total weight is zero or nan!\033[0m')
            self.mu[0] = np.nan
            self.mu[1] = np.nan
            self.mu[2] = np.nan
            self.mu[3] = np.nan

    def kidnap(self, state_kidnap):
        delta = state_kidnap - self.mu
        self.p.x = self.p.x + matlib.repmat(delta.T, self.n, 1)
        print("kidnap delta = ", delta)





