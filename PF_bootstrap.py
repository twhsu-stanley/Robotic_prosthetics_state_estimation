import numpy as np
from numpy import matlib
from numpy.random import randn, rand
from scipy.stats import multivariate_normal, gaussian_kde
import matplotlib.pyplot as plt
from model_framework import *

# Process model for the EKF
def A(dt):
    return np.array([[1, dt, 0], [0, 1, 0], [0, 0, 1]])

def process_model(x, dt):
    return A(dt) @ x.T

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

class particles():
    def __init__(self, n):
        self.x = np.zeros((n, 3)) # particles
        self.w = np.zeros(n) # associated weights

class bootstrap_particle_filter:
    def __init__(self, system, init):
        # Constructor
        # Input:
        #   system: system and noise models
        #   init:   initialization parameters
        self.f = system.f  # process model
        self.Q = system.Q  # process model noise covariance
        self.LQ = np.linalg.cholesky(self.Q)  # Cholesky factor of Q
        self.h = system.h  # measurement model
        self.Psi = system.Psi # measurement model parameters
        self.R = system.R  # measurement noise covariance
        
        self.n = init.n  # dynamic number of particles
        self.p = particles(self.n)  # particles
        self.mu = init.mu # prior mean
        self.Sigma = init.Sigma # prior covariance
        L_init = np.linalg.cholesky(init.Sigma)
        self.std = np.zeros(3)

        wu = 1 / self.n  # uniform weights
        for i in range(self.n):
            self.p.x[i, :] = (np.dot(L_init, randn(3)) + self.mu)
            self.p.w[i] = wu  # uniform weights
        xspace = np.linspace(0, 1,100)
        yspace = np.linspace(0, 2,100)
        XX, YY = np.meshgrid(xspace, yspace)
        fig, axs = plt.subplots(1,1)
        positions = np.vstack([XX.ravel(), YY.ravel()])
        values = np.vstack([self.p.x[:, 0], self.p.x[:, 1]])
        kernel = gaussian_kde(values) # kernel density estimate to get contours
        f = np.reshape(kernel(positions).T, XX.shape)
        axs.contour(XX, YY, f,)
        axs.plot(self.p.x[:, 0], self.p.x[:, 1], 'x', color='grey', alpha=0.1)
        axs.set_xlabel("x1", fontsize=14)
        axs.set_ylabel("x2", fontsize=14)
        plt.show()

    def particles_propagation(self, dt):
        for i in range(self.n):
            self.p.x[i, :] = self.f(self.p.x[i, :], dt) + np.dot(self.LQ, randn(3))
            if self.p.x[i, 1] > 2:
                self.p.x[i, 1] = 2
            elif self.p.x[i, 1] < 0:
                self.p.x[i, 1] = 0    
            
            if self.p.x[i, 2] > 2:
                self.p.x[i, 2] = 2
            elif self.p.x[i, 2] < 0:
                self.p.x[i, 2] = 0
        
    def importance_measurement(self, z, using_atan2 = True):
        # Inputs:
        #   z: measurement
        for i in range(self.n):
            z_hat = self.h.evaluate_h_func(self.Psi, warpToOne(self.p.x[i,0]), self.p.x[i,1], self.p.x[i,2])[:,0]
            if using_atan2 == True:
                z_hat[2] += self.mu[0] * 2 * np.pi
            innov = z - z_hat
            if using_atan2 == True:
                innov[2] = np.arctan2(np.sin(innov[2]), np.cos(innov[2]))# wrap to pi
            v = multivariate_normal.pdf(innov, mean = [0, 0, 0], cov = self.R)
            self.p.w[i] = self.p.w[i] * v # update weights
        self.p.w = self.p.w / np.sum(self.p.w) # normalize weights

        self.Neff = 1 / np.sum(np.power(self.p.w, 2))  # effective number of particles
        if self.Neff < self.n / 5:
            self.resampling(mode = "low_variance")
            #self.resample()
        
    def resampling(self, mode):
        if mode == "uniform":
            # generate uniformly distributed state
            self.p.x = np.zeros((self.n, 3))
            self.p.w = np.zeros(self.n)
            for m in range(self.n):
                self.p.x[m, :] = np.array([np.random.uniform(0, 1), np.random.uniform(0, 2), np.random.uniform(0, 2)])
                self.p.w[m] = 1 / self.n

        elif mode == "low_variance":
            # low variance resampling
            W = np.cumsum(self.p.w)
            r = rand(1) / self.n
            j = 0
            x_temp = self.p.x
            self.p.x = np.zeros((self.n, 3)) 
            self.p.w = np.zeros(self.n)
            for m in range(self.n):
                u = r + m / self.n
                while u > W[j]:
                    j = j + 1
                self.p.x[m, :] = x_temp[j, :]
                self.p.w[m] = 1 / self.n
        
        else:
            rr = np.arange(self.n) # get an ordered set of numbers from 0 to N-1
            # Randomly choose the integers (with replacement) between 0 to N-1 with probabilities given by the weights
            samp_inds = np.random.choice(rr, self.n, p = self.p.w) 
            # subselect the samples chosen
            self.p.x = self.p.x[samp_inds, :]
            # return uniform weights
            self.p.w = np.ones(self.n) / self.n

    def mean_std(self):
        # compute mean
        self.mu[0] = np.sum(self.p.x[:, 0] * self.p.w)
        self.mu[1] = np.sum(self.p.x[:, 1] * self.p.w)
        self.mu[2] = np.sum(self.p.x[:, 2] * self.p.w)
        if self.mu[0] > 1:
            self.mu[0] = self.mu[0] - 1
            self.p.x[:,0] = self.p.x[:,0] - 1
        elif self.mu[0] < 0:
            self.mu[0] = self.mu[0] + 1
            self.p.x[:,0] = self.p.x[:,0] + 1
        
        # compute std
        self.std[0] = np.sqrt(np.sum(self.p.w * (self.p.x[:, 0] - self.mu[0])**2))
        self.std[1] = np.sqrt(np.sum(self.p.w * (self.p.x[:, 1] - self.mu[1])**2))
        self.std[2] = np.sqrt(np.sum(self.p.w * (self.p.x[:, 2] - self.mu[2])**2))

    def kidnap(self, state_kidnap):
        delta = state_kidnap - self.mu
        self.p.x = self.p.x + matlib.repmat(delta.T, self.n, 1)
    
    def plot_2d(self):
        ## Plot an empirical distribution
        # Low-variance resampling
        W = np.cumsum(self.p.w)
        r = rand(1) / self.n
        j = 0
        x_temp = self.p.x
        particles = np.zeros((self.n, 3)) 
        for m in range(self.n):
            u = r + m /self.n 
            while u > W[j]:
                j = j + 1
            particles[m, :] = x_temp[j, :]
        
        xspace = np.linspace(0, 1, 100)
        yspace = np.linspace(0, 2, 100)
        XX, YY = np.meshgrid(xspace, yspace)
        fig, axs = plt.subplots(1,1)
        positions = np.vstack([XX.ravel(), YY.ravel()])
        values = np.vstack([particles[:, 0], particles[:, 1]])
        kernel = gaussian_kde(values) # kernel density estimate to get contours
        f = np.reshape(kernel(positions).T, XX.shape)
        axs.contour(XX, YY, f,)
        axs.plot(particles[:, 0], particles[:, 1], 'x', color='grey', alpha=0.1)
        axs.set_xlabel("x1", fontsize=14)
        axs.set_ylabel("x2", fontsize=14)
        return fig, axs
