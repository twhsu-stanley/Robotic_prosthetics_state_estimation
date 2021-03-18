import numpy as np
from numpy.random import randn
from PF import particle_filter
import matplotlib.pyplot as plt
import math

class myStruct:
    pass

def A(dt):
    return np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

def process_model(x, dt, n):
    # dt = 0.01 # data sampling rate: 100 Hz
    # n: additive noise
    return A(dt) @ x.T + n

def measurement_model(x):
    h = np.array([[np.sqrt(x[0] ** 2 + x[1] ** 2)], [math.atan2(x[0], x[1])]])
    return h.reshape([2, 1])

if __name__ == '__main__':

    # measurements
    R = np.diag(np.power([0.05, 0.01], 2))
    # Cholesky factor of covariance for sampling
    L = np.linalg.cholesky(R)
    z = np.zeros([2, len(gt.x)])
    for i in range(len(gt.x)):
        # sample from a zero mean Gaussian with covariance R
        noise = np.dot(L, randn(2, 1)).reshape(-1)
        # noise = np.dot(L, np.array([[0.05], [0.1]])).reshape(-1)
        z[:, i] = np.array([np.sqrt(gt.x[i] ** 2 + gt.y[i] ** 2), math.atan2(gt.x[i], gt.y[i])]) + noise

    # build the system
    sys = myStruct()
    sys.f = process_model
    sys.h = measurement_model
    sys.Q = 1e-1 * np.eye(2)
    sys.R = np.diag(np.power([0.05, 0.01], 2))

    # initialization
    init = myStruct()
    init.n = 100
    init.x = np.zeros([2, 1])
    init.x[0, 0] = z[0, 0] * np.sin(z[1, 0])
    init.x[1, 0] = z[0, 0] * np.cos(z[1, 0])
    init.Sigma = 1 * np.eye(2)

    filter = particle_filter(sys, init)
    x = np.empty([2, np.shape(z)[1]])  # state
    x[:, 0] = [np.nan, np.nan]

    # main loop; iterate over the measurements
    for i in range(1, np.shape(z)[1], 1):
        filter.sample_motion()
        filter.importance_measurement(z[:, i].reshape([2, 1]))
        if filter.Neff < filter.n / 5:
            filter.resampling()
        wtot = np.sum(filter.p.w)
        if wtot > 0:
            a = filter.p.x
            b = filter.p.w
            x[0, i] = np.sum(filter.p.x[:, 0] * filter.p.w.reshape(-1)) / wtot
            x[1, i] = np.sum(filter.p.x[:, 1] * filter.p.w.reshape(-1)) / wtot
        else:
            print('\033[91mWarning: Total weight is zero or nan!\033[0m')
            x[:, i] = [np.nan, np.nan]
        



