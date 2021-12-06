import numpy as np

def unscented_points(mean, cov, alpha = 1, beta = 2, kappa = 0, alg = 'chol'):
    ## Generate unscented points
    dim = cov.shape[0]
    lam = alpha * alpha * (dim + kappa) - dim
    
    # determine the algorithm for computing matrix sqrt
    if alg == "chol":
        L = np.linalg.cholesky(cov)
    elif alg == "svd":
        u, s, v = np.linalg.svd(cov)
        L = np.dot(u, np.sqrt(np.diag(s)))
    
    pts = np.zeros((2 * dim + 1, dim))
    pts[0, :] = mean
    for i in range(1, dim+1):        
        pts[i, :] = mean + np.sqrt(dim + lam) * L[:, i-1]        
        pts[i+dim,:] = mean - np.sqrt(dim + lam) * L[:, i-1]
    W0m = lam / (dim + lam)
    W0c = lam / (dim + lam) + (1 - alpha**2 + beta)
    Wim = 1/ 2 / (dim + lam)
    Wic = 1/ 2 / (dim + lam)
    return pts, (W0m, Wim, W0c, Wic)

def warpToOne(phase):
    phase_wrap = np.remainder(phase, 1)
    while np.abs(phase_wrap) > 1:
        phase_wrap = phase_wrap - np.sign(phase_wrap)
    return phase_wrap

class myStruct:
    pass

class unscented_kalman_filter:
    def __init__(self, system, init):
        self.f = system.f  # process model
        self.Q = system.Q  # input noise covariance
        self.h = system.h  # measurement model
        self.R = system.R  # measurement noise covariance
        
        self.x = init.x  # state vector
        self.Sigma = init.Sigma  # state covariance
        self.dim = self.Sigma.shape[0]
        
        self.alpha = 1
        self.beta = 2
        self.kappa = 0
        self.alg = 'chol'

    def prediction(self, dt):
        ## UKF propagation (prediction) step
        #  dt: time step

        pts, (W0m, Wim, W0c, Wic) = unscented_points(self.x, self.Sigma, self.alpha, self.beta, self.kappa, self.alg)
        
        # transform the unscented points using the process model
        mean = np.zeros(np.shape(self.x))
        self.x_pts_p = np.array(np.zeros((2 * self.dim + 1, self.dim)))
        for i in range(2 * self.dim + 1):
            pp = self.f(pts[i, :], dt)
            if i == 0:
                mean += pp * W0m
            else:
                mean += pp * Wim
            self.x_pts_p[i,:] = pp
        self.x = mean  # predicted state
        self.x[0] = warpToOne(self.x[0]) # wrap to be between 0 and 1

        cov = np.zeros(np.shape(self.Sigma))
        for i in range(2 * self.dim + 1):
            x_temp = (self.x_pts_p[i,:] - self.x).reshape(self.dim,1)
            if x_temp[0] > 0.5:
                x_temp[0] = x_temp[0]-1
            elif x_temp[0] < -0.5:
                x_temp[0] = 1+x_temp[0]
                
            if i == 0:
                cov += x_temp @ x_temp.T * W0c
            else:
                cov += x_temp @ x_temp.T * Wic
        self.Sigma = cov + self.Q  # predicted state covariance

    def correction(self, z, Psi, using_atan2 = False):
        ## UKF correction step
        #  z:  measurement
        #z = np.array([z])

        pts, (W0m, Wim, W0c, Wic) = unscented_points(self.x, self.Sigma, self.alpha, self.beta, self.kappa, self.alg)
        
        # transform the unscented points using the measurement model
        mean = np.zeros(np.shape(z))
        self.z_pts_p = np.array(np.zeros((2 * self.dim + 1, 3)))
        for i in range(2 * self.dim + 1):
            pts[i, 0] = warpToOne(pts[i, 0])
            pp = self.h.evaluate_h_func(Psi, pts[i, 0], pts[i, 1], pts[i, 2], pts[i, 3])[:,0]
            if i == 0:
                mean += pp * W0m
            else:
                mean += pp * Wim
            self.z_pts_p[i,:] = pp
        self.z_hat = mean  # predicted measurement
        
        if using_atan2:
            self.z_hat[2] += self.x[0] * 2 * np.pi

        cov = np.zeros((np.shape(z)[0], np.shape(z)[0]))
        cov_xz = np.zeros((np.shape(self.x)[0], np.shape(z)[0]))
        for i in range(2 * self.dim + 1):
            z_temp = (self.z_pts_p[i,:] - self.z_hat).reshape(np.shape(z)[0],1)
            x_temp = (self.x_pts_p[i,:] - self.x).reshape(self.dim,1)
            if i == 0:
                cov += z_temp @ z_temp.T * W0c
                cov_xz += x_temp @ z_temp.T * W0c
            else:
                cov += z_temp @ z_temp.T * Wic
                cov_xz += x_temp @ z_temp.T * Wic
        self.S = cov + self.R  # innovation covariance
        self.Cov_xz = cov_xz  # state-measurement cross covariance

        # filter gain
        K = self.Cov_xz @ np.linalg.inv(self.S) # Kalman (filter) gain

        # correct the predicted state statistics
        v = z - self.z_hat
        if using_atan2:
            # wrap to pi
            v[2] = np.arctan2(np.sin(v[2]), np.cos(v[2]))
        self.x = self.x + K @ v
        self.x[0] = warpToOne(self.x[0])
        self.Sigma = self.Sigma - K @ self.S @ K.T

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
