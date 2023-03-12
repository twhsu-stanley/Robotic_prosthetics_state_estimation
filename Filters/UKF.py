import numpy as np

def unscented_points(mean, cov, alpha = 1, beta = 0, kappa = 0, alg = 'chol'):
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
        self.Psi = system.Psi
        self.R = system.R  # measurement noise covariance
        self.alpha = system.alpha
        self.beta = system.beta
        self.kappa = system.kappa
        self.alg = 'chol'
        self.saturation = system.saturation
        self.saturation_range = system.saturation_range
        self.reset = system.reset

        self.x = init.x  # state vector
        self.Sigma = init.Sigma  # state covariance
        self.dim = self.Sigma.shape[0]

    def prediction(self, dt):
        ## UKF propagation (prediction) step
        #  dt: time step

        pts, (W0m, Wim, W0c, Wic) = unscented_points(self.x, self.Sigma, self.alpha, self.beta, self.kappa, self.alg)
        
        # transform the unscented points using the process model
        mean_x = np.zeros(np.shape(self.x))
        self.x_pts_p = np.array(np.zeros((2 * self.dim + 1, self.dim)))
        for i in range(2 * self.dim + 1):
            pp_x = self.f(pts[i, :], dt)
            if i == 0:
                mean_x += pp_x * W0m
            else:
                mean_x += pp_x * Wim
            self.x_pts_p[i,:] = pp_x
        self.x = mean_x  # predicted state

        cov_xx = np.zeros(np.shape(self.Sigma))
        for i in range(2 * self.dim + 1):
            x_temp = (self.x_pts_p[i,:] - self.x).reshape(self.dim, 1)
            if i == 0:
                cov_xx += x_temp @ x_temp.T * W0c
            else:
                cov_xx += x_temp @ x_temp.T * Wic
        self.Sigma = cov_xx + self.Q # predicted state covariance

        if self.saturation == True:
            self.state_saturation(self.saturation_range)

    def correction(self, z, using_atan2 = False):
        ## UKF correction step
        #  z:  measurement

        pts, (W0m, Wim, W0c, Wic) = unscented_points(self.x, self.Sigma, self.alpha, self.beta, self.kappa, self.alg)
        
        # transform the unscented points using the measurement model
        mean_z = np.zeros(np.shape(z))
        self.z_pts_p = np.array(np.zeros((2 * self.dim + 1, np.shape(z)[0])))
        self.x_pts_p = np.array(np.zeros((2 * self.dim + 1, self.dim)))
        for i in range(2 * self.dim + 1):
            pp_z = self.h.evaluate_h_func(self.Psi, pts[i, 0], pts[i, 1], pts[i, 2])[:,0]
            if i == 0:
                mean_z += pp_z * W0m
            else:
                mean_z += pp_z * Wim
            self.z_pts_p[i,:] = pp_z
            self.x_pts_p[i,:] = pts[i, :]
        self.z_hat = mean_z  # predicted measurement
        
        if using_atan2:
            self.z_hat[2] += self.x[0] * 2 * np.pi

        cov_zz = np.zeros((np.shape(z)[0], np.shape(z)[0]))
        cov_xz = np.zeros((np.shape(self.x)[0], np.shape(z)[0]))
        for i in range(2 * self.dim + 1):
            z_temp = (self.z_pts_p[i,:] - self.z_hat).reshape(np.shape(z)[0],1)
            x_temp = (self.x_pts_p[i,:] - self.x).reshape(self.dim,1)
            if i == 0:
                cov_zz += z_temp @ z_temp.T * W0c
                cov_xz += x_temp @ z_temp.T * W0c
            else:
                cov_zz += z_temp @ z_temp.T * Wic
                cov_xz += x_temp @ z_temp.T * Wic
        self.S = cov_zz + self.R  # innovation covariance
        self.Cov_xz = cov_xz  # state-measurement cross covariance

        # filter gain
        K = self.Cov_xz @ np.linalg.pinv(self.S) # Kalman (filter) gain

        # correct the predicted state statistics
        v = z - self.z_hat
        if using_atan2:
            v[2] = np.arctan2(np.sin(v[2]), np.cos(v[2]))# wrap to pi
        self.x = self.x + K @ v
        self.x[0] = warpToOne(self.x[0])
        self.Sigma = self.Sigma - K @ self.S @ K.T

        self.MD_square = v.T @ np.linalg.pinv(self.S) @ v
        if self.reset == True and self.MD_square > 25:
            self.x = np.array([0.5, 0.8, 1.1]) # mid-stance
            self.Sigma = np.diag([1e-2, 1e-1, 1e-1])

        if self.saturation == True:
            self.state_saturation(self.saturation_range)

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
