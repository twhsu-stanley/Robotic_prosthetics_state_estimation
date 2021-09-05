import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import time
import pickle
from continuous_data import *
from basis_model_fitting import measurement_noise_covariance

ramp_angles = np.array([-10, -7.5, -5, -2.5, 0, 2.5, 5, 7.5, 10])
r, = np.where(ramp_angles == -5)
print(r[0])