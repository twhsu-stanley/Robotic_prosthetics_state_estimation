import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import time
import pickle
from continuous_data import *
from basis_model_fitting import measurement_noise_covariance

a = np.array([-12,22,-13,4,-5,-6,-27])
print(abs(a)>13)
print(np.all(abs(a)<13))
