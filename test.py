import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import time
import pickle
from continuous_data import *
from basis_model_fitting import measurement_noise_covariance

exit("exit!")
a = np.array([1,2,3,4])
b = np.array([0,4,6,2])
print(abs(a-b))
print(np.all(abs(a-b)<2))