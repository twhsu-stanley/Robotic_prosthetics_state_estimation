import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from cvxopt.modeling import variable, max
import time
import pickle

a = np.array([0,-2,3,5])
c =  np.array([0,4,27,15])
b = np.array([1,5,-9,10])
print(abs(a+c)/abs(b))