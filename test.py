import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from cvxopt.modeling import variable, max
import time
import pickle

a = np.array([0,-2,3,5])
Q = np.array([[1, 2, 1], [1, 2, 1], [1, 2, 1]])
x = np.array([2,3,4])
y = np.array([2,1,3])
print(x**2)
print(np.dot(Q,x))
print(np.dot(Q, x.T))
print(np.array([x]).T)
print(np.dot(Q, np.array([x]).T))