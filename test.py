import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import time
import pickle
from continuous_data import *
from basis_model_fitting import measurement_noise_covariance

sensors = ['globalThighAngles', 'globalThighVelocities', 'atan2']
R = measurement_noise_covariance(*sensors)
print("R = \n", R)

# Dictionary of the sensors
sensors_dict = {'global_thigh_angle': 0, 'force_z_ankle': 1, 'force_x_ankle': 2,
                'moment_y_ankle': 3, 'global_thigh_angle_vel': 4, 'atan2': 5}

# Determine which sensors to be used
sensors = ['global_thigh_angle', 'global_thigh_angle_vel', 'atan2']
sensor_id = [sensors_dict[key] for key in sensors]

with open('R.pickle', 'rb') as file:
    R = pickle.load(file)
print("R = \n", R['Generic'][np.ix_(sensor_id, sensor_id)])
