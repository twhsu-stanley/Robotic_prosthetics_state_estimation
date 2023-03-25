from pickle import FROZENSET
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from mpl_toolkits import mplot3d
from model_framework import *
from wrapping import *
from load_Psi import *

# Determine what sensors to be used
sensors = ['globalThighAngles', 'globalThighVelocities' ,'atan2', 'globalFootAngles']

sensor_id = 3

# Load model
model_name = "Measurement_model"
for s in sensors:
    model_name += ('_' + s)
model_name += ".pickle"
m_model = model_loader(model_name)

Psi = np.array([load_Psi()[key] for key in sensors], dtype = object)

# Load training data
with open(('Gait_training_data_incExp/' + sensors[sensor_id] + '_training_dataset.pickle'), 'rb') as file:
    gait_training_dataset = pickle.load(file)
data_training = gait_training_dataset['training_data']
phase_training = gait_training_dataset['phase']
phase_dot_training = gait_training_dataset['phase_dot']
step_length_training = gait_training_dataset['step_length']
ramp_training = gait_training_dataset['ramp']

"""
with open(('Gait_training_data_R01/' + sensors[sensor_id] + '_walking_training_dataset.pickle'), 'rb') as file:
    gait_training_dataset = pickle.load(file)
data_training = np.vstack((data_training, gait_training_dataset['training_data']))
phase_training = np.vstack((phase_training, gait_training_dataset['phase']))
phase_dot_training = np.vstack((phase_dot_training, gait_training_dataset['phase_dot']))
step_length_training = np.vstack((step_length_training, gait_training_dataset['step_length']))
ramp_training = np.vstack((ramp_training, gait_training_dataset['ramp']))
"""

## A. Visualize Measurement Model w.r.t. phase_dot ===========================================================================
phases = np.linspace(0, 1, num = 50)
phase_dots = np.linspace(0.5, 1.2, num = 50)
step_lengths = 1.3
ramps = 0

measurement = np.zeros((len(phases), len(phase_dots)))

for i in range(len(phases)):
    for j in range(len(phase_dots)):
        z = m_model.evaluate_h_func(Psi, phases[i], phase_dots[j], step_lengths, ramps)
        if sensors[sensor_id] == 'atan2':
            measurement[i,j] = z[sensor_id] + 2*np.pi*phases[i]
            measurement[i,j] = wrapTo2pi(measurement[i,j])
        else:
            measurement[i,j] = z[sensor_id]

fig = plt.figure()
X, Y = np.meshgrid(phases, phase_dots)
ax = plt.axes(projection='3d')
for p in range(np.shape(data_training)[0]):
    if p % 10 == 0:
        ax.plot(phase_training[p,:], phase_dot_training[p,:], data_training[p,:], 'r')
ax.plot_surface(X, Y, measurement.T, alpha = 0.7)
ax.set_xlabel('$\phi$', fontsize = 14)
ax.set_ylabel('$\dot{\phi}$ (1/s)', fontsize = 14)
ax.set_zlabel(sensors[sensor_id], fontsize = 14)

## B. Visualize Measurement Model w.r.t. stpe_length ===========================================================================
phases = np.linspace(0, 1, num = 50)
phase_dots = 0.85
step_lengths = np.linspace(0.8, 2, num = 50)
ramps = 0

measurement = np.zeros((len(phases), len(step_lengths)))

for i in range(len(phases)):
    for j in range(len(step_lengths)):
        z = m_model.evaluate_h_func(Psi, phases[i], phase_dots, step_lengths[j], ramps)
        if sensors[sensor_id] == 'atan2':
            measurement[i,j] = z[sensor_id] + 2*np.pi*phases[i]
            measurement[i,j] = wrapTo2pi(measurement[i,j])
        else:
            measurement[i,j] = z[sensor_id]

fig = plt.figure()
X, Y = np.meshgrid(phases, step_lengths)
ax2 = plt.axes(projection='3d')
for p in range(np.shape(data_training)[0]):
    if p % 10 == 0:
        ax2.plot(phase_training[p,:], step_length_training[p,:], data_training[p,:], 'r')
ax2.plot_surface(X, Y, measurement.T, alpha = 0.7)
ax2.set_xlabel('$\phi$', fontsize = 14)
ax2.set_ylabel('$l$', fontsize = 14)
ax2.set_zlabel(sensors[sensor_id], fontsize = 14)

## C. Visualize Measurement Model w.r.t. ramp angle ===========================================================================
phases = np.linspace(0, 1, num = 50)
phase_dots = 0.85
step_lengths = 1.3 
ramps = np.linspace(-10, 10, num = 50)

measurement = np.zeros((len(phases), len(ramps)))

for i in range(len(phases)):
    for j in range(len(ramps)):
        z = m_model.evaluate_h_func(Psi, phases[i], phase_dots, step_lengths, ramps[j])
        if sensors[sensor_id] == 'atan2':
            measurement[i,j] = z[sensor_id] + 2*np.pi*phases[i]
            measurement[i,j] = wrapTo2pi(measurement[i,j])
        else:
            measurement[i,j] = z[sensor_id]

fig = plt.figure()
X, Y = np.meshgrid(phases, ramps)
ax2 = plt.axes(projection='3d')
for p in range(np.shape(data_training)[0]):
    if p % 10 == 0:
        ax2.plot(phase_training[p,:], ramp_training[p,:], data_training[p,:], 'r')
ax2.plot_surface(X, Y, measurement.T, alpha = 0.7)
ax2.set_xlabel('$\phi$', fontsize = 14)
ax2.set_ylabel('$r$  (deg)', fontsize = 14)
ax2.set_zlabel(sensors[sensor_id], fontsize = 14)
plt.show()