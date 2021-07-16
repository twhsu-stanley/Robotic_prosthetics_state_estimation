import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from mpl_toolkits import mplot3d
from model_framework import *
from model_fit import *
from EKF import wrapTo2pi, load_Psi

"""
# Dictionary of the sensors
sensors_dict = {'global_thigh_angle': 0, 'force_z_ankle': 1, 'force_x_ankle': 2,
                'moment_y_ankle': 3, 'global_thigh_angle_vel': 4, 'atan2': 5}

# Determine which sensors to be used
sensors = ['global_thigh_angle', 'global_thigh_angle_vel', 'atan2']

sensor_id = 1

m_model = model_loader('Measurement_model_3.pickle')
Psi = np.array([load_Psi('Generic')[key] for key in sensors], dtype = object)

## A. Visualize Measurement Model w.r.t. phase_dot ===========================================================================
phases = np.linspace(0, 1, num = 50)
phase_dots = np.linspace(0.6, 1, num = 50)
step_lengths = 1.1
ramps = 0

measurement = np.zeros((len(phases), len(phase_dots)))

for i in range(len(phases)):
    for j in range(len(phase_dots)):
        #z[i,j] = model_prediction(m_model.models[sensor], Psi[sensor], phases[i], phase_dots[j], step_lengths, ramps)
        z = m_model.evaluate_h_func(Psi, phases[i], phase_dots[j], step_lengths, ramps)
        if sensor_id == 7:
            #measurement[i,j] = wrapTo2pi(m[sensor] + 2 * np.pi * phases[i])
            measurement[i,j] = z[sensor_id]
        else:
            measurement[i,j] = z[sensor_id]

fig = plt.figure()
X, Y = np.meshgrid(phases, phase_dots)
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, measurement.T)
ax.set_xlabel('phase')
ax.set_ylabel('phase_dot')
# ==========================================================================================================================

## B. Visualize Measurement Model w.r.t. stpe_length ===========================================================================
phases = np.linspace(0, 1, num = 50)
phase_dots = 0.8
step_lengths = np.linspace(0.5, 1.5, num = 50)
ramps = 0

measurement = np.zeros((len(phases), len(step_lengths)))

for i in range(len(phases)):
    for j in range(len(step_lengths)):
        #z[i,j] = model_prediction(m_model.models[sensor], Psi[sensor], phases[i], phase_dots[j], step_lengths, ramps)
        z = m_model.evaluate_h_func(Psi, phases[i], phase_dots, step_lengths[j], ramps)
        if sensor_id == 7:
            #measurement[i,j] = wrapTo2pi(m[sensor] + 2 * np.pi * phases[i])
            measurement[i,j] = z[sensor_id]
        else:
            measurement[i,j] = z[sensor_id]

fig = plt.figure()
X, Y = np.meshgrid(phases, step_lengths)
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, measurement.T)
ax.set_xlabel('phase')
ax.set_ylabel('step_length')
# ==========================================================================================================================
"""


c_model = model_loader('Control_model.pickle')
with open('New_Psi/Psi_kneeAngles.pickle', 'rb') as file:#_withoutNan
    Psi_knee = pickle.load(file)
with open('New_Psi/Psi_ankleAngles.pickle', 'rb') as file:
    Psi_ankle = pickle.load(file)
## C. Visualize Joint Model w.r.t. step_length ===============================================================================================
phases = np.linspace(0, 1, num = 50)
phase_dots = 0.8
step_lengths = np.linspace(0.9, 1.6, num = 50)
ramps = 0

knee_angle_model = np.zeros((len(phases), len(step_lengths)))
ankle_angle_model = np.zeros((len(phases), len(step_lengths)))

for i in range(len(phases)):
    for j in range(len(step_lengths)):
        joint_angles = c_model.evaluate_h_func([Psi_knee, Psi_ankle], phases[i], phase_dots, step_lengths[j], ramps)
        knee_angle_model[i,j] = joint_angles[0]
        ankle_angle_model[i,j] = joint_angles[1]
        

fig = plt.figure()
X, Y = np.meshgrid(phases, step_lengths)
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, ankle_angle_model.T)
ax.set_xlabel('phase')
ax.set_ylabel('step_length')
ax.set_zlabel('ankle angle (deg)')
ax.set_zlim(-20,40)

fig = plt.figure()
X, Y = np.meshgrid(phases, step_lengths)
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, knee_angle_model.T)
ax.set_xlabel('phase')
ax.set_ylabel('step_length')
ax.set_zlabel('knee angle (deg)')

#==============================================================================================================================

## D. Visualize Joint Model w.r.t. ramp ===============================================================================================
phases = np.linspace(0, 1, num = 50)
phase_dots = 0.8
step_lengths = 1.2
ramps = np.linspace(-10, 10, num = 50)

knee_angle_model = np.zeros((len(phases), len(ramps)))
ankle_angle_model = np.zeros((len(phases), len(ramps)))

for i in range(len(phases)):
    for j in range(len(ramps)):
        joint_angles = c_model.evaluate_h_func([Psi_knee, Psi_ankle], phases[i], phase_dots, step_lengths, ramps[j])
        knee_angle_model[i,j] = joint_angles[0]
        ankle_angle_model[i,j] = joint_angles[1]
        

fig = plt.figure()
X, Y = np.meshgrid(phases, ramps)
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, -ankle_angle_model.T) # flip the sign to compare to Kyle's figures
ax.set_xlabel('phase')
ax.set_ylabel('ramp')
ax.set_zlabel('ankle angle (deg)')
ax.set_zlim(-20,40)

fig = plt.figure()
X, Y = np.meshgrid(phases, ramps)
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, -knee_angle_model.T) # flip the sign to compare to Kyle's figures
ax.set_xlabel('phase')
ax.set_ylabel('ramp')
ax.set_zlabel('knee angle (deg)')
ax.set_zlim(-50,100)
#==============================================================================================================================

## D. Visualize Joint Model w.r.t. phase_dot ===============================================================================================
phases = np.linspace(0, 1, num = 50)
phase_dots = np.linspace(0.8, 1.2, num = 50)
step_lengths = 1.1
ramps = 0

knee_angle_model = np.zeros((len(phases), len(phase_dots)))
ankle_angle_model = np.zeros((len(phases), len(phase_dots)))

for i in range(len(phases)):
    for j in range(len(phase_dots)):
        joint_angles = c_model.evaluate_h_func([Psi_knee, Psi_ankle], phases[i], phase_dots[j], step_lengths, ramps)
        knee_angle_model[i,j] = joint_angles[0]
        ankle_angle_model[i,j] = joint_angles[1]
        

fig = plt.figure()
X, Y = np.meshgrid(phases, phase_dots)
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, ankle_angle_model.T)
ax.set_xlabel('phase')
ax.set_ylabel('phase_dots')
ax.set_zlabel('ankle angle (deg)')
ax.set_zlim(-20,40)

fig = plt.figure()
X, Y = np.meshgrid(phases, phase_dots)
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, knee_angle_model.T)
ax.set_xlabel('phase')
ax.set_ylabel('phase_dots')
ax.set_zlabel('knee angle (deg)')

#==============================================================================================================================

plt.show()