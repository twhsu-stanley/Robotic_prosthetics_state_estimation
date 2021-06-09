import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from mpl_toolkits import mplot3d
from model_framework import *
from model_fit import *
from EKF import wrapTo2pi

m_model = model_loader('Measurement_model_6_sp.pickle')

subject = 'AB01'
Psi = load_Psi(subject)[[0, 1, 2, 3, 6, 7]]

sensor = 3

phases = np.linspace(0, 1, num = 100)
phase_dots = np.linspace(0.6, 1, num = 500)
z1 = np.zeros((len(phases), len(phase_dots)))
step_lengths = 1.6
ramps = -10
for i in range(len(phases)):
    for j in range(len(phase_dots)):
        #z[i,j] = model_prediction(m_model.models[sensor], Psi[sensor], phases[i], phase_dots[j], step_lengths, ramps)
        m = m_model.evaluate_h_func(Psi, phases[i], phase_dots[j], step_lengths, ramps)
        if sensor == 7:
            #z1[i,j] = wrapTo2pi(m[sensor] + 2 * np.pi * phases[i])
            z1[i,j] = m[sensor]
        else:
            z1[i,j] = m[sensor]

fig = plt.figure()
X, Y = np.meshgrid(phases, phase_dots)
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, z1.T)
ax.set_xlabel('phase')
ax.set_ylabel('phase_dot')

phases = 0.9
phase_dots = 0.8
step_lengths = np.linspace(0.9, 1.6, num = 50)
ramps = np.linspace(-10, 10, num = 100)
z2 = np.zeros((len(step_lengths), len(ramps)))
for i in range(len(step_lengths)):
    for j in range(len(ramps)):
        #z[i,j] = model_prediction(m_model.models[sensor], Psi[sensor], phases[i], phase_dots[j], step_lengths, ramps)
        m = m_model.evaluate_h_func(Psi, phases, phase_dots, step_lengths[i], ramps[j])
        if sensor == 7:
            z2[i,j] = wrapTo2pi(m[sensor] + 2 * np.pi * phases)
            
        else:
            z2[i,j] = m[sensor]

fig = plt.figure()
X, Y = np.meshgrid(step_lengths, ramps)
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, z2.T)
ax.set_xlabel('step_lengths')
ax.set_ylabel('ramps')

plt.show()
