import numpy as np
import matplotlib.pyplot as plt
from data_generators import *
from model_framework import *

subject = 'AB01'
stride_id = 11
n = 10

with open('Global_thigh_angle.npz', 'rb') as file:
    g_t = np.load(file)
    global_thigh_angle_Y = g_t[subject][0]

with open('Reaction_wrench.npz', 'rb') as file:
    r_w = np.load(file)
    force_x_ankle = r_w[subject][0]
    force_y_ankle = r_w[subject][1]
    force_z_ankle = r_w[subject][2]
    moment_x_ankle = r_w[subject][3]
    moment_y_ankle = r_w[subject][4]
    moment_z_ankle = r_w[subject][5]

phases = get_phase(global_thigh_angle_Y)
phase_dots = get_phase_dot(subject)
step_lengths = get_step_length(subject)
ramps = get_ramp(subject)

m_model = model_loader('Measurement_model.pickle')

with open('Measurement_model_coeff.npz', 'rb') as file:
    Measurement_model_coeff = np.load(file, allow_pickle = True)
    psi_thigh_Y = Measurement_model_coeff['global_thigh_angle_Y']
    psi_force_z = Measurement_model_coeff['reaction_force_z_ankle']
    psi_force_x = Measurement_model_coeff['reaction_force_x_ankle']
    psi_moment_y = Measurement_model_coeff['reaction_moment_y_ankle']

global_thigh_angle_Y_pred = model_prediction(m_model.models[0], psi_thigh_Y.item()[subject], phases[stride_id-n:stride_id+n,:].ravel(),\
                                                phase_dots[stride_id-n:stride_id+n,:].ravel(),\
                                                step_lengths[stride_id-n:stride_id+n,:].ravel(),\
                                                ramps[stride_id-n:stride_id+n,:].ravel())

force_z_ankle_pred = model_prediction(m_model.models[1], psi_force_z.item()[subject], phases[stride_id-n:stride_id+n,:].ravel(),\
                                                phase_dots[stride_id-n:stride_id+n,:].ravel(),\
                                                step_lengths[stride_id-n:stride_id+n,:].ravel(),\
                                                ramps[stride_id-n:stride_id+n,:].ravel())

force_x_ankle_pred = model_prediction(m_model.models[2], psi_force_x.item()[subject], phases[stride_id-n:stride_id+n,:].ravel(),\
                                                phase_dots[stride_id-n:stride_id+n,:].ravel(),\
                                                step_lengths[stride_id-n:stride_id+n,:].ravel(),\
                                                ramps[stride_id-n:stride_id+n,:].ravel())

moment_y_ankle_pred = model_prediction(m_model.models[3], psi_moment_y.item()[subject], phases[stride_id-n:stride_id+n,:].ravel(),\
                                                phase_dots[stride_id-n:stride_id+n,:].ravel(),\
                                                step_lengths[stride_id-n:stride_id+n,:].ravel(),\
                                                ramps[stride_id-n:stride_id+n,:].ravel())
# compute covariance from samples
err_gthY = global_thigh_angle_Y[stride_id-n:stride_id+n,:].ravel() - global_thigh_angle_Y_pred
print("mean gtY ", np.mean(err_gthY))
print("std gtY ", np.std(err_gthY))
err_fz = force_z_ankle[stride_id-n:stride_id+n,:].ravel() - force_z_ankle_pred
print("mean fz ", np.mean(err_fz))
print("std fz ", np.std(err_fz))
err_fx = force_x_ankle[stride_id-n:stride_id+n,:].ravel() - force_x_ankle_pred
print("mean fx ", np.mean(err_fx))
print("std fx ", np.std(err_fx))
err_my = moment_y_ankle[stride_id-n:stride_id+n,:].ravel() - moment_y_ankle_pred
print("mean my ", np.mean(err_my))
print("std my ", np.std(err_my))
err = np.stack((err_gthY, err_fz, err_fx, err_my))
R = np.cov(err)
print("R = ", R)

plt.figure()
plt.subplot(411)
plt.plot(global_thigh_angle_Y[stride_id-n:stride_id+n,:].ravel(), 'b-')
plt.plot(global_thigh_angle_Y_pred,'k--')
plt.legend(['actual','predicted'])
plt.ylabel('global_thigh_angle_Y')

plt.subplot(412)
plt.plot(force_z_ankle[stride_id-n:stride_id+n,:].ravel(), 'b-')
plt.plot(force_z_ankle_pred, 'k--')
plt.legend(['actual','predicted'])
plt.ylabel('force_z_ankle')

plt.subplot(413)
plt.plot(force_x_ankle[stride_id-n:stride_id+n,:].ravel(), 'b-')
plt.plot(force_x_ankle_pred, 'k--')
plt.legend(['actual','predicted'])
plt.ylabel('force_x_ankle')

plt.subplot(414)
plt.plot(moment_y_ankle[stride_id-n:stride_id+n,:].ravel(), 'b-')
plt.plot(moment_y_ankle_pred, 'k--')
plt.legend(['actual','predicted'])
plt.ylabel('moment_y_ankle')

plt.show()