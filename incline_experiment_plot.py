import numpy as np
import scipy as sp
import scipy.io as sio
import h5py as hp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from incline_experiment_utils import *

# Visualization of the experimental data 
# {World frame}: X: walking direction; Y: to the left; Z: upwards

# load data
mat = hp.File("../InclineExperiment.mat", "r")
markers = mat['Continuous']['AB07']['s0x8i0']['kinematics']['markers']
jointangles = mat['Continuous']['AB07']['s0x8i0']['kinematics']['jointangles'] #deg
forceplate = mat['Continuous']['AB07']['s0x8i0']['kinetics']['forceplate']
jointmoment = mat['Continuous']['AB07']['s0x8i0']['kinetics']['jointmoment']
jointforce = mat['Continuous']['AB07']['s0x8i0']['kinetics']['jointforce']
subject = mat['Continuous']['AB07']['subjectdetails']

step_ndx = 445

ax = plt.gcf().add_subplot(111, projection='3d')

#plot markers of joints
stick_plot_3d(ax, markers, step_ndx = step_ndx)

# forceplate origin location
# in Vicon frame 
vicon_leftbelt_offset = np.array([-768, 885])*1e-3 #[m]
vicon_rightbelt_offset = np.array([-255, 885])*1e-3 #[m]
# in world frame
left_plate_origin = np.array([vicon_leftbelt_offset[1], -vicon_leftbelt_offset[0], 0.0]) 
right_plate_origin = np.array([vicon_rightbelt_offset[1], -vicon_rightbelt_offset[0], 0.0])
# plot forceplate origin in world frame
plot3d(ax, left_plate_origin, 'kx', zorder=15, ms=5)
plot3d(ax, right_plate_origin, 'kx', zorder=15, ms=5)

# plot COP and ground reaction force
# plot COP location
ax.plot([forceplate['left']['cop'][1,step_ndx]*1e-3+vicon_leftbelt_offset[1],], [-forceplate['left']['cop'][0,step_ndx]*1e-3+-vicon_leftbelt_offset[0],], [forceplate['left']['cop'][2,step_ndx]*1e-3,],"gx")
ax.plot([forceplate['right']['cop'][1,step_ndx]*1e-3+vicon_rightbelt_offset[1],], [-forceplate['right']['cop'][0,step_ndx]*1e-3-vicon_rightbelt_offset[0],], [forceplate['right']['cop'][2,step_ndx]*1e-3,],"gx")
# plot ground reaction force
cop_force(ax, forceplate['left']['cop'], forceplate['left']['force'], step_ndx, 'y' , offset = vicon_leftbelt_offset)
cop_force(ax, forceplate['right']['cop'], forceplate['right']['force'], step_ndx, 'y' , offset = vicon_rightbelt_offset)

# plot joint moment
#joint_moment(ax, markers['left']['ankle'], jointmoment['left']['ankle'], step_ndx, 'c')
#joint_moment(ax, markers['right']['ankle'], jointmoment['right']['ankle'], step_ndx, 'c')
#joint_moment(ax, markers['left']['knee'], jointmoment['left']['knee'], step_ndx, 'c')
#joint_moment(ax, markers['right']['knee'], jointmoment['right']['knee'], step_ndx, 'c')
#joint_moment(ax, markers['left']['asi'], jointmoment['left']['hip'], step_ndx, 'c')
#joint_moment(ax, markers['right']['asi'], jointmoment['right']['hip'], step_ndx, 'c')

# plot force in at ankle in anfle frame
wrench_ankle_conti(ax, forceplate['left']['force'], forceplate['left']['moment'], markers['left'], vicon_leftbelt_offset, step_ndx, plot = True, N2m = 1/1000.)
wrench_ankle_conti(ax, forceplate['right']['force'], forceplate['right']['moment'], markers['right'], vicon_rightbelt_offset, step_ndx, plot = True, N2m = 1/1000.)

simulate_bounding_box(ax)
#plt.show()

# Global thigh angle
#Left
R_wp_L = YXZ_Euler_rotation(-jointangles['left']['pelvis'][0, step_ndx], jointangles['left']['pelvis'][1, step_ndx], -jointangles['left']['pelvis'][2, step_ndx])
R_pt_L = YXZ_Euler_rotation(jointangles['left']['hip'][0, step_ndx], -jointangles['left']['hip'][1, step_ndx], -jointangles['left']['hip'][2, step_ndx])
R_wt_L = R_wp_L @ R_pt_L
Y_th_L, X_th_L, Z_th_L = YXZ_Euler_angles(R_wt_L) #[deg]
print("(Y_th_L, X_th_L, Z_th_L) = ", Y_th_L, X_th_L, Z_th_L)

#Right
R_wp_R = YXZ_Euler_rotation(-jointangles['right']['pelvis'][0, step_ndx], -jointangles['right']['pelvis'][1, step_ndx], jointangles['right']['pelvis'][2, step_ndx])
R_pt_R = YXZ_Euler_rotation(jointangles['right']['hip'][0, step_ndx], jointangles['right']['hip'][1, step_ndx], jointangles['right']['hip'][2, step_ndx])
R_wt_R = R_wp_R @ R_pt_R
Y_th_R, X_th_R, Z_th_R = YXZ_Euler_angles(R_wt_R) #[deg]
print("(Y_th_R, X_th_R, Z_th_R) = ", Y_th_R, X_th_R, Z_th_R)

# Plot Time Series ##########################################################################################################################################
n_s = np.size(jointangles['left']['pelvis'][0,:]) # number of steps

# plot wrench in {ankle frame} ######################################################################################################################
# plot force at forceplate in {world frame}
plt.figure(2)
plt.subplot(211)
plt.title("Ground reaction force at forceplate in world frame: left")
plt.plot(range(n_s), forceplate['left']['force'][1,:], range(n_s), -forceplate['left']['force'][0,:], range(n_s), -forceplate['left']['force'][2,:])
plt.ylabel("force [N]")
plt.legend(('X','Y','Z'))
plt.subplot(212)
plt.title("Ground reaction force at forceplate in world frame: right")
plt.plot(range(n_s), forceplate['right']['force'][1,:], range(n_s), -forceplate['right']['force'][0,:], range(n_s), -forceplate['right']['force'][2,:])
plt.ylabel("force [N]")
plt.legend(('X','Y','Z'))

# plot moment at forceplate in {world frame}
plt.figure(3)
plt.subplot(211)
plt.title("Ground reaction moment at forceplate in world frame: left")
plt.plot(range(n_s), forceplate['left']['moment'][1,:]*1e-3, range(n_s), -forceplate['left']['moment'][0,:]*1e-3, range(n_s), -forceplate['left']['moment'][2,:]*1e-3)
plt.ylabel("Moment [N-m]")
plt.legend(('X','Y','Z'))
plt.subplot(212)
plt.title("Ground reaction moment at forceplate in world frame: right")
plt.plot(range(n_s), forceplate['right']['moment'][1,:]*1e-3, range(n_s), -forceplate['right']['moment'][0,:]*1e-3, range(n_s), -forceplate['right']['moment'][2,:]*1e-3)
plt.ylabel("Moment [N-m]")
plt.legend(('X','Y','Z'))

# plot force & moment in {ankle frame}
n_s = 600
force_ankle_x_L = np.zeros((1, n_s))
force_ankle_y_L = np.zeros((1, n_s))
force_ankle_z_L = np.zeros((1, n_s))
moment_ankle_x_L = np.zeros((1, n_s))
moment_ankle_y_L = np.zeros((1, n_s))
moment_ankle_z_L = np.zeros((1, n_s))
force_ankle_x_R = np.zeros((1, n_s))
force_ankle_y_R = np.zeros((1, n_s))
force_ankle_z_R = np.zeros((1, n_s))
moment_ankle_x_R = np.zeros((1, n_s))
moment_ankle_y_R = np.zeros((1, n_s))
moment_ankle_z_R = np.zeros((1, n_s))
for i in range(n_s):
    force_ankle_x_L[0, i], force_ankle_y_L[0, i], force_ankle_z_L[0, i], moment_ankle_x_L[0, i], moment_ankle_y_L[0, i], moment_ankle_z_L[0, i] =\
        wrench_ankle_conti(ax, forceplate['left']['force'], forceplate['left']['moment'], markers['left'], vicon_leftbelt_offset, i, plot = False)
    force_ankle_x_R[0, i], force_ankle_y_R[0, i], force_ankle_z_R[0, i], moment_ankle_x_R[0, i], moment_ankle_y_R[0, i], moment_ankle_z_R[0, i] =\
        wrench_ankle_conti(ax, forceplate['right']['force'], forceplate['right']['moment'], markers['right'], vicon_rightbelt_offset, i, plot = False)
plt.figure(4)
plt.subplot(211)
plt.title("Ground reaction force in ankle frame: left")
plt.plot(range(n_s), force_ankle_x_L[0,:], range(n_s), force_ankle_y_L[0,:], range(n_s), force_ankle_z_L[0,:])
plt.ylabel("force [N]")
plt.legend(('X_ankle','Y_ankle','Z_ankle'))
plt.subplot(212)
plt.title("Ground reaction force in ankle frame: right")
plt.plot(range(n_s), force_ankle_x_R[0,:], range(n_s), force_ankle_y_R[0,:], range(n_s), force_ankle_z_R[0,:])
plt.ylabel("force [N]")
plt.legend(('X_ankle','Y_ankle','Z_ankle'))

plt.figure(5)
plt.subplot(211)
plt.title("Ground reaction moment in ankle frame: left")
plt.plot(range(n_s), moment_ankle_x_L[0,:], range(n_s), moment_ankle_y_L[0,:], range(n_s), moment_ankle_z_L[0,:])
plt.ylabel("Moment [N-m]")
plt.legend(('X_ankle','Y_ankle','Z_ankle'))
plt.subplot(212)
plt.title("Ground reaction moment in ankle frame: right")
plt.plot(range(n_s), moment_ankle_x_R[0,:], range(n_s), moment_ankle_y_R[0,:], range(n_s), moment_ankle_z_R[0,:])
plt.ylabel("Moment [N-m]")
plt.legend(('X_ankle','Y_ankle','Z_ankle'))

# plot Global thigh angle ##########################################################################################################################
# plot joint angles: pelvis
n_s = np.size(jointangles['left']['pelvis'][0,:]) # number of steps
plt.figure(6)
plt.subplot(211)
plt.title("Pelvis (absolute) YXZ Euler angles: left")
plt.plot(range(n_s), -jointangles['left']['pelvis'][0,:], range(n_s), jointangles['left']['pelvis'][1,:], range(n_s), -jointangles['left']['pelvis'][2,:])
plt.ylabel("angles [deg]")
plt.legend(('-x=Y','y=X','z=Z'))
plt.subplot(212)
plt.title("Pelvis (absolute) YXZ Euler angles: right")
plt.plot(range(n_s), -jointangles['right']['pelvis'][0,:], range(n_s), -jointangles['right']['pelvis'][1,:], range(n_s), jointangles['right']['pelvis'][2,:])
plt.ylabel("angles [deg]")
plt.legend(('-x=Y','y=X','z=Z'))

# plot jointangles: hip
plt.figure(7)
plt.subplot(211)
plt.title("Hip (relative) YXZ Euler angles: left")
plt.plot(range(n_s), jointangles['left']['hip'][0,:], range(n_s), -jointangles['left']['hip'][1,:], range(n_s), -jointangles['left']['hip'][2,:])
plt.ylabel("angles [deg]")
plt.legend(('-x=Y','y=X','z=Z'))
plt.subplot(212)
plt.title("Hip (relative) YXZ Euler angles: right")
plt.plot(range(n_s), jointangles['right']['hip'][0,:], range(n_s), jointangles['right']['hip'][1,:], range(n_s), jointangles['right']['hip'][2,:])
plt.ylabel("angles [deg]")
plt.legend(('-x=Y','y=X','z=Z'))

# plot global thigh angle
n_s = 600
Y_th_L = np.zeros((1, n_s))
X_th_L = np.zeros((1, n_s))
Z_th_L = np.zeros((1, n_s))
Y_th_R = np.zeros((1, n_s))
X_th_R = np.zeros((1, n_s))
Z_th_R = np.zeros((1, n_s))
for i in range(n_s):
    R_wp_L = YXZ_Euler_rotation(-jointangles['left']['pelvis'][0, i], jointangles['left']['pelvis'][1, i], -jointangles['left']['pelvis'][2, i])
    R_pt_L = YXZ_Euler_rotation(jointangles['left']['hip'][0, i], -jointangles['left']['hip'][1, i], -jointangles['left']['hip'][2, i])
    R_wt_L = R_wp_L @ R_pt_L
    Y_th_L[0, i], X_th_L[0, i], Z_th_L[0, i] = YXZ_Euler_angles(R_wt_L)

    R_wp_R = YXZ_Euler_rotation(-jointangles['right']['pelvis'][0, i], -jointangles['right']['pelvis'][1, i], jointangles['right']['pelvis'][2, i])
    R_pt_R = YXZ_Euler_rotation(jointangles['right']['hip'][0, i], jointangles['right']['hip'][1, i], jointangles['right']['hip'][2, i])
    R_wt_R = R_wp_R @ R_pt_R
    Y_th_R[0, i], X_th_R[0, i], Z_th_R[0, i] = YXZ_Euler_angles(R_wt_R)

plt.figure(8)
plt.subplot(211)
plt.title("Global thigh YXZ Euler angles: left")
plt.plot(range(n_s), Y_th_L[0,:], range(n_s), X_th_L[0,:], range(n_s), Z_th_L[0,:])
plt.ylabel("angles [deg]")
plt.legend(('Y','X','Z'))
plt.subplot(212)
plt.title("Global thigh YXZ Euler angles: right")
plt.plot(range(n_s), Y_th_R[0,:], range(n_s), X_th_R[0,:], range(n_s), Z_th_R[0,:])
plt.ylabel("angles [deg]")
plt.legend(('Y','X','Z'))

plt.show()
