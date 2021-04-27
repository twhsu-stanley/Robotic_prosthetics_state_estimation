import numpy as np
import scipy as sp
import scipy.io as sio
from scipy.signal import butter, lfilter, filtfilt
import h5py as hp
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# low-pass filter
def butter_lowpass_filter(data, cutoff, fs, order = 1):
  # cutoff: desired cutoff frequency of the filter (Hz)
  # fs: sampling rate (Hz)
  nyq = 0.5 * fs
  normal_cutoff = cutoff / nyq
  b, a = butter(order, normal_cutoff, btype='low', analog=False)
  data_filtered = lfilter(b, a, data)
  #data_filtered = filtfilt(b, a, data)
  
  return data_filtered

# band-pass filter
def butter_bandpass_filter(data, lowcut, highcut, fs, order = 1):
  nyq = 0.5 * fs
  normal_lowcut = lowcut / nyq
  normal_highcut = highcut / nyq
  b, a = butter(order, [normal_lowcut, normal_highcut], btype='band', analog=False)
  data_filtered = lfilter(b, a, data)
  #data_filtered = filtfilt(b, a, data)
  
  return data_filtered

def plot3d(ax, data, *args, **kwargs):
  if len(data.shape)==1:
    Xs = [data[0]]
    Ys = [data[1]]
    Zs = [data[2]]
  ax.plot(Xs, Ys, Zs, *args, **kwargs)

def stick_plot_3d(ax, markers, step_ndx = 445):
  # Plots a stick-figure version of the character, with undefined colors
  ax.plot([markers['left']['toe'][1,step_ndx]*1e-3,], [-markers['left']['toe'][0,step_ndx]*1e-3,], [markers['left']['toe'][2,step_ndx]*1e-3,],"ko")
  ax.plot([markers['left']['heel'][1,step_ndx]*1e-3,], [-markers['left']['heel'][0,step_ndx]*1e-3,], [markers['left']['heel'][2,step_ndx]*1e-3,],"ko")
  ax.plot([markers['left']['asi'][1,step_ndx]*1e-3,], [-markers['left']['asi'][0,step_ndx]*1e-3,], [markers['left']['asi'][2,step_ndx]*1e-3,],"ko")
  ax.plot([markers['left']['ankle'][1,step_ndx]*1e-3,], [-markers['left']['ankle'][0,step_ndx]*1e-3,], [markers['left']['ankle'][2,step_ndx]*1e-3,],"ko")
  ax.plot([markers['left']['knee'][1,step_ndx]*1e-3,], [-markers['left']['knee'][0,step_ndx]*1e-3,], [markers['left']['knee'][2,step_ndx]*1e-3,],"ko")
  # ax.plot([markers['left']['thigh'][1,step_ndx]*1e-3,], [-markers['left']['thigh'][0,step_ndx]*1e-3,], [markers['left']['thigh'][2,step_ndx]*1e-3,],"ko")
  # ax.plot([markers['left']['tibia'][1,step_ndx]*1e-3,], [-markers['left']['tibia'][0,step_ndx]*1e-3,], [markers['left']['tibia'][2,step_ndx]*1e-3,],"ko")
  ax.plot([markers['left']['psi'][1,step_ndx]*1e-3,], [-markers['left']['psi'][0,step_ndx]*1e-3,], [markers['left']['psi'][2,step_ndx]*1e-3,],"ko")
  ax.plot([markers['right']['toe'][1,step_ndx]*1e-3,], [-markers['right']['toe'][0,step_ndx]*1e-3,], [markers['right']['toe'][2,step_ndx]*1e-3,],"ko")
  ax.plot([markers['right']['heel'][1,step_ndx]*1e-3,], [-markers['right']['heel'][0,step_ndx]*1e-3,], [markers['right']['heel'][2,step_ndx]*1e-3,],"ko")
  ax.plot([markers['right']['asi'][1,step_ndx]*1e-3,], [-markers['right']['asi'][0,step_ndx]*1e-3,], [markers['right']['asi'][2,step_ndx]*1e-3,],"ko")
  ax.plot([markers['right']['ankle'][1,step_ndx]*1e-3,], [-markers['right']['ankle'][0,step_ndx]*1e-3,], [markers['right']['ankle'][2,step_ndx]*1e-3,],"ko")
  ax.plot([markers['right']['knee'][1,step_ndx]*1e-3,], [-markers['right']['knee'][0,step_ndx]*1e-3,], [markers['right']['knee'][2,step_ndx]*1e-3,],"ko")
  # ax.plot([markers['right']['thigh'][1,step_ndx]*1e-3,], [-markers['right']['thigh'][0,step_ndx]*1e-3,], [markers['right']['thigh'][2,step_ndx]*1e-3,],"ko")
  # ax.plot([markers['right']['tibia'][1,step_ndx]*1e-3,], [-markers['right']['tibia'][0,step_ndx]*1e-3,], [markers['right']['tibia'][2,step_ndx]*1e-3,],"ko")
  ax.plot([markers['right']['psi'][1,step_ndx]*1e-3,], [-markers['right']['psi'][0,step_ndx]*1e-3,], [markers['right']['psi'][2,step_ndx]*1e-3,],"ko")
  xyz_line = [markers['left']['toe'], markers['left']['heel'], markers['left']['ankle'],
              markers['left']['knee'], markers['left']['asi'], markers['left']['psi'],
              markers['right']['psi'], markers['right']['asi'], markers['right']['knee'],
              markers['right']['ankle'], markers['right']['heel'], markers['right']['toe']
              ]
  Xs = [q[1, step_ndx]*1e-3 for q in xyz_line]
  Ys = [-q[0,step_ndx]*1e-3 for q in xyz_line]
  Zs = [q[2, step_ndx]*1e-3 for q in xyz_line]
  ax.plot(Xs, Ys, Zs, 'k')

def markers_to_kinematics(markers, index):
  lmark = markers['left']
  rmark = markers['right']
  marker_names = ['ankle', 'asi', 'heel', 'knee', 'psi', 'thigh', 'tibia', 'toe']
  points, point_names = [], []

  return points, point_names

def cop_force(ax, COP, force, step_ndx, *args, offset=[1000,0], N2m = 1/1000., **kwargs):
  Xs = [COP[1,step_ndx]*1e-3 + offset[1], COP[1,step_ndx]*1e-3 + offset[1] + force[1,step_ndx]*N2m]
  Ys = [-COP[0,step_ndx]*1e-3-offset[0], -COP[0,step_ndx]*1e-3 - offset[0] - force[0,step_ndx]*N2m]
  Zs = [-COP[2,step_ndx]*1e-3, -COP[2,step_ndx]*1e-3 -force[2,step_ndx]*N2m]
  ax.plot(Xs, Ys, Zs, *args, **kwargs)

# Wrench in {ankle frame} for Continuous structure / plot
def wrench_ankle_conti(force, moment, markers, vicon_offset):
  #force, moment, markers in CONTINUOUS structure
  # form xyz coords. for the {ankle frame}
  x_ankle = np.array([markers['toe'][1]*1e-3 - markers['heel'][1]*1e-3, 
                      -markers['toe'][0]*1e-3 + markers['heel'][0]*1e-3,
                      markers['toe'][2]*1e-3 - markers['heel'][2]*1e-3]) #in {world frame}
  x_ankle = x_ankle / np.linalg.norm(x_ankle) # normalize
  v = np.array([markers['knee'][1]*1e-3 - markers['ankle'][1]*1e-3, 
                -markers['knee'][0]*1e-3 + markers['ankle'][0]*1e-3,
                markers['knee'][2]*1e-3 - markers['ankle'][2]*1e-3]) #in {world frame}
  y_ankle = np.cross(v, x_ankle) #in {world frame}
  y_ankle = y_ankle / np.linalg.norm(y_ankle) # normalize
  z_ankle = np.cross(x_ankle, y_ankle) #in {world frame}
  z_ankle = z_ankle / np.linalg.norm(z_ankle) # normalize

  #form rotation matrix of ankle coords in {world frame} relative to forceplate coords in {world frame}
  R_fa = np.array([[x_ankle[0], y_ankle[0], z_ankle[0]],
                   [x_ankle[1], y_ankle[1], z_ankle[1]],
                   [x_ankle[2], y_ankle[2], z_ankle[2]]])
  
  #force in {forceplate frame}
  force_f = np.array([[force[1]],
                      [-force[0]], 
                      [-force[2]]]) #[N]

  #force in {ankle frame}         
  force_ankle = R_fa.T @ force_f
  force_ankle_x = force_ankle[0]
  force_ankle_y = force_ankle[1]
  force_ankle_z = force_ankle[2]

  #moment in {forceplate frame}
  moment_f = np.array([[moment[1]*1e-3],
                       [-moment[0]*1e-3], 
                       [-moment[2]*1e-3]]) #[N-m]
  
  #translation from {forceplate frame} to {ankle frame}
  p_fa = np.array([markers['ankle'][1]*1e-3 - vicon_offset[1],
                  -markers['ankle'][0]*1e-3 + vicon_offset[0],
                   markers['ankle'][2]*1e-3]) #[m]
  p_fa_hat = np.array([[0, -p_fa[2], p_fa[1]], [p_fa[2], 0, -p_fa[0]], [-p_fa[1], p_fa[0], 0]])
      
  #moment in {ankle frame}
  moment_ankle = -R_fa.T @ p_fa_hat @ force_f + R_fa.T @ moment_f
  moment_ankle_x = moment_ankle[0]
  moment_ankle_y = moment_ankle[1]
  moment_ankle_z = moment_ankle[2]

  """
  if plot == True:
    ankle = [markers['ankle'][1]*1e-3, -markers['ankle'][0]*1e-3, markers['ankle'][2]*1e-3]
    for coord in [x_ankle, y_ankle, z_ankle]:
      Xc = [ankle[0], ankle[0] + coord[0]] # in {world frame}
      Yc = [ankle[1], ankle[1] + coord[1]] # in {world frame}
      Zc = [ankle[2], ankle[2] + coord[2]] # in {world frame}
      ax.plot(Xc, Yc, Zc, "r--")
      print("x_ankle norm: ", np.linalg.norm(x_ankle))
      print("y_ankle norm: ", np.linalg.norm(y_ankle))
      print("z_ankle norm: ", np.linalg.norm(z_ankle))
  """

  return force_ankle_x, force_ankle_y, force_ankle_z, moment_ankle_x, moment_ankle_y, moment_ankle_z

# Wrench in {ankle frame} for GAITCYCLE structure
def wrench_ankle(force, moment, markers, vicon_offset):
  #force, moment, markers in GAITCYCLE structure
  data_shape = np.shape(force['x'][:])
  force_ankle_x = np.zeros(data_shape)
  force_ankle_y = np.zeros(data_shape)
  force_ankle_z = np.zeros(data_shape)
  moment_ankle_x = np.zeros(data_shape)
  moment_ankle_y = np.zeros(data_shape)
  moment_ankle_z = np.zeros(data_shape)
  for i in np.arange(data_shape[0]):
    for j in np.arange(data_shape[1]):
      # form xyz coords. for the {ankle frame}
      x_ankle = np.array([markers['toe']['y'][i,j]*1e-3 - markers['heel']['y'][i,j]*1e-3, 
                          -markers['toe']['x'][i,j]*1e-3 + markers['heel']['x'][i,j]*1e-3,
                          markers['toe']['z'][i,j]*1e-3 - markers['heel']['z'][i,j]*1e-3]) #in {world frame}
      x_ankle = x_ankle / np.linalg.norm(x_ankle) # normalize
      v = np.array([markers['knee']['y'][i,j]*1e-3 - markers['ankle']['y'][i,j]*1e-3, 
                    -markers['knee']['x'][i,j]*1e-3 + markers['ankle']['x'][i,j]*1e-3,
                    markers['knee']['z'][i,j]*1e-3 - markers['ankle']['z'][i,j]*1e-3]) #in {world frame}
      y_ankle = np.cross(v, x_ankle) #in {world frame}
      y_ankle = y_ankle / np.linalg.norm(y_ankle) # normalize
      z_ankle = np.cross(x_ankle, y_ankle) #in {world frame}
      z_ankle = z_ankle / np.linalg.norm(z_ankle) # normalize

      #form rotation matrix of ankle coords in {world frame} relative to forceplate coords in {world frame}
      R_fa = np.array([[x_ankle[0], y_ankle[0], z_ankle[0]],
                       [x_ankle[1], y_ankle[1], z_ankle[1]],
                       [x_ankle[2], y_ankle[2], z_ankle[2]]])
      
      #force in {forceplate frame}
      force_f = np.array([[force['y'][i,j]],
                          [-force['x'][i,j]], 
                          [-force['z'][i,j]]]) #[N]
      #force in {ankle frame}         
      force_a = R_fa.T @ force_f #[N]
      force_ankle_x[i,j] = force_a[0]
      force_ankle_y[i,j] = force_a[1]
      force_ankle_z[i,j] = force_a[2]

      #moment in {forceplate frame}
      moment_f = np.array([[moment['y'][i,j]*1e-3],
                           [-moment['x'][i,j]*1e-3], 
                           [-moment['z'][i,j]*1e-3]]) #[N-m]

      #translation from {forceplate frame} to {ankle frame}
      p_fa = np.array([markers['ankle']['y'][i,j]*1e-3 - vicon_offset[1],
                      -markers['ankle']['x'][i,j]*1e-3 + vicon_offset[0],
                       markers['ankle']['z'][i,j]*1e-3]) #[m]
      p_fa_hat = np.array([[0, -p_fa[2], p_fa[1]], [p_fa[2], 0, -p_fa[0]], [-p_fa[1], p_fa[0], 0]]) #to se(3)
      
      #moment in {ankle frame}
      moment_a = -R_fa.T @ p_fa_hat @ force_f + R_fa.T @ moment_f
      moment_ankle_x[i,j] = moment_a[0]
      moment_ankle_y[i,j] = moment_a[1]
      moment_ankle_z[i,j] = moment_a[2]

  return force_ankle_x, force_ankle_y, force_ankle_z, moment_ankle_x, moment_ankle_y, moment_ankle_z

def YXZ_Euler_rotation(y, x, z):
  #deg to rad
  x = x / 180 * math.pi
  y = y / 180 * math.pi
  z = z / 180 * math.pi

  #elementary rotations
  #R_x = np.array([[1, 0, 0], [0, math.cos(x), -math.sin(x)], [0, math.sin(x), math.cos(x)]])
  #R_y = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
  #R_z = np.array([[math.cos(z), -math.sin(z), 0], [math.sin(z), math.cos(z), 0], [0, 0, 1]])
  R_yxz = np.array([[math.cos(y)*math.cos(z)+math.sin(x)*math.sin(y)*math.sin(z), math.cos(z)*math.sin(x)*math.sin(y)-math.cos(y)*math.sin(z), math.cos(x)*math.sin(y)],
                    [math.cos(x)*math.sin(z), math.cos(x)*math.cos(z), -math.sin(x)],
                    [-math.cos(z)*math.sin(y)+math.cos(y)*math.sin(x)*math.sin(z), math.cos(y)*math.cos(z)*math.sin(x)+math.sin(y)*math.sin(z), math.cos(x)*math.cos(y)]])
  #Net rotation matrix
  #R_yxz = R_y @ R_x @ R_z
  return R_yxz

def YXZ_Euler_angles(R):
  #R: YXZ Euler rotation matrix in SO(3)
  if R[1,2] < 1:
    if R[1,2] > -1:
      X = np.arcsin(-R[1,2])
      Y = np.arctan2(R[0,2], R[2,2])
      Z = np.arctan2(R[1,0], R[1,1])
    else: #R12 = -1
    #Not a unique solution: Z − Y = atan2(−R01, R00)
      X = math.pi / 2
      Y = -np.arctan2(-R[0,1], R[0,0]) 
      Z = 0 
  else: #R12 = 1
    #Not a unique solution: Z + Y = atan2(−R01, R00)
    X = -math.pi / 2
    Y = np.arctan2(-R[0,1], R[0,0])
    Z = 0

  # rad to deg
  X = X / math.pi * 180
  Y = Y / math.pi * 180
  Z = Z / math.pi * 180
  return Y, X, Z

def joint_moment(ax, joint, moment, step_ndx, *args, Nm2m = 1/1000., **kwargs):
  Xs = [joint[1,step_ndx]*1e-3, joint[1,step_ndx]*1e-3 + moment[0,step_ndx]*Nm2m]
  Ys = [-joint[0,step_ndx]*1e-3, -joint[0,step_ndx]*1e-3 + moment[1,step_ndx]*Nm2m]
  Zs = [joint[2,step_ndx]*1e-3, joint[2,step_ndx]*1e-3 + moment[2,step_ndx]*Nm2m]
  ax.plot(Xs, Ys, Zs, *args, **kwargs)

def simulate_bounding_box(ax, max_range = 1.5/2, mid_x = 1, mid_y = .5, mid_z = .5):
  # Create cubic bounding box to simulate equal aspect ratio
  ax.set_xlim(mid_x - max_range, mid_x + max_range)
  ax.set_ylim(mid_y - max_range, mid_y + max_range)
  ax.set_zlim(mid_z - max_range, mid_z + max_range)

  ax.set_xlabel("x")
  ax.set_ylabel('y')
  ax.set_zlabel("z")
