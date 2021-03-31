import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import time
from EKF import *
from model_framework import *
from data_generators import *
from continuous_data import *
from model_fit import *

with open('Measurement_error_cov.pickle', 'rb') as file:
    R = pickle.load(file)

m_model = model_loader('Measurement_model.pickle')

def A(dt):
    return np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

def process_model(x, dt):
    #dt = 0.01 # data sampling rate: 100 Hz
    return A(dt) @ x

def ekf_test(subject, trial, side, kidnap = True, plot = False):
    dt = 1/100
    # load ground truth
    phases, phase_dots, step_lengths, ramps = Conti_state_vars(subject, trial, side)
    # load measurements
    global_thigh_angle_Y, force_z_ankle, force_x_ankle, moment_y_ankle = load_Conti_measurement_data(subject, trial, side)
    z = np.array([[global_thigh_angle_Y],\
                  [force_z_ankle], \
                  [force_x_ankle],\
                  [moment_y_ankle]])
    z = np.squeeze(z)

    Psi = load_Psi(subject)

    # build the system
    sys = myStruct()
    sys.f = process_model
    sys.A = A
    sys.h = m_model
    sys.Q = np.diag([0, 1e-4, 1e-6, 5e-4]) # process model noise covariance
    sys.R = R[subject] # measurement noise covariance

    # initialize the state
    init = myStruct()
    init.x = np.array([[phases[0]], [phase_dots[0]], [step_lengths[0]+0.5], [ramps[0]+10]])
    init.Sigma = np.diag([0, 5e-4, 1e-3, 1e-1])

    ekf = extended_kalman_filter(sys, init)
    
    heel_strike_index = Conti_heel_strikes(subject, trial, side) - Conti_heel_strikes(subject, trial, side)[0]
    kidnap_index = np.random.randint(heel_strike_index[0, 0], heel_strike_index[1, 0]) # step at which kidnapping occurs
    
    phase_kidnap = np.random.uniform(0, 1)
    phase_dot_kidnap = np.random.uniform(0, 5)
    step_length_kidnap = np.random.uniform(0, 2)
    ramp_kidnap = np.random.uniform(-45, 45)
    state_kidnap = np.array([[phase_kidnap], [phase_dot_kidnap], [step_length_kidnap], [ramp_kidnap]])

    total_step =  1000 #np.shape(z)[1]
    phases = phases[0 : total_step]
    phase_dots = phase_dots[0 : total_step]
    step_lengths = step_lengths[0 : total_step]
    ramps = ramps[0 : total_step]

    x = np.zeros((total_step, 4))  # state estimate
    t_step_max = 0
    t_step_tot = 0
    for i in range(total_step):
        t = time.time()
        
        # kidnap
        if kidnap == True and i == kidnap_index:
            ekf.x = state_kidnap

        ekf.prediction(dt)
        ekf.correction(z[:, i], Psi)
        x[i,:] = ekf.x.T
        
        t_step = time.time() - t
        t_step_tot += t_step
        if t_step > t_step_max:
            t_step_max = t_step

    print("longest time step = ", t_step_max)
    print("mean time step = ", t_step_tot / total_step)

    # evaluate robustness
    # compare x and ground truth:
    track = True
    track_tol = 0.08
    start_check = 4
    se = 0
    for i in range(total_step):
        error_phase = phase_error(x[i, 0], phases[i])
        se += error_phase ** 2
        if i >= int(heel_strike_index[start_check]):
            track = track and (error_phase < track_tol)
    
    RMSE_phase = np.sqrt(se / total_step)
    track = track or (RMSE_phase < 0.05)
    print("RMSE phase = ", RMSE_phase)

    if kidnap == True:
        phase_dot_akn = x[kidnap_index, 1]
        phase_dot_b4kn = x[kidnap_index - 1, 1]
        kidnap_step = kidnap_index / (heel_strike_index[1, 0] - heel_strike_index[0, 0]) * 100 # kidmap step % of stride
        print("kidnapping step (%_stride) = ", kidnap_step)
        print("phase_dot right after kidnap = ", phase_dot_akn)
        print("phase_dot right before kidnap = ", phase_dot_b4kn)
        print("recover from kidnap? ", track)
        result = (track, RMSE_phase, phase_dot_b4kn, phase_dot_akn, kidnap_step)
    elif kidnap == False:
        print("track without kidnapping? ", track)
        result = (track, RMSE_phase)
    
    if plot == True:
        # plot results
        plt.figure()
        plt.subplot(411)
        plt.plot(phases)
        plt.plot(x[:, 0], '--')
        plt.ylabel('phase')
        plt.subplot(412)
        plt.plot(phase_dots)
        plt.plot(x[:, 1], '--')
        plt.ylabel('phase dot')
        plt.subplot(413)
        plt.plot(step_lengths)
        plt.plot(x[:, 2], '--')
        plt.ylabel('step length')
        plt.subplot(414)
        plt.plot(ramps)
        plt.plot(x[:, 3], '--')
        plt.ylabel('ramp')
        plt.show()

    return result

def ekf_bank_test(subject, trial, side, plot = True):
    dt = 1/100
    # load ground truth
    phases, phase_dots, step_lengths, ramps = Conti_state_vars(subject, trial, side)
    # load measurements
    global_thigh_angle_Y, force_z_ankle, force_x_ankle, moment_y_ankle = load_Conti_measurement_data(subject, trial, side)
    z = np.array([[global_thigh_angle_Y],\
                  [force_z_ankle], \
                  [force_x_ankle],\
                  [moment_y_ankle]])
    z = np.squeeze(z)
    
    Psi = load_Psi(subject)

    # build the system
    sys = myStruct()
    sys.f = process_model
    sys.A = A
    sys.h = m_model
    sys.Q = np.diag([0, 1e-4, 1e-6, 5e-4]) #([0, 1e-5, 1e-7, 5e-5]) # process model noise covariance
    sys.R = R[subject] # measurement noise covariance
    init = myStruct()

    total_step =  1000 #np.shape(z)[1]
    # ground truth states
    phases = phases[0 : total_step]
    phase_dots = phase_dots[0 : total_step]
    step_lengths = step_lengths[0 : total_step]
    ramps = ramps[0 : total_step]
    
    N = 100 # number of EKFs in the EKF bank
    kidnap_index = 30 # step at which kidnapping occurs
    x = np.zeros((N, total_step, 4))  # state estimate
    phase_dot_ROC = np.zeros(N)
    phase_rakn = np.zeros(N)
    phase_dot_rakn = np.zeros(N)
    step_length_rakn = np.zeros(N)
    ramp_rakn = np.zeros(N)
    for n in range(N):
        # initialize the state
        init.x = np.array([[phases[0]], [phase_dots[0]], [step_lengths[0]], [ramps[0]]])
        init.Sigma = np.diag([0, 5e-4, 1e-3, 1e-1])
        # build EKF
        ekf = extended_kalman_filter(sys, init)

        phase_kidnap = np.random.uniform(0, 1)
        phase_dot_kidnap = np.random.uniform(0, 5)
        step_length_kidnap = np.random.uniform(0, 2)
        ramp_kidnap = np.random.uniform(-45, 45)
        state_kidnap = np.array([[phase_kidnap], [phase_dot_kidnap], [step_length_kidnap], [ramp_kidnap]])
    
        for i in range(total_step):
            # kidnap
            if i == kidnap_index:
                ekf.x = state_kidnap
            ekf.prediction(dt)
            ekf.correction(z[:, i], Psi)
            x[n, i,:] = ekf.x.T
        
        phase_rakn[n] = x[n, kidnap_index, 0] #- phases[kidnap_index]
        phase_dot_rakn[n] = x[n, kidnap_index, 1] #- phase_dots[kidnap_index]
        step_length_rakn[n] = x[n, kidnap_index, 2] #- step_lengths[kidnap_index]
        ramp_rakn[n] = x[n, kidnap_index, 3] #- ramps[kidnap_index]
        phase_dot_ROC[n] = x[n, -1, 1]
    
    if plot == True:
        # plot results
        plt.figure("states")
        plt.subplot(411)
        plt.plot(phases)
        plt.plot(range(total_step), x[:, :, 0].T, '--')
        plt.ylabel('phase')
        plt.subplot(412)
        plt.plot(phase_dots)
        plt.plot(range(total_step), x[:, :, 1].T, '--')
        plt.ylabel('phase dot')
        plt.subplot(413)
        plt.plot(step_lengths)
        plt.plot(range(total_step), x[:, :, 2].T, '--')
        plt.ylabel('step length')
        plt.subplot(414)
        plt.plot(ramps)
        plt.plot(range(total_step), x[:, :, 3].T, '--')
        plt.ylabel('ramp')

        plt.figure("phase_dot cluster")
        plt.hist(phase_dot_ROC)
        plt.xlabel('phase_dot in the end')
        plt.ylabel('counts')

        plt.figure("Region of attraction_1")
        ax = plt.axes(projection='3d')
        for n in range(N):
            if phase_dot_ROC[n] < phase_dots[-1] + 0.2 and phase_dot_ROC[n] > phase_dots[-1] - 0.2:
                ax.scatter3D(phase_rakn[n], phase_dot_rakn[n], step_length_rakn[n], c = 'r')
            elif phase_dot_ROC[n] < 0.2 and phase_dot_ROC[n] > - 0.3:
                ax.scatter3D(phase_rakn[n], phase_dot_rakn[n], step_length_rakn[n], c = 'b')
            else:
                ax.scatter3D(phase_rakn[n], phase_dot_rakn[n], step_length_rakn[n], c = 'k')
        ax.set_xlabel('phase right after kidnapping')
        ax.set_ylabel('phase_dot right after kidnapping')
        ax.set_zlabel('step_length right after kidnapping')

        plt.figure("Region of attraction_2")
        ax = plt.axes(projection='3d')
        for n in range(N):
            if phase_dot_ROC[n] < phase_dots[-1] + 0.2 and phase_dot_ROC[n] > phase_dots[-1] - 0.2:
                ax.scatter3D(phase_rakn[n], phase_dot_rakn[n], ramp_rakn[n], c = 'r')
            elif phase_dot_ROC[n] < 0.2 and phase_dot_ROC[n] > - 0.3:
                ax.scatter3D(phase_rakn[n], phase_dot_rakn[n], ramp_rakn[n], c = 'b')
            else:
                ax.scatter3D(phase_rakn[n], phase_dot_rakn[n], ramp_rakn[n], c = 'k')
        ax.set_xlabel('phase right after kidnapping')
        ax.set_ylabel('phase_dot right after kidnapping')
        ax.set_zlabel('ramp right after kidnapping')

        plt.figure("Region of attraction_3")
        ax = plt.axes(projection='3d')
        for n in range(N):
            if phase_dot_ROC[n] < phase_dots[-1] + 0.2 and phase_dot_ROC[n] > phase_dots[-1] - 0.2:
                ax.scatter3D(phase_rakn[n], ramp_rakn[n], step_length_rakn[n], c = 'r')
            elif phase_dot_ROC[n] < 0.2 and phase_dot_ROC[n] > - 0.3:
                ax.scatter3D(phase_rakn[n], ramp_rakn[n], step_length_rakn[n], c = 'b')
            else:
                ax.scatter3D(phase_rakn[n], ramp_rakn[n], step_length_rakn[n], c = 'k')
        ax.set_xlabel('phase right after kidnapping')
        ax.set_ylabel('ramp right after kidnapping')
        ax.set_zlabel('step_length right after kidnapping')

        plt.figure("Region of attraction_4")
        ax = plt.axes(projection='3d')
        for n in range(N):
            if phase_dot_ROC[n] < phase_dots[-1] + 0.2 and phase_dot_ROC[n] > phase_dots[-1] - 0.2:
                ax.scatter3D(ramp_rakn[n], phase_dot_rakn[n], step_length_rakn[n], c = 'r')
            elif phase_dot_ROC[n] < 0.2 and phase_dot_ROC[n] > - 0.3:
                ax.scatter3D(ramp_rakn[n], phase_dot_rakn[n], step_length_rakn[n], c = 'b')
            else:
                ax.scatter3D(ramp_rakn[n], phase_dot_rakn[n], step_length_rakn[n], c = 'k')
        ax.set_xlabel('ramp right after kidnapping')
        ax.set_ylabel('phase_dot right after kidnapping')
        ax.set_zlabel('step_length right after kidnapping')

        plt.show()

def ekf_robustness(kidnap = True, RMSE_heatmap = False):
    track_count = 0
    total_trials = 0
    RMSerror_phase = []

    #for subject in Conti_subject_names():
    for subject in ['AB01', 'AB02', 'AB09']:
        print("subject: ", subject)
        for trial in Conti_trial_names(subject):
        #for trial in ['s1x2d2x5']:
            if trial == 'subjectdetails':
                continue
            print("trial: ", trial)
            for side in ['left']:
                #print("side: ", side)
                total_trials = total_trials + 1
                
                if kidnap == True:
                    track, RMSE_phase, phase_dot_b4kn, phase_dot_akn, kidnap_step = ekf_test(subject, trial, side, kidnap, plot = False)
                    RMSerror_phase.append([RMSE_phase, phase_dot_akn / phase_dot_b4kn, kidnap_step])
                else:
                    track, RMSE_phase = ekf_test(subject, trial, side, kidnap, plot = False)
                    RMSerror_phase.append([RMSE_phase])
                
                if  track == True:
                    track_count = track_count + 1

    robustness = track_count / total_trials * 100
    print("robustness (%) = ", robustness)

    if kidnap == True:
        RMSerror_phase = np.array(RMSerror_phase).reshape(-1, 3)
        RMSerror_phase_df = pd.DataFrame(RMSerror_phase, columns = ['RMSE', 'x', 'y'])
        sns.heatmap(RMSerror_phase_df.pivot('y', 'x', 'RMSE'))
        plt.title("RMSE of phase")
        plt.xlabel("phase_dot after kidnapping / phase_dot")
        plt.ylabel("kidnapping step (%_stride)")
        plt.show()
    else:
        pass
        # heatmap for normal test

    return robustness

if __name__ == '__main__':
    subject = 'AB02'
    trial= 's1x2i10'
    side = 'left'

    #ekf_test(subject, trial, side, kidnap = False, plot = True)
    ekf_bank_test(subject, trial, side, plot = True)
    #ekf_robustness(kidnap = True, RMSE_heatmap = True)
