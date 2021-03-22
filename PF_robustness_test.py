import numpy as np
import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PF import *
from model_framework import *
from continuous_data import *
from model_fit import *

with open('Measurement_error_cov.pickle', 'rb') as file:
    R = pickle.load(file)

measurement_model = model_loader('Measurement_model.pickle')

class myStruct:
    pass

def process_model(x, dt, pn):
    # dt = 0.01 # data sampling rate: 100 Hz
    # pn: additive noise
    return np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ x + pn

def pf_test(subject, trial, side, kidnap = True, plot = True):
    dt = 1/100
    phases, phase_dots, step_lengths, ramps = Conti_state_vars(subject, trial, side)
    global_thigh_angle_Y, force_z_ankle, force_x_ankle, moment_y_ankle = load_Conti_measurement_data(subject, trial, side)
    Psi = load_Psi(subject)

    # build the system
    sys = myStruct()
    sys.f = process_model
    sys.h = measurement_model
    sys.Q = np.diag([1e-14, 1e-7, 1e-7, 5e-5]) # process model noise covariance
    sys.R = R[subject] # measurement noise covariance

    # initialization
    init = myStruct()
    init.n = 100
    init.mu = np.array([[phases[0]], [phase_dots[0]], [step_lengths[0]], [ramps[0]]])
    #init.Sigma = np.diag([1e-14, 5e-4, 1e-3, 1e-1])
    init.Sigma = np.diag([1e-14, 1e-14, 1e-14, 1e-14])

    pf = particle_filter(sys, init)

    z = np.array([[global_thigh_angle_Y],\
                  [force_z_ankle], \
                  [force_x_ankle],\
                  [moment_y_ankle]])
    z = np.squeeze(z)

    heel_strike_index = Conti_heel_strikes(subject, trial, side) - Conti_heel_strikes(subject, trial, side)[0]
    kidnap_index = np.random.randint(heel_strike_index[0], heel_strike_index[1]) # step at which kidnapping occurs

    # kidnapping state
    phase_kidnap = np.random.uniform(0, 1)
    phase_dot_kidnap = np.random.uniform(0, 5)
    step_length_kidnap = np.random.uniform(0, 2)
    ramp_kidnap = np.random.uniform(-45, 45)
    state_kidnap = np.array([[phase_kidnap], [phase_dot_kidnap], [step_length_kidnap], [ramp_kidnap]])

    total_step = 800 # = np.shape(z)[1]
    phases = phases[0 : total_step]
    phase_dots = phase_dots[0 : total_step]
    step_lengths = step_lengths[0 : total_step]
    ramps = ramps[0 : total_step]
    
    x = np.zeros((total_step, 4))  # state estimate
    #Sigma_norm = np.zeros(total_step)
    Neff = np.zeros(total_step)
    for i in range(total_step):
        # kidnap
        if kidnap == True and i == kidnap_index:
            pf.kidnap(state_kidnap)
            print("kidnap index = ", kidnap_index)
        
        pf.particles_propagation(dt)
        pf.importance_measurement(z[:, i], Psi)
        x[i,:] = pf.mu.T
        #Sigma_norm[i] = np.linalg.norm(pf.Sigma, 2)
        #Sigma_norm[i] = pf.Sigma[0,0]
        Neff[i] = pf.Neff

    # evaluate robustness
    track = True
    track_tol = 0.08
    start_check = 4
    se = 0
    #for i in range(int(heel_strike_index[0]), int(heel_strike_index[np.size(heel_strike_index)-1])):
    for i in range(total_step):
        error_phase = phase_error(x[i, 0], phases[i])
        se += error_phase ** 2
        if i >= int(heel_strike_index[start_check]):
            track = track and (error_phase < track_tol)
    
    RMSE_phase = np.sqrt(se / total_step)
    #track = (RMSE_phase < track_tol)
    print("RMSE phase = ", RMSE_phase)

    if kidnap == True:
        print("recover from kidnap? ", track)
        phase_dot_akn = x[kidnap_index, 1][0]
        phase_dot_b4kn = x[kidnap_index - 1, 1][0]
        print("phase_dot right after kidnap = ", phase_dot_akn)
        print("phase_dot right before kidnap = ", phase_dot_b4kn)
    elif kidnap == False:
        print("track without kidnapping? ", track)

    # plot results
    if plot == True:
        plt.figure('Effective number of particles')
        plt.plot(Neff)
        plt.ylabel('Neff')

        plt.figure('Estimates')
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
    
    return track, RMSE_phase, phase_dot_b4kn, phase_dot_akn

def pf_robustness(kidnap = True, RMSE_heatmap = False):    
    track_count = 0
    total_trials = 0
    RMSerror_phase = []

    #for subject in Conti_subject_names():
    for subject in ['AB01', 'AB10']:
        print("subject: ", subject)

        for trial in Conti_trial_names(subject):
        #for trial in ['s1x2d2x5']:
            if trial == 'subjectdetails':
                continue
            print("trial: ", trial)
            for side in ['left']:
                #print("side: ", side)
                total_trials = total_trials + 1
                track, RMSE_phase, phase_dot_b4kn, phase_dot_akn = pf_test(subject, trial, side, kidnap = True, plot = False)
                RMSerror_phase.append([RMSE_phase, phase_dot_b4kn, phase_dot_akn])
                if  track == True:
                    track_count = track_count + 1
    
    robustness = track_count / total_trials * 100

    RMSerror_phase = np.array(RMSerror_phase).reshape(-1, 3)
    with open('RMSE_phase_PF.pickle', 'wb') as file:
            pickle.dump(RMSerror_phase, file)
    
    if RMSE_heatmap == True:
        RMSerror_phase_df = pd.DataFrame(RMSerror_phase, columns=['RMSE', 'x', 'y'])
        sns.heatmap(RMSerror_phase_df.pivot('y', 'x', 'RMSE'))
        plt.title("RMSE of phase")
        plt.xlabel("phase_dot before kidnapping")
        plt.ylabel("phase_dot after kidnapping")
        plt.show()

    return robustness

if __name__ == '__main__':
    subject = 'AB10'
    trial= 's1x2d2x5'
    side = 'left'

    #pf_test(subject, trial, side, kidnap = True, plot = True)
    pf_robustness(kidnap = True, RMSE_heatmap = True)

    #Q = np.diag([1e-14, 1e-7, 1e-7, 5e-5]) # process model noise covariance
    #print("Q =\n", Q)
    #robustness = ekf_robustness(Q, kidnap = False, RMSE_heatmap = True)
    #print("robustness (%): ", robustness)
