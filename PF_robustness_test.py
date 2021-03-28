import numpy as np
import math
import pandas as pd
import seaborn as sns
import time
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
    sys.Q = np.diag([1e-50, 5e-5, 1e-4, 5e-2]) # process model noise covariance [1e-50, 5e-5, 1e-4, 5e-2]
    sys.R = R[subject] # measurement noise covariance

    # initialization
    init = myStruct()
    init.n = 100
    init.mu = np.array([[phases[0]], [phase_dots[0]], [step_lengths[0]], [ramps[0]]])
    init.Sigma = np.diag([1e-14, 1e-14, 1e-14, 1e-14])

    pf = particle_filter(sys, init)

    z = np.array([[global_thigh_angle_Y],\
                  [force_z_ankle], \
                  [force_x_ankle],\
                  [moment_y_ankle]])
    z = np.squeeze(z)

    heel_strike_index = Conti_heel_strikes(subject, trial, side) - Conti_heel_strikes(subject, trial, side)[0]
    kidnap_index = np.random.randint(heel_strike_index[0, 0], heel_strike_index[1, 0]) # step at which kidnapping occurs

    # kidnapping state
    phase_kidnap = np.random.uniform(0, 1)
    phase_dot_kidnap = np.random.uniform(0, 5)
    step_length_kidnap = np.random.uniform(0, 2)
    ramp_kidnap = np.random.uniform(-45, 45)
    state_kidnap = np.array([[phase_kidnap], [phase_dot_kidnap], [step_length_kidnap], [ramp_kidnap]])

    total_step =  800 #np.shape(z)[1]
    phases = phases[0 : total_step]
    phase_dots = phase_dots[0 : total_step]
    step_lengths = step_lengths[0 : total_step]
    ramps = ramps[0 : total_step]
    
    x = np.zeros((total_step, 4))  # state estimate
    #Sigma_norm = np.zeros(total_step)
    Neff = np.zeros(total_step)

    t_step_max = 0
    t_step_tot = 0
    for i in range(total_step):
        t = time.time()
        # kidnap
        if kidnap == True and i == kidnap_index:
            pf.kidnap(state_kidnap)
        pf.particles_propagation(dt)
        pf.importance_measurement(z[:, i], Psi)
        x[i,:] = pf.mu.T
        #Sigma_norm[i] = np.linalg.norm(pf.Sigma, 2)
        #Sigma_norm[i] = pf.Sigma[0,0]
        Neff[i] = pf.Neff

        t_step = time.time() - t
        t_step_tot += t_step
        if t_step > t_step_max:
            t_step_max = t_step

    print("longest time step = ", t_step_max)
    print("mean time step = ", t_step_tot / total_step)
    
    # evaluate robustness
    #track = True
    track_tol = 0.05
    #start_check = 3
    se = 0
    for i in range(kidnap_index + 100, total_step):
        error_phase = phase_error(x[i, 0], phases[i])
        se += error_phase ** 2
        #if i >= int(heel_strike_index[start_check]):
            #track = track and (error_phase < track_tol)
    RMSE_phase = np.sqrt(se / (total_step - kidnap_index - 100))
    track = (RMSE_phase < track_tol)
    print("RMSE phase = ", RMSE_phase)

    if kidnap == True:
        phase_dot_akn = x[kidnap_index, 1]
        phase_dot_b4kn = x[kidnap_index - 1, 1]
        kidnap_step = kidnap_index / (heel_strike_index[1, 0] - heel_strike_index[0, 0]) * 100 # kidmap step % of stride
        print("kidnapping step (%_stride) = ", kidnap_step)
        print("phase_dot right after kidnap = ", phase_dot_akn)
        print("phase_dot right before kidnap = ", phase_dot_b4kn)
        print("recover from kidnap? ", track)
        print("---------------------------------------------------------------")
        result = (track, RMSE_phase, phase_dot_b4kn, phase_dot_akn, kidnap_step)
    elif kidnap == False:
        print("track without kidnapping? ", track)
        result = (track, RMSE_phase)

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
    
    return result

def pf_robustness(kidnap = True):    
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
                    track, RMSE_phase, phase_dot_b4kn, phase_dot_akn, kidnap_step = pf_test(subject, trial, side, kidnap, plot = False)
                    RMSerror_phase.append([RMSE_phase, phase_dot_akn / phase_dot_b4kn, kidnap_step])
                else:
                    track, RMSE_phase = pf_test(subject, trial, side, kidnap, plot = False)
                    #RMSerror_phase.append([RMSE_phase])

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
    subject = 'AB05'
    trial= 's1x2d2x5'
    side = 'left'

    result = pf_test(subject, trial, side, kidnap = False, plot = True)
    #robustness = pf_robustness(kidnap = True)

    #print("robustness (%): ", robustness)
