import numpy as np
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

def A(dt):
    return np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

def process_model(x, dt, pn):
    # dt = 0.01 # data sampling rate: 100 Hz
    # pn: additive noise
    return A(dt) @ x + pn

def pf_test(subject, trial, side, kidnap = True, plot = True):
    dt = 1/100
    phases, phase_dots, step_lengths, ramps = Conti_state_vars(subject, trial, side)
    global_thigh_angle_Y, force_z_ankle, force_x_ankle, moment_y_ankle = load_Conti_measurement_data(subject, trial, side)
    Psi = load_Psi(subject)

    # build the system
    sys = myStruct()
    sys.f = process_model
    sys.h = measurement_model
    sys.Q = np.diag([1e-14, 5e-3, 5e-3, 5e-1]) # process model noise covariance
    sys.R = R[subject] # measurement noise covariance

    # initialization
    init = myStruct()
    init.n = 200
    init.mu = np.array([[phases[0]], [phase_dots[0]], [step_lengths[0]], [ramps[0]]])
    init.Sigma = np.diag([1e-14, 5e-4, 1e-3, 1e-1])

    pf = particle_filter(sys, init)

    z = np.array([[global_thigh_angle_Y],\
                  [force_z_ankle], \
                  [force_x_ankle],\
                  [moment_y_ankle]])
    z = np.squeeze(z)
    
    heel_strike_index = Conti_heel_strikes(subject, trial, side) - Conti_heel_strikes(subject, trial, side)[0]
    kidnap_index = np.random.randint(heel_strike_index[0], heel_strike_index[1]) # step at which kidnapping occurs
    
    phase_kidnap = np.random.uniform(0, 1)
    phase_dot_kidnap = np.random.uniform(0, 5)
    step_length_kidnap = np.random.uniform(0, 2)
    ramp_kidnap = np.random.uniform(-45, 45)
    state_kidnap = np.array([[phase_kidnap], [phase_dot_kidnap], [step_length_kidnap], [ramp_kidnap]])

    
    phases = phases[0:1000]
    phase_dots = phase_dots[0:1000]
    step_lengths = step_lengths[0:1000]
    ramps = ramps[0:1000]
    #x = np.zeros((np.shape(z)[1], 4))  # state estimate
    x = np.zeros((1000, 4))  # state estimate
    #for i in range(np.shape(z)[1]):
    for i in range(1000):
        # kidnap
        if kidnap == True and i == kidnap_index:
            pf.kidnap(state_kidnap)
            print("kidnap index = ", kidnap_index)

        pf.sample_motion(dt)
        pf.importance_measurement(z[:, i], Psi)
        x[i,:] = pf.mu.T

    # evaluate robustness
    # compare x and ground truth:
    track = True
    track_tol = 0.08
    start_check = 4
    se = 0
    #for i in range(int(heel_strike_index[0]), int(heel_strike_index[np.size(heel_strike_index)-1])):
    for i in range(1000):
        error_phase = phase_error(x[i, 0], phases[i])
        se += error_phase ** 2
        if i >= int(heel_strike_index[start_check]):
            track = track and (error_phase < track_tol)
        
    RMSE_phase = np.sqrt(se / np.size(phases))
    print("RMSE phase = ", RMSE_phase)

    if kidnap == True:
        print("recover from kidnap? ", track)
    elif kidnap == False:
        print("track without kidnapping? ", track)

    # plot results
    if plot == True:
        plt.figure()
        plt.subplot(411)
        plt.plot(phases)
        plt.plot(x[:, 0], '--')
        plt.ylabel('phase')
        plt.subplot(412)
        plt.plot(phase_dots)
        plt.plot(x[:, 1], '--')
        plt.ylabel('phase dot')
        #plt.ylim(phase_dots.min()-0.2, phase_dots.max() +0.2)
        plt.subplot(413)
        plt.plot(step_lengths)
        plt.plot(x[:, 2], '--')
        #plt.ylim(step_lengths.min()-0.02, step_lengths.max() +0.02)
        plt.ylabel('step length')
        plt.subplot(414)
        plt.plot(ramps)
        plt.plot(x[:, 3], '--')
        plt.ylabel('ramp')
        plt.show()
    
    return track, RMSE_phase

def pf_robustness(kidnap = True, RMSE_heatmap = False):    
    track_count = 0
    total_trials = 0
    RMSerror_phase = np.zeros((10, 27))
    s = 0
    for subject in Conti_subject_names():
    #for subject in ['AB05']:
        print("subject: ", subject)
        t = 0
        for trial in Conti_trial_names(subject):
            if trial == 'subjectdetails':
                continue
            print("trial: ", trial)
            for side in ['left']:
                #print("side: ", side)
                total_trials = total_trials + 1
                track, RMSE_phase = pf_test(subject, trial, side, kidnap = True, plot = False)
                if  track == True:
                    track_count = track_count + 1
                RMSerror_phase[s, t] = RMSE_phase
            t += 1
        s += 1

    with open('RMSE_phase_PF.pickle', 'wb') as file:
        pickle.dump(RMSerror_phase, file)

    if RMSE_heatmap:
        plt.imshow(RMSerror_phase)
        plt.title("RMSE of phase")
        plt.xlabel("trials")
        plt.ylabel("subjects")
        plt.colorbar()
        plt.show()
    robustness = track_count / total_trials * 100
    return robustness

if __name__ == '__main__':
    subject = 'AB05'
    trial= 's1x2d2x5'
    side = 'left'

    pf_test(subject, trial, side, kidnap = True, plot = True)

    #Q = np.diag([1e-14, 1e-7, 1e-7, 5e-5]) # process model noise covariance
    #print("Q =\n", Q)
    #robustness = ekf_robustness(Q, kidnap = False, RMSE_heatmap = True)
    #print("robustness (%): ", robustness)




