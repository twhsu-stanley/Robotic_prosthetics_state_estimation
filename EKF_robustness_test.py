import numpy as np
from EKF import *
from model_framework import *
from data_generators import *
from continuous_data import *
from model_fit import *

with open('Measurement_error_cov.pickle', 'rb') as file:
    R = pickle.load(file)

m_model = model_loader('Measurement_model.pickle')

def ekf_kidnap_test(subject, trial, side, ekf, kidnap = True):
    dt = 1/100
    Psi = load_Psi(subject)
    phases, _, _, _ = Conti_state_vars(subject, trial, side)
    global_thigh_angle_Y, force_z_ankle, force_x_ankle, moment_y_ankle = load_Conti_measurement_data(subject, trial, side)

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

    x = []  # state estimate
    for i in range(np.shape(z)[1]):
        # kidnap
        if kidnap == True and i == kidnap_index:
            ekf.x = state_kidnap

        ekf.prediction(dt)
        ekf.correction(z[:, i], Psi)
        x.append(ekf.x)
    x = np.array(x).squeeze()

    # evaluate robustness
    # compare x and ground truth:
    track = True
    track_tol = 0.075
    start_check = 4
    se = 0
    for i in range(int(heel_strike_index[0]), int(heel_strike_index[np.size(heel_strike_index)-1])):
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
    
    return track, RMSE_phase

def ekf_robustness(Q, kidnap = True, RMSE_heatmap = False):
    # build the system
    sys = myStruct()
    sys.f = process_model
    sys.A = A
    sys.h = m_model
    sys.Q = Q
                
    # initialize the state
    init = myStruct()
    init.Sigma = np.diag([0, 1e-3, 1e-3, 5e-1])

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
                sys.R = R[subject]
                phases, phase_dots, step_lengths, ramps = Conti_state_vars(subject, trial, side)
                init.x = np.array([[phases[0]], [phase_dots[0]], [step_lengths[0]], [ramps[0]]])
                ekf = extended_kalman_filter(sys, init)

                total_trials = total_trials + 1
                track, RMSE_phase = ekf_kidnap_test(subject, trial, side, ekf, kidnap)
                if  track == True:
                    track_count = track_count + 1
                RMSerror_phase[s, t] = RMSE_phase
            t += 1
        s += 1

    with open('RMSE_phase.pickle', 'wb') as file:
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
    Q_best = np.diag([0, 0, 0, 0])
    robustness = 0
    # iterate through Q -5 -5 -4
    for Q_phase_dot in [1e-7]:
        for Q_step_length in [1e-7]:
            for Q_ramp in [5e-5]:
                Q = np.diag([0, Q_phase_dot, Q_step_length, Q_ramp]) # process model noise covariance
                print("Q =\n", Q)
                rob = ekf_robustness(Q, kidnap = False, RMSE_heatmap = True)
                if rob > robustness:
                    robustness = rob
                    Q_best = Q

print("Q best =\n", Q_best)
print("robustness (%): ", robustness)

#with open('Process_error_cov.pickle', 'wb') as file:
    #pickle.dump(Q_best, file)
