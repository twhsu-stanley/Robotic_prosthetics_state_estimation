from data_generators import *
from model_framework import *
import matplotlib.pyplot as plt
import numpy as np
import h5py

##Get the names of all the subjects
subject_names = get_subject_names()

def model_fit(model, mode):
    #mode: input data 
        #mode = 'global_thigh_angle_X/Y/Z'
        #mode = 'reaction_force_x/y/z_ankle'
        #mode = 'reaction_moment_x/y/z_ankle'

    # dictionary of Fourier coefficients: psi
    PSI = dict()

    # dictionary of  RMS error: rmse
    RMSE = dict()
    
    # Calculate a SUBJECT SPECIFIC model
    #for subject in ['AB01', 'AB03']:
    for subject in subject_names:
        # generate data
        print("Doing subject: " + subject)
        if mode == 'global_thigh_angle_Y':
            with open('Global_thigh_angle.npz', 'rb') as file:
                g_t = np.load(file)
                measurement_input = g_t[subject][0]
        elif mode =='reaction_force_z_ankle':
            with open('Reaction_wrench.npz', 'rb') as file:
                r_w = np.load(file)
                measurement_input = r_w[subject][2]
        elif mode =='reaction_force_y_ankle':
            with open('Reaction_wrench.npz', 'rb') as file:
                r_w = np.load(file)
                measurement_input = r_w[subject][1]
        elif mode =='reaction_force_x_ankle':
            with open('Reaction_wrench.npz', 'rb') as file:
                r_w = np.load(file)
                measurement_input = r_w[subject][0]
        elif mode =='reaction_moment_z_ankle':
            with open('Reaction_wrench.npz', 'rb') as file:
                r_w = np.load(file)
                measurement_input = r_w[subject][5]
        elif mode =='reaction_moment_y_ankle':
            with open('Reaction_wrench.npz', 'rb') as file:
                r_w = np.load(file)
                measurement_input = r_w[subject][4]
        elif mode =='reaction_moment_x_ankle':
            with open('Reaction_wrench.npz', 'rb') as file:
                r_w = np.load(file)
                measurement_input = r_w[subject][3]
        else:
            sys.exit('Error: no such mode of input')
        
        phases = get_phase(measurement_input)
        phase_dots = get_phase_dot(subject)
        step_lengths = get_step_length(subject)
        ramps = get_ramp(subject)
        print("Obtained data for: " + subject)

        # Fit the model for the subject
        psi= least_squares(model, measurement_input.ravel(), phases.ravel(), phase_dots.ravel(), step_lengths.ravel(), ramps.ravel())
        PSI[subject] = psi

        # RMSE of each stride
        rmse = np.zeros((np.shape(measurement_input)[0], 1))
        for i in range(np.shape(measurement_input)[0]):
            measurement_pred = model_prediction(model, psi, phases[i,:], phase_dots[i,:], step_lengths[i,:], ramps[i,:])
            mse = np.square(np.subtract(measurement_input[i,:], measurement_pred)).mean()
            rmse[i] = np.sqrt(mse)
        
        RMSE[subject] = rmse
        #print("mode: ", str(mode), "; Subject: ", str(subject))
        print("RMSE mean: ", rmse.mean())
        print("RMSE max: ", rmse.max())

    return PSI, RMSE

def load_Psi(subject):
    with open('Measurement_model_coeff.npz', 'rb') as file:
        Measurement_model_coeff = np.load(file, allow_pickle = True)
        psi_thigh_Y = Measurement_model_coeff['global_thigh_angle_Y']
        psi_force_z = Measurement_model_coeff['reaction_force_z_ankle']
        psi_force_x = Measurement_model_coeff['reaction_force_x_ankle']
        psi_moment_y = Measurement_model_coeff['reaction_moment_y_ankle']
    Psi = np.array([psi_thigh_Y.item()[subject],\
                    psi_force_z.item()[subject],\
                    psi_force_x.item()[subject],\
                    psi_moment_y.item()[subject]]) # Psi: 4 x 336
    return Psi

def measurement_error_cov(subject):
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

    Psi = load_Psi(subject)

    global_thigh_angle_Y_pred = model_prediction(m_model.models[0], Psi[0], phases.ravel(), phase_dots.ravel(), step_lengths.ravel(), ramps.ravel())
    force_z_ankle_pred = model_prediction(m_model.models[1], Psi[1], phases.ravel(), phase_dots.ravel(), step_lengths.ravel(), ramps.ravel())
    force_x_ankle_pred = model_prediction(m_model.models[2], Psi[2], phases.ravel(), phase_dots.ravel(), step_lengths.ravel(), ramps.ravel())
    moment_y_ankle_pred = model_prediction(m_model.models[3], Psi[3], phases.ravel(), phase_dots.ravel(), step_lengths.ravel(), ramps.ravel())

    # compute covariance from samples
    err_gthY = global_thigh_angle_Y.ravel() - global_thigh_angle_Y_pred
    print("subject: ",  subject)
    print("mean g_th_Y ", np.mean(err_gthY))
    print("std g_th_Y ", np.std(err_gthY))
    err_fz = force_z_ankle.ravel() - force_z_ankle_pred
    print("mean f_z ", np.mean(err_fz))
    print("std f_z ", np.std(err_fz))
    err_fx = force_x_ankle.ravel() - force_x_ankle_pred
    print("mean f_x ", np.mean(err_fx))
    print("std f_x ", np.std(err_fx))
    err_my = moment_y_ankle.ravel() - moment_y_ankle_pred
    print("mean m_y ", np.mean(err_my))
    print("std m_y ", np.std(err_my))
    err = np.stack((err_gthY, err_fz, err_fx, err_my))
    R = np.cov(err)
    print("R = ", R)

    return R

if __name__ == '__main__':
    """
    # dictionary storing all measurement model coefficients
    Measurement_model_coeff = dict()
    Measurement_model_RMSE = dict()

    # Orders of the measurement model
    F = 11
    N = 3

    phase_model = Fourier_Basis(F, 'phase')
    phase_dot_model = Polynomial_Basis(1, 'phase_dot')
    step_length_model = Berstein_Basis(N,'step_length')
    ramp_model = Berstein_Basis(N, 'ramp')

    # Measrurement model for global_thigh_angle_Y
    model_thigh_Y = Kronecker_Model(phase_model, phase_dot_model, step_length_model, ramp_model)
    psi_thigh_Y, RMSE_thigh_Y = model_fit(model_thigh_Y, 'global_thigh_angle_Y')
    Measurement_model_coeff['global_thigh_angle_Y'] = psi_thigh_Y
    Measurement_model_RMSE['global_thigh_angle_Y'] = RMSE_thigh_Y

    # Measrurement model for reaction_force_z_ankle
    model_force_z = Kronecker_Model(phase_model, phase_dot_model, step_length_model, ramp_model)
    psi_force_z, RMSE_force_z = model_fit(model_force_z, 'reaction_force_z_ankle')
    Measurement_model_coeff['reaction_force_z_ankle'] = psi_force_z
    Measurement_model_RMSE['reaction_force_z_ankle'] = RMSE_force_z

    # Measrurement model for reaction_force_x_ankle
    model_force_x = Kronecker_Model(phase_model, phase_dot_model, step_length_model, ramp_model)
    psi_force_x, RMSE_force_x = model_fit(model_force_x, 'reaction_force_x_ankle')
    Measurement_model_coeff['reaction_force_x_ankle'] = psi_force_x
    Measurement_model_RMSE['reaction_force_x_ankle'] = RMSE_force_x

    # Measrurement model for reaction_moment_y_ankle
    model_moment_y = Kronecker_Model(phase_model, phase_dot_model, step_length_model, ramp_model)
    psi_moment_y, RMSE_moment_y = model_fit(model_moment_y, 'reaction_moment_y_ankle')
    Measurement_model_coeff['reaction_moment_y_ankle'] = psi_moment_y
    Measurement_model_RMSE['reaction_moment_y_ankle'] = RMSE_moment_y

    # save measurement model coeffiecients (Psi)
    with open('Measurement_model_coeff.npz', 'wb') as file:
        np.savez(file, **Measurement_model_coeff, allow_pickle = True)

    # save RMSE
    with open('Measurement_model_RMSE.npz', 'wb') as file:
        np.savez(file, **Measurement_model_RMSE, allow_pickle = True)

    # save measurement model
    m_model = Measurement_Model(model_thigh_Y, model_force_z, model_force_x, model_moment_y)
    model_saver(m_model, 'Measurement_model.pickle')
    """
    
    R = dict()
    for subject in subject_names:
        R[subject] = measurement_error_cov(subject)
        
    with open('Measurement_error_cov.pickle', 'wb') as file:
    	pickle.dump(R, file)
