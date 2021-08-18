from model_framework import *
import matplotlib.pyplot as plt
import numpy as np
from EKF import wrapTo2pi

def basis_model_fitting(model, mode):
    # Generic basis model fitting using gait training data

    # Input:
    #   model: basis model to be fitted
    #   mode : training data, e.g. 'kneeAngles', 'ankleAngles'
    
    with open(('Gait_training_data/' + mode + '_training_dataset.pickle'), 'rb') as file:
        gait_training_dataset = pickle.load(file)

    data = gait_training_dataset['training_data']
    phase = gait_training_dataset['phase']
    phase_dot = gait_training_dataset['phase_dot']
    step_length = gait_training_dataset['step_length']
    ramp = gait_training_dataset['ramp']

    print("Shape of data: ", np.shape(data))
    print("Shape of phase: ", np.shape(phase))
    print("Shape of phase dot: ", np.shape(phase_dot))
    print("Shape of step length: ", np.shape(step_length))
    print("Shape of ramp: ", np.shape(ramp))

    # Fit the model
    if mode == 'atan2':
        data_atan2 = data.ravel() - 2 * np.pi * phase.ravel()
        # wrap to [-pi, pi]
        for i in range(np.shape(data_atan2)[0]):
            data_atan2[i] = np.arctan2(np.sin(data_atan2[i]), np.cos(data_atan2[i]))
        Psi = least_squares(model, data_atan2,\
                            phase.ravel(), phase_dot.ravel(), step_length.ravel(), ramp.ravel())
    else:
        Psi = least_squares(model, data.ravel(),\
                            phase.ravel(), phase_dot.ravel(), step_length.ravel(), ramp.ravel())

    with open('Psi/Psi_' + mode + '_r.pickle', 'wb') as file:
        pickle.dump(Psi, file)

    print("Finished fitting the basis model!")

    return Psi

def basis_model_residuals(model, mode):
    with open(('Gait_training_data/' + mode + '_training_dataset.pickle'), 'rb') as file:
        gait_training_dataset = pickle.load(file)

    data = gait_training_dataset['training_data']
    phase = gait_training_dataset['phase']
    phase_dot = gait_training_dataset['phase_dot']
    step_length = gait_training_dataset['step_length']
    ramp = gait_training_dataset['ramp']

    print("Shape of data: ", np.shape(data))
    print("Shape of phase: ", np.shape(phase))
    print("Shape of phase dot: ", np.shape(phase_dot))
    print("Shape of step length: ", np.shape(step_length))
    print("Shape of ramp: ", np.shape(ramp))

    with open('Psi/Psi_' + mode + '.pickle', 'rb') as file:
        Psi = pickle.load(file)
    
    if mode == 'atan2':
        data_pred = model_prediction(model, Psi, phase.ravel(), phase_dot.ravel(), step_length.ravel(), ramp.ravel()) + 2*np.pi*phase.ravel()
        data_pred = wrapTo2pi(data_pred)
        residuals = data.ravel() - data_pred
        residuals = np.arctan2(np.sin(residuals), np.cos(residuals))
    else:
        # Non-heteroscedastic noise model
        data_pred = model_prediction(model, Psi, phase.ravel(), phase_dot.ravel(), step_length.ravel(), ramp.ravel())
        residuals = data.ravel() - data_pred

        # Heteroscedastic noise model
        #residuals = np.zeros(np.shape(data))
        #for i in range(np.shape(data)[0]):
        #    data_pred = model_prediction(model, Psi, phase[i, :], phase_dot[i, :], step_length[i, :], ramp[i, :])
        #    residuals[i, :] = data[i, :] - data_pred

    with open(('Basis_model/' + mode + '_residuals.pickle'), 'wb') as file:
    	pickle.dump(residuals, file)

def measurement_noise_covariance(*sensors):
    #residuals = []
    covariance = []
    for sensor in sensors:
        with open(('Basis_model/' + sensor + '_residuals.pickle'), 'rb') as file:
            r = pickle.load(file)
            #residuals.append(r)
            covariance.append(np.cov(r))
    #R = np.cov(residuals)
    R = np.diag(covariance)
    return R

def heteroscedastic_measurement_noise_covariance(*sensors):
    #residuals = []
    covariance = np.zeros((150, 1))
    R = np.zeros((150, len(sensors), len(sensors)))
    for sensor in sensors:
        with open(('Basis_model/' + sensor + '_residuals.pickle'), 'rb') as file:
            r = pickle.load(file)
            for i in range(150):
                covariance[i] = np.cov(r[i, :])
        plt.figure()
        plt.plot(covariance)
        plt.show()

def saturation_bounds():
    with open(('Gait_training_data/globalThighAngles_training_dataset.pickle'), 'rb') as file:
        gait_training_dataset = pickle.load(file)

    phase_dot = gait_training_dataset['phase_dot']
    phase_dots_sup = np.max(phase_dot)
    phase_dots_inf = np.min(phase_dot)
    phase_dots_std = np.std(phase_dot)
    phase_dots_mean = np.average(phase_dot)

    step_length = gait_training_dataset['step_length']
    step_lengths_sup = np.max(step_length)
    step_lengths_inf = np.min(step_length)
    step_lengths_std = np.std(step_length)
    step_lengths_mean = np.average(step_length)

    nu = np.sqrt(6.635)
    saturation_range = np.array([min(phase_dots_sup, phase_dots_mean + nu * phase_dots_std),\
                                 max(phase_dots_inf, phase_dots_mean - nu * phase_dots_std),\
                                 min(step_lengths_sup, step_lengths_mean + nu * step_lengths_std),\
                                 max(step_lengths_inf, step_lengths_mean - nu * step_lengths_std)])
    print("Saturation bounds: ", saturation_range)
    return saturation_range
    
if __name__ == '__main__': 
    #sensors = ['ankleMoment']
    #heteroscedastic_measurement_noise_covariance(*sensors)

    F = 11
    N = 3
    phase_model = Fourier_Basis(F, 'phase')
    phase_dot_model = Polynomial_Basis(1, 'phase_dot')
    step_length_model = Berstein_Basis(N,'step_length')
    ramp_model = Berstein_Basis(N, 'ramp')

    model_globalThighAngles = Kronecker_Model(phase_model, phase_dot_model, step_length_model, ramp_model)
    #psi_globalThighAngles = basis_model_fitting(model_globalThighAngles, 'globalThighAngles')
    #basis_model_residuals(model_globalThighAngles, 'globalThighAngles')

    model_ankleMoment = Kronecker_Model(phase_model, phase_dot_model, step_length_model, ramp_model)
    #psi_ankleMoment = basis_model_fitting(model_ankleMoment, 'ankleMoment')
    basis_model_residuals(model_ankleMoment, 'ankleMoment')

    model_tibiaForce = Kronecker_Model(phase_model, phase_dot_model, step_length_model, ramp_model)
    #psi_tibiaForce = basis_model_fitting(model_tibiaForce, 'tibiaForce')
    #basis_model_residuals(model_tibiaForce, 'tibiaForce')

    model_kneeAngles = Kronecker_Model(phase_model, phase_dot_model, step_length_model, ramp_model)
    #psi_kneeAngles = basis_model_fitting(model_kneeAngles, 'kneeAngles')

    model_ankleAngles = Kronecker_Model(phase_model, phase_dot_model, step_length_model, ramp_model)
    #psi_ankleAngles = basis_model_fitting(model_ankleAngles, 'ankleAngles')

    phase_dot_model = Polynomial_Basis(2, 'phase_dot')
    model_globalThighVelocities = Kronecker_Model(phase_model, phase_dot_model, step_length_model, ramp_model)
    #psi_globalThighAngles = basis_model_fitting(model_globalThighVelocities, 'globalThighVelocities')
    #basis_model_residuals(model_globalThighVelocities, 'globalThighVelocities')
    
    # tibiaForce fitting without step_length
    #phase_model = Fourier_Basis(F, 'phase')
    #phase_dot_model = Polynomial_Basis(1, 'phase_dot')
    #step_length_model = Berstein_Basis(0,'step_length')
    #ramp_model = Berstein_Basis(N, 'ramp')
    #model_tibiaForce = Kronecker_Model(phase_model, phase_dot_model, step_length_model, ramp_model)
    #psi_tibiaForce = basis_model_fitting(model_tibiaForce, 'tibiaForce')
    #basis_model_residuals(model_tibiaForce, 'tibiaForce')


    # Atan2 fitting
    phase_model = Fourier_Basis(F, 'phase')
    phase_dot_model = Polynomial_Basis(1, 'phase_dot')
    step_length_model = Berstein_Basis(0,'step_length')
    ramp_model = Berstein_Basis(0, 'ramp')

    model_atan2 = Kronecker_Model(phase_model, phase_dot_model, step_length_model, ramp_model)
    #psi_atan2 = basis_model_fitting(model_atan2, 'atan2')
    #basis_model_residuals(model_atan2, 'atan2')

    # sensors_dict = {'globalThighAngles': 0, 'globalThighVelocities': 1, 'ankleMoment': 2, 'tibiaForce':3,  'atan2': 4}
    #m_model = Measurement_Model(model_globalThighAngles, model_globalThighVelocities, model_ankleMoment, model_tibiaForce, model_atan2)
    #model_saver(m_model, 'Measurement_model_01234.pickle')
    