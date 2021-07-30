from model_framework import *
import matplotlib.pyplot as plt
import numpy as np

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

    with open('New_Psi/Psi_' + mode + '.pickle', 'wb') as file:
        pickle.dump(Psi, file)

    print("Finished fitting the basis model!")

    return Psi

if __name__ == '__main__':
    
    F = 11
    N = 3
    phase_model = Fourier_Basis(F, 'phase')
    phase_dot_model = Polynomial_Basis(1, 'phase_dot')
    step_length_model = Berstein_Basis(N,'step_length')
    ramp_model = Berstein_Basis(N, 'ramp')

    model_globalThighAngles = Kronecker_Model(phase_model, phase_dot_model, step_length_model, ramp_model)
    #psi_globalThighAngles = basis_model_fitting(model_globalThighAngles, 'globalThighAngles')

    phase_dot_model = Polynomial_Basis(2, 'phase_dot')
    model_globalThighVelocities = Kronecker_Model(phase_model, phase_dot_model, step_length_model, ramp_model)
    #psi_globalThighAngles = basis_model_fitting(model_globalThighVelocities, 'globalThighVelocities')

    #model_kneeAngles = Kronecker_Model(phase_model, phase_dot_model, step_length_model, ramp_model)
    #psi_kneeAngles = basis_model_fitting(model_kneeAngles, 'kneeAngles')

    #model_ankleAngles = Kronecker_Model(phase_model, phase_dot_model, step_length_model, ramp_model)
    #psi_ankleAngles = basis_model_fitting(model_ankleAngles, 'ankleAngles')

    #model_ankleMoment = Kronecker_Model(phase_model, phase_dot_model, step_length_model, ramp_model)
    #psi_ankleMoment = basis_model_fitting(model_ankleMoment, 'ankleMoment')

    # Atan2 fitting
    phase_model = Fourier_Basis(F, 'phase')
    phase_dot_model = Polynomial_Basis(1, 'phase_dot')
    step_length_model = Berstein_Basis(0,'step_length')
    ramp_model = Berstein_Basis(0, 'ramp')

    model_atan2 = Kronecker_Model(phase_model, phase_dot_model, step_length_model, ramp_model)
    #psi_atan2 = basis_model_fitting(model_atan2, 'atan2')

    m_model = Measurement_Model(model_globalThighAngles, model_globalThighVelocities, model_atan2)
    model_saver(m_model, 'Measurement_model_3.pickle')
    