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
    """ TAKE EXTRA CARE IN ATAN2 DATA ***********************
    if mode == 'atan2':
        measurement = measurement_input.ravel() - 2 * np.pi * phases.ravel()
        # wrap to [-pi, pi]
        for i in range(np.shape(measurement)[0]):
            if measurement[i] < -np.pi:
                measurement[i] += 2 * np.pi
            elif measurement[i] > np.pi:
                measurement[i] -= 2 * np.pi
        psi = least_squares(model, measurement,\
                            phases.ravel(), phase_dots.ravel(), step_lengths.ravel(), ramps.ravel())
    """
    
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

    model_knee = Kronecker_Model(phase_model, phase_dot_model, step_length_model, ramp_model)
    psi_knee = basis_model_fitting(model_knee, 'kneeAngles')

    model_ankle = Kronecker_Model(phase_model, phase_dot_model, step_length_model, ramp_model)
    psi_ankle = basis_model_fitting(model_ankle, 'ankleAngles')
    
    #c_model = Measurement_Model(model_knee, model_ankle)
    #model_saver(c_model, 'Control_model.pickle')
    