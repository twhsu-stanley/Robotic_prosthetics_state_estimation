from model_framework import *
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def wrapTo2pi(ang):
    ang = ang % (2*np.pi)
    return ang

def virtual_training_data(n, gait_data):
    # Generate virtual training data that cause measurements go to zero as phase rate and stride langth approaches zero
    # n: number of vertual strides
    
    data_virtual_1 = np.zeros((n, 150))
    phase_virtual_1 = np.tile(np.linspace(0, 1, 150), (n,1))
    phase_dot_virtual_1 = np.zeros((n, 150))
    step_length_virtual_1 = np.zeros((n, 150))
    ramp_virtual_1 = np.zeros((n, 150))

    if gait_data == 'globalThighAngles' or gait_data == 'kneeAngles' or gait_data == 'ankleAngles' or gait_data == 'globalFootAngles':
        data_virtual = data_virtual_1
        phase_virtual = phase_virtual_1
        phase_dot_virtual = phase_dot_virtual_1
        step_length_virtual = step_length_virtual_1
        ramp_virtual = ramp_virtual_1
    
    elif gait_data == 'globalThighVelocities':
        data_virtual_2 = np.zeros((n, 150))
        phase_virtual_2 = np.tile(np.linspace(0, 1, 150), (n,1))
        phase_dot_virtual_2 = np.ones((n, 150))
        step_length_virtual_2 = np.zeros((n, 150))
        ramp_virtual_2 = np.zeros((n, 150))

        #data_virtual_3 = np.zeros((n, 150))
        #phase_virtual_3 = np.tile(np.linspace(0, 1, 150), (n,1))
        #phase_dot_virtual_3 = np.zeros((n, 150))
        #step_length_virtual_3 = np.ones((n, 150))
        #ramp_virtual_3 = np.zeros((n, 150))
        
        data_virtual = np.vstack((data_virtual_1, data_virtual_2)) #, data_virtual_3
        phase_virtual = np.vstack((phase_virtual_1, phase_virtual_2)) #, phase_virtual_3
        phase_dot_virtual = np.vstack((phase_dot_virtual_1, phase_dot_virtual_2)) #, phase_dot_virtual_3
        step_length_virtual = np.vstack((step_length_virtual_1, step_length_virtual_2)) #, step_length_virtual_3
        ramp_virtual = np.vstack((ramp_virtual_1, ramp_virtual_2)) #, ramp_virtual_3
    
    return (data_virtual, phase_virtual, phase_dot_virtual, step_length_virtual, ramp_virtual)

def basis_model_fitting(model, gait_data):
    # Generic basis model fitting using gait training data

    # Input:
    #   model: basis model to be fitted
    #   gait_data : training data, e.g. 'kneeAngles', 'ankleAngles'
    
    ## InclineExp dataser
    with open(('Gait_training_data_incExp/' + gait_data + '_training_dataset.pickle'), 'rb') as file:
        gait_training_dataset = pickle.load(file)
    
    data = gait_training_dataset['training_data']
    phase = gait_training_dataset['phase']
    phase_dot = gait_training_dataset['phase_dot']
    step_length = gait_training_dataset['step_length']
    ramp = gait_training_dataset['ramp']

    print("Shape of data: ", np.shape(data))
    print("Shape of phase: ", np.shape(phase))
    print("Shape of phase dot: ", np.shape(phase_dot))
    print("  Range of phase dot: [%5.3f, %5.3f]" % (np.min(gait_training_dataset['phase_dot'].ravel()), np.max(gait_training_dataset['phase_dot'].ravel())))
    print("Shape of step length: ", np.shape(step_length))
    print("  Range of step length: [%5.3f, %5.3f]" % (np.min(gait_training_dataset['step_length'].ravel()), np.max(gait_training_dataset['step_length'].ravel())))
    print("Shape of ramp: ", np.shape(ramp))
    
    #plt.figure()
    #plt.plot(phase_dot.ravel(), step_length.ravel(), 'r.')

    ## R01 dataset
    """
    with open(('Gait_training_data_R01/' + gait_data + '_walking_training_dataset.pickle'), 'rb') as file:
        gait_training_dataset = pickle.load(file)
    
    data = np.vstack((data, gait_training_dataset['training_data']))
    phase = np.vstack((phase, gait_training_dataset['phase']))
    phase_dot = np.vstack((phase_dot, gait_training_dataset['phase_dot']))
    step_length = np.vstack((step_length, gait_training_dataset['step_length']))
    ramp = np.vstack((ramp, gait_training_dataset['ramp']))

    print("Shape of data: ", np.shape(data))
    print("Shape of phase: ", np.shape(phase))
    print("Shape of phase dot: ", np.shape(phase_dot))
    print("  Range of phase dot: [%5.3f, %5.3f]" % (np.min(gait_training_dataset['phase_dot'].ravel()), np.max(gait_training_dataset['phase_dot'].ravel())))
    print("Shape of step length: ", np.shape(step_length))
    print("  Range of step length: [%5.3f, %5.3f]" % (np.min(gait_training_dataset['step_length'].ravel()), np.max(gait_training_dataset['step_length'].ravel())))
    print("Shape of ramp: ", np.shape(ramp))
    """

    # Test plot: all data
    #plt.figure()
    #plt.plot(np.arange(150), data.T)
    #plt.xlabel('normalized time (1/150)')
    #plt.ylabel('data')
    #plt.show()
    """
    if gait_data != 'atan2':
        (data_virtual, phase_virtual, phase_dot_virtual, step_length_virtual, ramp_virtual) = virtual_training_data(2000, gait_data)
        data = np.vstack((data, data_virtual))
        phase = np.vstack((phase, phase_virtual))
        phase_dot = np.vstack((phase_dot, phase_dot_virtual))
        step_length = np.vstack((step_length, step_length_virtual))
        ramp = np.vstack((ramp, ramp_virtual))

        print("Shape of data: ", np.shape(data))
        print("Shape of phase: ", np.shape(phase))
        print("Shape of phase dot: ", np.shape(phase_dot))
        print("Shape of step length: ", np.shape(step_length))
        print("Shape of ramp: ", np.shape(ramp))
    """

    # Fit the model =====================================================================================================
    if gait_data == 'atan2':
        data_atan2 = data.ravel() - 2 * np.pi * phase.ravel()
        # wrap to [-pi, pi]
        for i in range(np.shape(data_atan2)[0]):
            data_atan2[i] = np.arctan2(np.sin(data_atan2[i]), np.cos(data_atan2[i]))
        Psi = least_squares(model, data_atan2,\
                            phase.ravel(), phase_dot.ravel(), step_length.ravel(), ramp.ravel())
    else:
        Psi = least_squares(model, data.ravel(),\
                            phase.ravel(), phase_dot.ravel(), step_length.ravel(), ramp.ravel())
    
    with open('Psi/Psi_' + gait_data, 'wb') as file:
        pickle.dump(Psi, file)

    print("Finished fitting the basis model for " + gait_data + "!")

    return Psi

def basis_model_residuals(model, gait_data, heteroscedastic = False):
    with open(('Gait_training_data_incExp/' + gait_data + '_training_dataset.pickle'), 'rb') as file:
        gait_training_dataset = pickle.load(file)
    
    data = gait_training_dataset['training_data']
    phase = gait_training_dataset['phase']
    phase_dot = gait_training_dataset['phase_dot']
    step_length = gait_training_dataset['step_length']
    ramp = gait_training_dataset['ramp']

    with open(('Gait_training_data_R01/' + gait_data + '_walking_training_dataset.pickle'), 'rb') as file:
        gait_training_dataset = pickle.load(file)
    
    data = np.vstack((data, gait_training_dataset['training_data']))
    phase = np.vstack((phase, gait_training_dataset['phase']))
    phase_dot = np.vstack((phase_dot, gait_training_dataset['phase_dot']))
    step_length = np.vstack((step_length, gait_training_dataset['step_length']))
    ramp = np.vstack((ramp, gait_training_dataset['ramp']))

    print("Shape of data: ", np.shape(data))
    print("Shape of phase: ", np.shape(phase))
    print("Shape of phase dot: ", np.shape(phase_dot))
    print("  Range of phase dot: [%5.3f, %5.3f]" % (np.min(gait_training_dataset['phase_dot'].ravel()), np.max(gait_training_dataset['phase_dot'].ravel())))
    print("Shape of step length: ", np.shape(step_length))
    print("  Range of step length: [%5.3f, %5.3f]" % (np.min(gait_training_dataset['step_length'].ravel()), np.max(gait_training_dataset['step_length'].ravel())))
    print("Shape of ramp: ", np.shape(ramp))

    if gait_data == 'atan2':
        with open('Psi/Psi_' + gait_data, 'rb') as file:
            Psi = pickle.load(file)
        
        if heteroscedastic == False:
            data_pred = model_prediction(model, Psi, phase.ravel(), phase_dot.ravel(), step_length.ravel(), ramp.ravel()) + 2*np.pi*phase.ravel()
            data_pred = wrapTo2pi(data_pred)
            residuals = data.ravel() - data_pred
            residuals = np.arctan2(np.sin(residuals), np.cos(residuals))
            with open(('Basis_model/' + gait_data + '_residuals.pickle'), 'wb') as file:
                pickle.dump(residuals, file)
        
        elif heteroscedastic == True:
            # Heteroscedastic noise model
            residuals = np.zeros(np.shape(data))
            for i in range(np.shape(data)[0]):
                data_pred = model_prediction(model, Psi, phase[i, :], phase_dot[i, :], step_length[i, :], ramp[i, :])
                data_pred = wrapTo2pi(data_pred)
                residuals[i, :] = data[i, :] - data_pred
                residuals[i, :] = np.arctan2(np.sin(residuals[i, :]), np.cos(residuals[i, :]))
            with open(('Basis_model/' + gait_data + '_hetero_residuals.pickle'), 'wb') as file:
                pickle.dump(residuals, file)

    elif gait_data == 'ankleMoment' or gait_data == 'tibiaForce':
        with open('Psi/Psi_' + gait_data, 'rb') as file:
            Psi = pickle.load(file)
        
        if heteroscedastic == False:
            # residuals during stance
            residuals = []
            for i in range(np.shape(data)[0]):
                data_pred = model_prediction(model, Psi, phase[i, 0:60], phase_dot[i, 0:60], step_length[i, 0:60], ramp[i, 0:60])
                residuals.append(data[i, 0:60] - data_pred)
            residuals = np.array(residuals)
            residuals = residuals.ravel()
            with open(('Basis_model/' + gait_data + '_stance_residuals.pickle'), 'wb') as file:
                pickle.dump(residuals, file)
        
        elif heteroscedastic == True:
            # Heteroscedastic noise model
            residuals = np.zeros(np.shape(data))
            for i in range(np.shape(data)[0]):
                data_pred = model_prediction(model, Psi, phase[i, :], phase_dot[i, :], step_length[i, :], ramp[i, :])
                residuals[i, :] = data[i, :] - data_pred
            with open(('Basis_model/' + gait_data + '_hetero_residuals.pickle'), 'wb') as file:
                pickle.dump(residuals, file)

    elif gait_data == 'globalFootAngles':
        with open('Psi/Psi_' + gait_data, 'rb') as file:
            Psi = pickle.load(file)
        
        if heteroscedastic == False:
            # residuals during stance
            residuals = []
            for i in range(np.shape(data)[0]):
                data_pred = model_prediction(model, Psi, phase[i, 0:60], phase_dot[i, 0:60], step_length[i, 0:60], ramp[i, 0:60])
                residuals.append(data[i, 0:60] - data_pred)
            residuals = np.array(residuals)
            residuals = residuals.ravel()
            with open(('Basis_model/' + gait_data + '_stance_residuals.pickle'), 'wb') as file:
                pickle.dump(residuals, file)
        
        elif heteroscedastic == True:
            # Heteroscedastic noise model
            residuals = np.zeros(np.shape(data))
            for i in range(np.shape(data)[0]):
                data_pred = model_prediction(model, Psi, phase[i, :], phase_dot[i, :], step_length[i, :], ramp[i, :])
                residuals[i, :] = data[i, :] - data_pred
            with open(('Basis_model/' + gait_data + '_hetero_residuals.pickle'), 'wb') as file:
                pickle.dump(residuals, file)
    
    else:
        with open('Psi/Psi_' + gait_data, 'rb') as file:
            Psi = pickle.load(file)
        
        if heteroscedastic == False:
            # Non-heteroscedastic noise model
            data_pred = model_prediction(model, Psi, phase.ravel(), phase_dot.ravel(), step_length.ravel(), ramp.ravel())
            residuals = data.ravel() - data_pred
            with open(('Basis_model/' + gait_data + '_const_residuals.pickle'), 'wb') as file:
                pickle.dump(residuals, file)

        elif heteroscedastic == True:
            # Heteroscedastic noise model
            residuals = np.zeros(np.shape(data))
            for i in range(np.shape(data)[0]):
                data_pred = model_prediction(model, Psi, phase[i, :], phase_dot[i, :], step_length[i, :], ramp[i, :])
                residuals[i, :] = data[i, :] - data_pred
            with open(('Basis_model/' + gait_data + '_hetero_residuals.pickle'), 'wb') as file:
                pickle.dump(residuals, file)

def measurement_noise_covariance(*sensors):
    covariance = []
    for sensor in sensors:
        if sensor == 'globalThighAngles' or sensor == 'globalThighVelocities':
            with open(('Basis_model/' + sensor + '_const_residuals.pickle'), 'rb') as file:
                r = pickle.load(file)
                covariance.append(np.cov(r))
        elif sensor == 'atan2':
            with open(('Basis_model/' + sensor + '_residuals.pickle'), 'rb') as file:
                r = pickle.load(file)
                covariance.append(np.cov(r))
        elif sensor == 'ankleMoment' or sensor == 'tibiaForce':
            with open(('Basis_model/' + sensor + '_stance_residuals.pickle'), 'rb') as file:
                r = pickle.load(file)
                covariance.append(np.cov(r))
        elif sensor == 'globalFootAngles':
            with open(('Basis_model/' + sensor + '_stance_residuals.pickle'), 'rb') as file:
                r = pickle.load(file)
                covariance.append(np.cov(r))
    R = np.diag(covariance)
    if len(covariance) != len(sensors):
        print("ERROR: SIZE NOT MATCH!!")
    return R

def heteroscedastic_measurement_noise_covariance(*sensors):
    covariance = np.zeros((len(sensors), 150))
    R= []
    sn = 0
    for sensor in sensors:
        if sensor == 'atan2':
            with open(('Basis_model/' + sensor + '_hetero_residuals.pickle'), 'rb') as file:
                r = pickle.load(file)
        elif sensor == 'globalFootAngles':
            with open(('Basis_model/' + sensor + '_hetero_residuals.pickle'), 'rb') as file:
                r = pickle.load(file)
        else:
            with open(('Basis_model/' + sensor + '_hetero_residuals.pickle'), 'rb') as file:
                r = pickle.load(file)
        
        for i in range(150):
            covariance[sn, i] = np.cov(r[:, i])
        sn += 1
    
    #plt.figure()
    #plt.plot(range(150), covariance.T)
    #plt.show()

    #for i in range(150):
    #    R.append(np.diag(covariance[:, i]))
    
    return covariance
    #return R

def saturation_bounds():
    with open(('Gait_training_data_incExp/globalThighAngles_training_dataset.pickle'), 'rb') as file:
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

    nu = 3
    saturation_range = np.array([min(phase_dots_sup, phase_dots_mean + nu * phase_dots_std),\
                                 max(phase_dots_inf, phase_dots_mean - nu * phase_dots_std),\
                                 min(step_lengths_sup, step_lengths_mean + nu * step_lengths_std),\
                                 max(step_lengths_inf, step_lengths_mean - nu * step_lengths_std)])
    print("Saturation bounds: ", saturation_range)
    return saturation_range
    
if __name__ == '__main__': 
    #saturation_bounds()
    #sensors = ['globalThighAngles', 'globalThighVelocities', 'atan2', 'globalFootAngles', 'ankleMoment', 'tibiaForce']
    #print(np.diag(measurement_noise_covariance(*sensors)))
    #sensors = ['tibiaForce']
    #heteroscedastic_measurement_noise_covariance(*sensors)
    #F_test('globalThighAngles', 1, 2)
    #F_test('globalThighAngles', 2, 3)
    #F_test('globalThighVelocities', 1, 2)
    #F_test('globalThighVelocities', 2, 3)
    
    phase_model = Fourier_Basis(11, 'phase')
    phase_dot_model = Polynomial_Basis(0, 'phase_dot')
    step_length_model = Berstein_Basis(2,'step_length')
    ramp_model = Berstein_Basis(2, 'ramp')
    model_globalThighAngles = Kronecker_Model(phase_model, phase_dot_model, step_length_model, ramp_model)
    psi_globalThighAngles = basis_model_fitting(model_globalThighAngles, 'globalThighAngles')
    #basis_model_residuals(model_globalThighAngles, 'globalThighAngles', heteroscedastic = False)

    #model_ankleMoment = Kronecker_Model(phase_model, phase_dot_model, step_length_model)#, ramp_model)
    #psi_ankleMoment = basis_model_fitting(model_ankleMoment, 'ankleMoment')
    #basis_model_residuals(model_ankleMoment, 'ankleMoment', heteroscedastic = True)

    #model_tibiaForce = Kronecker_Model(phase_model, phase_dot_model, step_length_model)#, ramp_model)
    #psi_tibiaForce = basis_model_fitting(model_tibiaForce, 'tibiaForce')
    #basis_model_residuals(model_tibiaForce, 'tibiaForce', heteroscedastic = True)

    model_kneeAngles = Kronecker_Model(phase_model, phase_dot_model, step_length_model, ramp_model)
    #psi_kneeAngles = basis_model_fitting(model_kneeAngles, 'kneeAngles')

    model_ankleAngles = Kronecker_Model(phase_model, phase_dot_model, step_length_model, ramp_model)
    #psi_ankleAngles = basis_model_fitting(model_ankleAngles, 'ankleAngles')
    
    ## 
    phase_model = Fourier_Basis(11, 'phase')
    phase_dot_model = Polynomial_Basis(1, 'phase_dot')
    step_length_model = Berstein_Basis(2,'step_length')
    ramp_model = Berstein_Basis(2, 'ramp')
    model_globalThighVelocities = Kronecker_Model(phase_model, phase_dot_model, step_length_model, ramp_model)
    psi_globalThighVelocities = basis_model_fitting(model_globalThighVelocities, 'globalThighVelocities')
    #basis_model_residuals(model_globalThighVelocities, 'globalThighVelocities', heteroscedastic = False)
    
    ##
    phase_model = Fourier_Basis(11, 'phase')
    phase_dot_model = Polynomial_Basis(0, 'phase_dot')
    step_length_model = Berstein_Basis(2,'step_length')
    ramp_model = Berstein_Basis(2, 'ramp')
    model_globalFootAngles = Kronecker_Model(phase_model, phase_dot_model, step_length_model, ramp_model)
    psi_globalFootAngles = basis_model_fitting(model_globalFootAngles, 'globalFootAngles')
    #basis_model_residuals(model_globalFootAngles, 'globalFootAngles', heteroscedastic = False)

    # Atan2 fitting
    phase_model = Fourier_Basis(11, 'phase')
    phase_dot_model = Polynomial_Basis(0, 'phase_dot')
    step_length_model = Berstein_Basis(0,'step_length')
    ramp_model = Berstein_Basis(0, 'ramp')
    model_atan2 = Kronecker_Model(phase_model, phase_dot_model, step_length_model, ramp_model)
    psi_atan2 = basis_model_fitting(model_atan2, 'atan2')
    #basis_model_residuals(model_atan2, 'atan2', heteroscedastic = False)

    ## Store measurement models
    m_model = Measurement_Model(model_globalThighAngles, model_globalThighVelocities, model_atan2, model_globalFootAngles)
    model_saver(m_model, 'Measurement_model_globalThighAngles_globalThighVelocities_atan2_globalFootAngles.pickle')

    m_model = Measurement_Model(model_globalThighAngles, model_globalThighVelocities, model_globalFootAngles)
    model_saver(m_model, 'Measurement_model_globalThighAngles_globalThighVelocities_globalFootAngles.pickle')

    m_model = Measurement_Model(model_globalThighAngles, model_globalThighVelocities, model_atan2)
    model_saver(m_model, 'Measurement_model_globalThighAngles_globalThighVelocities_atan2.pickle')

    m_model = Measurement_Model(model_globalThighAngles, model_globalThighVelocities)
    model_saver(m_model, 'Measurement_model_globalThighAngles_globalThighVelocities.pickle')

    c_model = Measurement_Model (model_kneeAngles, model_ankleAngles)
    model_saver(c_model, 'Control_model_kneeAngles_ankleAngles.pickle')
    