from data_generators import get_joint_angle, get_reaction_wrench, get_global_thigh_angle, get_phase, get_phase_dot, get_step_length, get_ramp, get_subject_names
from model_framework import Fourier_Basis, Polynomial_Basis, Berstein_Basis, Kronecker_Model, Measurement_Model, least_squares, model_prediction, model_saver, model_loader
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio
pio.renderers.default = "browser"
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

    # Visualization
    """
    fig = go.Figure()
    colors = ['black','green','red','cyan','magenta','yellow','black','white',
            'cadetblue', 'darkgoldenrod', 'darkseagreen', 'deeppink', 'midnightblue']
    color_index = 0
    """
    
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
        
        # Vizualization
        """
        #Predict the average line 
        draw_measurement_input = measurement_input.mean(0)
        draw_phases = phases[0]
        draw_phase_dots = phase_dots.mean(0)
        draw_ramps = ramps.mean(0)
        draw_steps = step_lengths.mean(0)
        #Get prediction
        y_pred = model_prediction(model, psi, draw_phases.ravel(), draw_phase_dots.ravel(), draw_steps.ravel(), draw_ramps.ravel())

        #Plot the result
        fig.add_trace(go.Scatter(x = draw_phases, y = draw_measurement_input,
                                 line = dict(color = colors[color_index], width = 4),
                                 name = subject +' data'))
        fig.add_trace(go.Scatter(x = draw_phases, y = y_pred,
                                 line = dict(color = colors[color_index], width = 4, dash = 'dash'),
                                 name = subject + ' predicted'))
        color_index=(color_index + 1) % len(colors)
        
    #Plot everything
    fig.show()
    """

    return PSI, RMSE

if __name__ == '__main__':
    # dictionary storing all measurement model coefficients
    Measurement_model_coeff = dict()
    Measurement_model_RMSE = dict()

    # Orders of the measurement model
    F = 11
    N = 3

    # Measrurement model for global_thigh_angle_Y
    phase_model = Fourier_Basis(F, 'phase')
    phase_dot_model = Polynomial_Basis(1, 'phase_dot')
    step_length_model = Berstein_Basis(N,'step_length')
    ramp_model = Berstein_Basis(N, 'ramp')
    model_thigh_Y = Kronecker_Model(phase_model, phase_dot_model, step_length_model, ramp_model)
    psi_thigh_Y, RMSE_thigh_Y = model_fit(model_thigh_Y, 'global_thigh_angle_Y')
    Measurement_model_coeff['global_thigh_angle_Y'] = psi_thigh_Y
    Measurement_model_RMSE['global_thigh_angle_Y'] = RMSE_thigh_Y

    # Measrurement model for reaction_force_z_ankle
    phase_model = Fourier_Basis(F, 'phase')
    phase_dot_model = Polynomial_Basis(1, 'phase_dot')
    step_length_model = Berstein_Basis(N,'step_length')
    ramp_model = Berstein_Basis(N, 'ramp')
    model_force_z = Kronecker_Model(phase_model, phase_dot_model, step_length_model, ramp_model)
    psi_force_z, RMSE_force_z = model_fit(model_force_z, 'reaction_force_z_ankle')
    Measurement_model_coeff['reaction_force_z_ankle'] = psi_force_z
    Measurement_model_RMSE['reaction_force_z_ankle'] = RMSE_force_z

    # Measrurement model for reaction_force_x_ankle
    phase_model = Fourier_Basis(F, 'phase')
    phase_dot_model = Polynomial_Basis(1, 'phase_dot')
    step_length_model = Berstein_Basis(N,'step_length')
    ramp_model = Berstein_Basis(N, 'ramp')
    model_force_x = Kronecker_Model(phase_model, phase_dot_model, step_length_model, ramp_model)
    psi_force_x, RMSE_force_x = model_fit(model_force_x, 'reaction_force_x_ankle')
    Measurement_model_coeff['reaction_force_x_ankle'] = psi_force_x
    Measurement_model_RMSE['reaction_force_x_ankle'] = RMSE_force_x

    # Measrurement model for reaction_moment_y_ankle
    phase_model = Fourier_Basis(F, 'phase')
    phase_dot_model = Polynomial_Basis(1, 'phase_dot')
    step_length_model = Berstein_Basis(N,'step_length')
    ramp_model = Berstein_Basis(N, 'ramp')
    model_moment_y = Kronecker_Model(phase_model, phase_dot_model, step_length_model, ramp_model)
    psi_moment_y, RMSE_moment_y = model_fit(model_moment_y, 'reaction_moment_y_ankle')
    Measurement_model_coeff['reaction_moment_y_ankle'] = psi_moment_y
    Measurement_model_RMSE['reaction_moment_y_ankle'] = RMSE_moment_y

    # save measurement model coeffiecients
    with open('Measurement_model_coeff.npz', 'wb') as file:
        np.savez(file, **Measurement_model_coeff, allow_pickle = True)

    # save RMSE
    with open('Measurement_model_RMSE.npz', 'wb') as file:
        np.savez(file, **Measurement_model_RMSE, allow_pickle = True)

    # save measurement model
    m_model = Measurement_Model(model_thigh_Y, model_force_z, model_force_x, model_moment_y)
    model_saver(m_model, 'Measurement_model.pickle')
