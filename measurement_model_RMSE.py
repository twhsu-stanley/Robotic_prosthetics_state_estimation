import numpy as np
import matplotlib.pyplot as plt
from data_generators import *
from model_framework import *

def Measurement_model_RMSE(mode):
    m_model = model_loader('Measurement_model.pickle')
    with open('Measurement_model_coeff.npz', 'rb') as file:
        Measurement_model_coeff = np.load(file, allow_pickle = True)
        psi = Measurement_model_coeff[mode]
        subject_names = psi.item().keys()

    with open('Measurement_model_RMSE.npz', 'rb') as file:
        Measurement_model_RMSE = np.load(file, allow_pickle = True)
        RMSE = Measurement_model_RMSE[mode]

    # plot errorbar for RMSE of each subject
    plt.figure('rmse')
    plt.errorbar(range(10), [RMSE.item()[subject].mean() for subject in subject_names], fmt='ko',\
                 yerr = [np.std(RMSE.item()[subject]) for subject in subject_names])
    plt.plot(range(10),  [RMSE.item()[subject].max() for subject in subject_names], 'ro')
    plt.plot(range(10),  [RMSE.item()[subject].min() for subject in subject_names], 'bo')           
    plt.xticks(range(10), [str(subject) for subject in subject_names])
    plt.title(str(mode)+' : Mean/Max/Min RMSE of each subject')
    plt.legend([ 'Max', 'Min','Mean+/-stdev'])
    plt.grid()

    
    #for subject in subject_names:
    for subject in ['AB01', 'AB02', 'AB03']:
        print("mode: ", str(mode), "; Subject: ", str(subject))
        print("RMSE mean: ", RMSE.item()[subject].mean())
        print("RMSE max: ", RMSE.item()[subject].max())
        
        if mode == 'global_thigh_angle_Y':
            model = m_model.models[0]
            with open('Global_thigh_angle.npz', 'rb') as file:
                g_t = np.load(file)
                measurement_input = g_t[subject][0]
        elif mode =='reaction_force_z_ankle':
            model = m_model.models[1]
            with open('Reaction_wrench.npz', 'rb') as file:
                r_w = np.load(file)
                measurement_input = r_w[subject][2]
        elif mode =='reaction_force_x_ankle':
            model = m_model.models[2]
            with open('Reaction_wrench.npz', 'rb') as file:
                r_w = np.load(file)
                measurement_input = r_w[subject][0]
        elif mode =='reaction_moment_y_ankle':
            model = m_model.models[3]
            with open('Reaction_wrench.npz', 'rb') as file:
                r_w = np.load(file)
                measurement_input = r_w[subject][4]
        else:
            sys.exit('Error: no such mode of input')
            
        phases = get_phase(measurement_input)
        phase_dots = get_phase_dot(subject)
        step_lengths = get_step_length(subject)
        ramps = get_ramp(subject)

        index_max = np.where(RMSE.item()[subject] == RMSE.item()[subject].max())
        print(index_max)
        for index in index_max[0]:
            plt.figure(index)
            measurement_pred = model_prediction(model, psi.item()[subject], phases[index,:], phase_dots[index,:], step_lengths[index,:], ramps[index,:])
            plt.plot(measurement_pred, 'r--')
            plt.plot(measurement_input[index,:], 'r-')
            plt.legend(['prediction', 'actual'])
            plt.title(str(subject) + " / "+ str(mode)+' : Max RMSE = ' + str(RMSE.item()[subject][index, 0]))
        """
        index_med = np.where(RMSE.item()[subject] == np.median(RMSE.item()[subject]))
        for index in index_med[0]:
            plt.figure(index)
            measurement_pred = model_prediction(model, psi.item()[subject], phases[index,:], phase_dots[index,:], step_lengths[index,:], ramps[index,:])
            plt.plot(measurement_pred, 'k--')
            plt.plot(measurement_input[index,:], 'k-')
            plt.legend(['prediction', 'actual'])
            plt.title(str(subject) + " / "+ str(mode)+' : Median RMSE = ' + str(RMSE.item()[subject][index, 0]))
        """
        """
        index_min = np.where(RMSE.item()[subject] == RMSE.item()[subject].min())
        for index in index_min[0]:
            plt.figure(index)
            measurement_pred = model_prediction(model, psi.item()[subject], phases[index,:], phase_dots[index,:], step_lengths[index,:], ramps[index,:])
            plt.plot(measurement_pred, 'b--')
            plt.plot(measurement_input[index,:], 'b-')
            plt.legend(['prediction', 'actual'])
            plt.title(str(subject) + " / "+ str(mode)+' : Min RMSE = ' + str(RMSE.item()[subject][index, 0]))
        """

if __name__ == '__main__':


    #Measurement_model_RMSE(mode = 'global_thigh_angle_Y')
    Measurement_model_RMSE(mode = 'reaction_force_z_ankle')
    #Measurement_model_RMSE(mode = 'reaction_force_x_ankle')
    #Measurement_model_RMSE(mode = 'reaction_moment_y_ankle')

    plt.show()
