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
    #RMSE = dict()

    # Calculate a SUBJECT SPECIFIC model
    #for subject in ['AB01']:
    for subject in subject_names:
        # generate data
        print("Doing subject: " + subject)
        if mode == 'global_thigh_angle_Y':
            with open('Gait_cycle_data/Global_thigh_angle.npz', 'rb') as file:
                g_t = np.load(file)
                measurement_input = g_t[subject][0]
        #elif mode == 'global_thigh_angle_bp':
            #with open('Gait_cycle_data/Global_thigh_angle_bp.npz', 'rb') as file:
                #g_t = np.load(file)
                #measurement_input = g_t[subject]
        elif mode =='reaction_force_z_ankle':
            with open('Gait_cycle_data/Reaction_wrench.npz', 'rb') as file:
                r_w = np.load(file)
                measurement_input = r_w[subject][2]
        elif mode =='reaction_force_x_ankle':
            with open('Gait_cycle_data/Reaction_wrench.npz', 'rb') as file:
                r_w = np.load(file)
                measurement_input = r_w[subject][0]
        elif mode =='reaction_moment_y_ankle':
            with open('Gait_cycle_data/Reaction_wrench.npz', 'rb') as file:
                r_w = np.load(file)
                measurement_input = r_w[subject][4]
        elif mode == 'global_thigh_angVel_5hz':
            with open('Gait_cycle_data/global_thigh_angVel_5hz.npz', 'rb') as file:
                g_tv = np.load(file)
                measurement_input = g_tv[subject]
        elif mode == 'global_thigh_angVel_2x5hz':
            with open('Gait_cycle_data/global_thigh_angVel_2x5hz.npz', 'rb') as file:
                g_tv = np.load(file)
                measurement_input = g_tv[subject]
        elif mode == 'global_thigh_angVel_2hz':
            with open('Gait_cycle_data/global_thigh_angVel_2hz.npz', 'rb') as file:
                g_tv = np.load(file)
                measurement_input = g_tv[subject]
        elif mode == 'atan2':
            with open('Gait_cycle_data/atan2_s.npz', 'rb') as file: # scaled
                atan2 = np.load(file)
                measurement_input = atan2[subject]
        else:
            sys.exit('Error: no such mode of input')
        
        phases = get_phase(measurement_input)
        phase_dots = get_phase_dot(subject)
        step_lengths = get_step_length(subject)
        ramps = get_ramp(subject)
        print("Obtained data for: " + subject)

        # Fit the model for the subject
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
        else:
            psi = least_squares(model, measurement_input.ravel(),\
                               phases.ravel(), phase_dots.ravel(), step_lengths.ravel(), ramps.ravel())
        
        PSI[subject] = psi

        # RMSE of each stride
        """
        rmse = np.zeros((np.shape(measurement_input)[0], 1))
        for i in range(np.shape(measurement_input)[0]):
            measurement_pred = model_prediction(model, psi, phases[i,:], phase_dots[i,:], step_lengths[i,:], ramps[i,:])
            mse = np.square(np.subtract(measurement_input[i,:], measurement_pred)).mean()
            rmse[i] = np.sqrt(mse)
        
        RMSE[subject] = rmse
        #print("mode: ", str(mode), "; Subject: ", str(subject))
        print("RMSE mean: ", rmse.mean())
        print("RMSE max: ", rmse.max())
        """
    return PSI #, RMSE

def load_Psi(subject):
    with open('Psi/Psi_thigh_Y.npz', 'rb') as file:
        p = np.load(file, allow_pickle = True)
        Psi_thigh_Y = p['arr_0'].item()[subject]

    with open('Psi/Psi_force_Z.npz', 'rb') as file:
        p = np.load(file, allow_pickle = True)
        Psi_force_Z = p['arr_0'].item()[subject]

    with open('Psi/Psi_force_X.npz', 'rb') as file:
        p = np.load(file, allow_pickle = True)
        Psi_force_X = p['arr_0'].item()[subject]

    with open('Psi/Psi_moment_Y.npz', 'rb') as file:
        p = np.load(file, allow_pickle = True)
        Psi_moment_Y = p['arr_0'].item()[subject]
   
    with open('Psi/Psi_thighVel_5hz.npz', 'rb') as file:
        p = np.load(file, allow_pickle = True)
        Psi_thighVel_5hz = p['arr_0'].item()[subject]
    
    with open('Psi/Psi_thighVel_2x5hz.npz', 'rb') as file:
        p = np.load(file, allow_pickle = True)
        Psi_thighVel_2x5hz = p['arr_0'].item()[subject]
    
    with open('Psi/Psi_thighVel_2hz_p.npz', 'rb') as file:
        p = np.load(file, allow_pickle = True)
        Psi_thighVel_2hz = p['arr_0'].item()[subject]
    
    with open('Psi/Psi_atan2_s_p.npz', 'rb') as file:
        p = np.load(file, allow_pickle = True)
        Psi_atan2 = p['arr_0'].item()[subject]

    Psi = np.array([Psi_thigh_Y, Psi_force_Z, Psi_force_X, Psi_moment_Y,\
                    Psi_thighVel_5hz, Psi_thighVel_2x5hz, Psi_thighVel_2hz, Psi_atan2], dtype = object)
    
    return Psi

def measurement_error_cov(subject):
    with open('Gait_cycle_data/Global_thigh_angle.npz', 'rb') as file:
        g_t = np.load(file)
        global_thigh_angle_Y = g_t[subject][0]

    with open('Gait_cycle_data/Reaction_wrench.npz', 'rb') as file:
        r_w = np.load(file)
        force_x_ankle = r_w[subject][0]
        force_z_ankle = r_w[subject][2]
        moment_y_ankle = r_w[subject][4]

    with open('Gait_cycle_data/global_thigh_angVel_5hz.npz', 'rb') as file:
        g_tv = np.load(file)
        global_thigh_angVel_5hz = g_tv[subject]

    with open('Gait_cycle_data/global_thigh_angVel_2x5hz.npz', 'rb') as file:
        g_tv = np.load(file)
        global_thigh_angVel_2x5hz = g_tv[subject]

    with open('Gait_cycle_data/global_thigh_angVel_2hz.npz', 'rb') as file:
        g_tv = np.load(file)
        global_thigh_angVel_2hz = g_tv[subject]

    with open('Gait_cycle_data/atan2_s.npz', 'rb') as file:
        at = np.load(file)
        atan2 = at[subject]

    phases = get_phase(global_thigh_angle_Y)
    phase_dots = get_phase_dot(subject)
    step_lengths = get_step_length(subject)
    ramps = get_ramp(subject)

    m_model = model_loader('Measurement_model_8.pickle')
    Psi = load_Psi(subject)

    global_thigh_angle_Y_pred = model_prediction(m_model.models[0], Psi[0], phases.ravel(), phase_dots.ravel(), step_lengths.ravel(), ramps.ravel())
    force_z_ankle_pred = model_prediction(m_model.models[1], Psi[1], phases.ravel(), phase_dots.ravel(), step_lengths.ravel(), ramps.ravel())
    force_x_ankle_pred = model_prediction(m_model.models[2], Psi[2], phases.ravel(), phase_dots.ravel(), step_lengths.ravel(), ramps.ravel())
    moment_y_ankle_pred = model_prediction(m_model.models[3], Psi[3], phases.ravel(), phase_dots.ravel(), step_lengths.ravel(), ramps.ravel())
    global_thigh_angVel_5hz_pred = model_prediction(m_model.models[4], Psi[4], phases.ravel(), phase_dots.ravel(), step_lengths.ravel(), ramps.ravel())
    global_thigh_angVel_2x5hz_pred = model_prediction(m_model.models[5], Psi[5], phases.ravel(), phase_dots.ravel(), step_lengths.ravel(), ramps.ravel())
    global_thigh_angVel_2hz_pred = model_prediction(m_model.models[6], Psi[6], phases.ravel(), phase_dots.ravel(), step_lengths.ravel(), ramps.ravel())
    atan2_pred = model_prediction(m_model.models[7], Psi[7], phases.ravel(), phase_dots.ravel(), step_lengths.ravel(), ramps.ravel()) + 2*np.pi*phases.ravel()
    # wrap to [0, 2pi]
    for i in range(len(atan2_pred)):
        while atan2_pred[i] > 2*np.pi:
            atan2_pred[i] -= 2*np.pi

    # compute covariance from samples Measurement_errors
    print("subject: ",  subject)
    Err_thigh_Y = global_thigh_angle_Y.ravel() - global_thigh_angle_Y_pred
    #with open('Measurement_errors/Err_thigh_Y.npz', 'wb') as file:
        #np.savez(file, Err_thigh_Y, allow_pickle = True) 
    print("mean g_th_Y ", np.mean(Err_thigh_Y))
    print("std g_th_Y ", np.std(Err_thigh_Y))
    print('________________________________')

    Err_fz = force_z_ankle.ravel() - force_z_ankle_pred
    #with open('Measurement_errors/Err_fz.npz', 'wb') as file:
        #np.savez(file, Err_fz, allow_pickle = True) 
    print("mean f_z ", np.mean(Err_fz))
    print("std f_z ", np.std(Err_fz))
    print('________________________________')

    Err_fx = force_x_ankle.ravel() - force_x_ankle_pred
    #with open('Measurement_errors/Err_fx.npz', 'wb') as file:
        #np.savez(file, Err_fx, allow_pickle = True)
    print("mean f_x ", np.mean(Err_fx))
    print("std f_x ", np.std(Err_fx))
    print('________________________________')

    Err_my = moment_y_ankle.ravel() - moment_y_ankle_pred
    #with open('Measurement_errors/Err_my.npz', 'wb') as file:
        #np.savez(file, Err_my, allow_pickle = True)
    print("mean m_y ", np.mean(Err_my))
    print("std m_y ", np.std(Err_my))
    print('________________________________')
    
    Err_thigh_angVel_5hz = global_thigh_angVel_5hz.ravel() - global_thigh_angVel_5hz_pred
    #with open('Measurement_errors/Err_thigh_angVel_5hz.npz', 'wb') as file:
        #np.savez(file, Err_thigh_angVel_5hz, allow_pickle = True)
    print("mean gtv_5hz ", np.mean(Err_thigh_angVel_5hz))
    print("std gtv_5hz ", np.std(Err_thigh_angVel_5hz))
    print('________________________________')

    Err_thigh_angVel_2x5hz = global_thigh_angVel_2x5hz.ravel() - global_thigh_angVel_2x5hz_pred
    #with open('Measurement_errors/Err_thigh_angVel_2x5hz.npz', 'wb') as file:
        #np.savez(file, Err_thigh_angVel_2x5hz, allow_pickle = True)
    print("mean gtv_2x5hz ", np.mean(Err_thigh_angVel_2x5hz))
    print("std gtv_2x5hz ", np.std(Err_thigh_angVel_2x5hz))
    print('________________________________')
    
    Err_thigh_angVel_2hz = global_thigh_angVel_2hz.ravel() - global_thigh_angVel_2hz_pred
    #with open('Measurement_errors/Err_thigh_angVel_2hz.npz', 'wb') as file:
        #np.savez(file, Err_thigh_angVel_2hz, allow_pickle = True)
    print("mean gtv_2hz ", np.mean(Err_thigh_angVel_2hz))
    print("std gtv_2hz ", np.std(Err_thigh_angVel_2hz))
    print('________________________________')
    
    Err_atan2 = atan2.ravel() - atan2_pred
    for i in range(np.shape(Err_atan2)[0]):
        if Err_atan2[i] > np.pi:
            Err_atan2[i] -= 2*np.pi
        elif Err_atan2[i] < -np.pi:
            Err_atan2[i] += 2*np.pi
    #with open('Measurement_errors/Err_atan2.npz', 'wb') as file:
        #np.savez(file, Err_atan2, allow_pickle = True)
    print("mean atan2 ", np.mean(Err_atan2))
    print("std atan2 ", np.std(Err_atan2))

    Err = np.stack((Err_thigh_Y, Err_fz, Err_fx, Err_my,\
                    Err_thigh_angVel_5hz, Err_thigh_angVel_2x5hz, Err_thigh_angVel_2hz, Err_atan2))
    R = np.cov(Err)
    print("R = ", R)
    return R

if __name__ == '__main__':
    
    # dictionary storing all measurement model coefficients
    #Measurement_model_coeff = dict()
    #Measurement_model_RMSE = dict()

    # Orders of the measurement model
    F = 11
    N = 3

    phase_model = Fourier_Basis(F, 'phase')
    phase_dot_model = Polynomial_Basis(1, 'phase_dot')
    step_length_model = Berstein_Basis(N,'step_length')
    ramp_model = Berstein_Basis(N, 'ramp')
    
    # Measrurement model for global_thigh_angle_Y
    model_thigh_Y = Kronecker_Model(phase_model, phase_dot_model, step_length_model, ramp_model)
    #psi_thigh_Y = model_fit(model_thigh_Y, 'global_thigh_angle_Y')
    #with open('Psi/Psi_thigh_Y_2.npz', 'wb') as file:
    #    np.savez(file, psi_thigh_Y, allow_pickle = True) 
    
    # Measrurement model for reaction_force_z_ankle
    model_force_z = Kronecker_Model(phase_model, phase_dot_model, step_length_model, ramp_model)
    #psi_force_Z = model_fit(model_force_z, 'reaction_force_z_ankle')
    #with open('Psi/Psi_force_Z_2.npz', 'wb') as file:
    #    np.savez(file, psi_force_Z, allow_pickle = True)
    #Measurement_model_coeff['reaction_force_z_ankle'] = psi_force_z
    #Measurement_model_RMSE['reaction_force_z_ankle'] = RMSE_force_z

    # Measrurement model for reaction_force_x_ankle
    model_force_x = Kronecker_Model(phase_model, phase_dot_model, step_length_model, ramp_model)
    #psi_force_X = model_fit(model_force_x, 'reaction_force_x_ankle')
    #with open('Psi/Psi_force_X_2.npz', 'wb') as file:
    #    np.savez(file, psi_force_X, allow_pickle = True)
    #Measurement_model_coeff['reaction_force_x_ankle'] = psi_force_x
    #Measurement_model_RMSE['reaction_force_x_ankle'] = RMSE_force_x

    # Measrurement model for reaction_moment_y_ankle
    model_moment_y = Kronecker_Model(phase_model, phase_dot_model, step_length_model, ramp_model)
    #psi_moment_Y = model_fit(model_moment_y, 'reaction_moment_y_ankle')
    #with open('Psi/Psi_moment_Y_2.npz', 'wb') as file:
    #    np.savez(file, psi_moment_Y, allow_pickle = True) 
    #Measurement_model_coeff['reaction_moment_y_ankle'] = psi_moment_y
    #Measurement_model_RMSE['reaction_moment_y_ankle'] = RMSE_moment_y
    
    #phase_dot_model = Polynomial_Basis(2, 'phase_dot')

    # NEW ADDED: Measrurement model for global_thigh_angVel_Y
    model_thighVel_5hz = Kronecker_Model(phase_model, phase_dot_model, step_length_model, ramp_model)
    #psi_thighVel_5hz = model_fit(model_thighVel_5hz, 'global_thigh_angVel_5hz')
    #with open('Psi/Psi_thighVel_5hz_2.npz', 'wb') as file:
    #    np.savez(file, psi_thighVel_5hz, allow_pickle = True) 

    model_thighVel_2x5hz = Kronecker_Model(phase_model, phase_dot_model, step_length_model, ramp_model)
    #psi_thighVel_2x5hz = model_fit(model_thighVel_2x5hz, 'global_thigh_angVel_2x5hz')
    #with open('Psi/Psi_thighVel_2x5hz_2.npz', 'wb') as file:
    #    np.savez(file, psi_thighVel_2x5hz, allow_pickle = True)

    model_thighVel_2hz = Kronecker_Model(phase_model, phase_dot_model, step_length_model, ramp_model)
    #psi_thighVel_2hz = model_fit(model_thighVel_2hz, 'global_thigh_angVel_2hz')
    #with open('Psi/Psi_thighVel_2hz_p.npz', 'wb') as file:
    #   np.savez(file, psi_thighVel_2hz, allow_pickle = True)

    #phase_dot_model = Polynomial_Basis(1, 'phase_dot')
    step_length_model = Berstein_Basis(0,'step_length')
    ramp_model = Berstein_Basis(0, 'ramp')
    model_atan2 = Kronecker_Model(phase_model, phase_dot_model, step_length_model, ramp_model)
    #psi_atan2 = model_fit(model_atan2, 'atan2')
    #with open('Psi/Psi_atan2_s_pv.npz', 'wb') as file:
    #    np.savez(file, psi_atan2, allow_pickle = True)

    """
    with open('Measurement_model_coeff.npz', 'rb') as file:
        Measurement_model_coeff = np.load(file, allow_pickle = True)
        psi_thigh_Y = Measurement_model_coeff['global_thigh_angle_Y']
        psi_force_Z = Measurement_model_coeff['reaction_force_z_ankle']
        psi_force_X = Measurement_model_coeff['reaction_force_x_ankle']
        psi_moment_Y = Measurement_model_coeff['reaction_moment_y_ankle']
    
    with open('Psi/Psi_thigh_Y.npz', 'wb') as file:
        np.savez(file, psi_thigh_Y, allow_pickle = True) 
    with open('Psi/Psi_force_Z.npz', 'wb') as file:
        np.savez(file, psi_force_Z, allow_pickle = True) 
    with open('Psi/Psi_force_X.npz', 'wb') as file:
        np.savez(file, psi_force_X, allow_pickle = True) 
    with open('Psi/Psi_moment_Y.npz', 'wb') as file:
        np.savez(file, psi_moment_Y, allow_pickle = True) 
    """
    
    #####################################################################################################

    # save measurement model coeffiecients (Psi)
    #with open('Measurement_model_coeff.npz', 'wb') as file:
    #    np.savez(file, **Measurement_model_coeff, allow_pickle = True)

    # save RMSE
    #with open('Measurement_model_RMSE.npz', 'wb') as file:
    #    np.savez(file, **Measurement_model_RMSE, allow_pickle = True)

    # save measurement model
    m_model = Measurement_Model(model_thigh_Y, model_force_z, model_force_x, model_moment_y,\
                                model_thighVel_2hz)
    #m_model = Measurement_Model(model_thigh_Y, model_force_z, model_force_x, model_moment_y, model_thighVel_2hz)
    model_saver(m_model, 'Measurement_model_5_sp.pickle')

    #R = dict()
    #for subject in subject_names:
    #    R[subject] = measurement_error_cov(subject)
    #with open('R_s.pickle', 'wb') as file:
    #	pickle.dump(R, file)
