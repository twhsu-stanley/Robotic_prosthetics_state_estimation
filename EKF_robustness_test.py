from pickle import FALSE
import numpy as np
from numpy.core.numeric import ones
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from EKF import *
from model_framework import *
from continuous_data import *
from basis_model_fitting import measurement_noise_covariance, heteroscedastic_measurement_noise_covariance, saturation_bounds
import csv

# Dictionary of all sensors
# All measurements use the basis model, except for directRamp
sensors_dict = {'globalThighAngles':0, 'globalThighVelocities':1, 'atan2':2,
                'globalFootAngles':3, 'ankleMoment':4, 'tibiaForce':5}

# Determine what sensors to be used
# 1) measurements that use the basis model
sensors = ['globalThighAngles', 'globalThighVelocities', 'atan2']#, 'ankleMoment', 'tibiaForce'

sensor_id = [sensors_dict[key] for key in sensors]
sensor_id_str = ""
for i in range(len(sensor_id)):
    sensor_id_str += str(sensor_id[i])
m_model = model_loader('Measurement_model_' + sensor_id_str +'_NSL.pickle')

using_atan2 = np.any(np.array(sensors) == 'atan2')
using_ankleMoment = np.any(np.array(sensors) == 'ankleMoment')
using_tibiaForce = np.any(np.array(sensors) == 'tibiaForce')
using_footAngles = np.any(np.array(sensors) == 'globalFootAngles')

tibiaForce_threshold = -1.2

# 2) direct measurement
using_directRamp = False
R_directRamp = 8
L_cop_lower = 0.03
L_cop_upper = 0.07

print("Using sensors:", sensors, "| using direct ramp:", using_directRamp)

Psi = np.array([load_Psi('Generic')[key] for key in sensors], dtype = object)

dt = 1/100
inital_Sigma = np.diag([1e-6, 1e-6, 1e-6, 1e-6])
Q = np.diag([0, 1e-3, 4e-2, 0]) * dt 
U = np.diag([1, 1, 1])
R = U @ measurement_noise_covariance(*sensors) @ U.T
if using_directRamp == True:
    R = np.diag(np.append(np.diag(R), R_directRamp))

hetero_cov = heteroscedastic_measurement_noise_covariance(*sensors)

saturation_range = saturation_bounds()

if using_ankleMoment or using_tibiaForce or using_footAngles or using_directRamp:
    sensors_swing = []
    for i in range(len(sensors)):
        if sensors[i] == 'ankleMoment' or sensors[i] == 'tibiaForce' or sensors[i] == 'globalFootAngles':
            continue
        else:
            sensors_swing.append(sensors[i])
    sensor_swing_id = [sensors_dict[key] for key in sensors_swing]
    sensor_swing_id_str = ""
    for i in range(len(sensor_swing_id)):
        sensor_swing_id_str += str(sensor_swing_id[i])
    Psi_swing = np.array([load_Psi('Generic')[key] for key in sensors_swing], dtype = object)
    m_model_swing = model_loader('Measurement_model_' + sensor_swing_id_str +'_NSL.pickle')
    U_swing = np.diag(np.diag(U)[sensor_swing_id])
    R_swing = U_swing @ measurement_noise_covariance(*sensors_swing) @ U_swing.T
    if using_directRamp == True:
        R_swing = np.diag(np.append(np.diag(R_swing), R_directRamp))

# Skip trials with problematic measurements
with open('Continuous_data/Measurements_with_Nan.pickle', 'rb') as file:
    nan_dict = pickle.load(file)

# Stride in which kidnapping occurs
kidnap_stride = 3
total_strides = 12

# Roecover Criteria
phase_recover_thr = 0.05
step_length_recover_thr = 0.1
ramp_recover_thr = 1

## From loco_OSL.py: Load referenced trajectories
def loadTrajectory(trajectory = 'walking'):
    # Create path to the reference csv trajectory
    if trajectory.lower() == 'walking':
        # walking data uses convention from D. A. Winter, “Biomechanical Motor Patterns in Normal Walking,”  
        # J. Mot. Behav., vol. 15, no. 4, pp. 302–330, Dec. 1983.
        pathFile = r'OSL_walking_data/walkingWinter_deg.csv'
        # Gains to scale angles to OSL convention
        ankGain = -1
        ankOffs = -0.15 # [deg] Small offset to take into accoun that ankle ROM is -10 deg< ankle < 19.65 deg
        kneGain = -1
        kneOffs = 0
        hipGain = 1
        hipOffs = 0
    else:
        raise ValueError('Please select a suported trajectory type')
    # Extract content from csv
    with open(pathFile, 'r') as f:
        datasetReader = csv.reader(f, quoting = csv.QUOTE_NONNUMERIC)
        data = np.transpose( np.array([row for row in datasetReader ]) )
    # Parse data to knee-ankle trajectories using OSL angle convention (+ ankle = plantarflexion. + knee = flexion)
    trajectory = dict(ankl = ankGain*data[0] + ankOffs)
    trajectory["ankd"] = ankGain*data[1]
    trajectory["andd"] = ankGain*data[2]
    trajectory["knee"] = kneGain*data[3] + kneOffs
    trajectory["kned"] = kneGain*data[4]
    trajectory["kndd"] = kneGain*data[5]
    trajectory["hip_"] = hipGain*data[6] + hipOffs
    trajectory["hipd"] = hipGain*data[7]
    trajectory["hidd"] = hipGain*data[8]
    trajectory["phas"] = data[9]
    trajectory["time"] = data[10]

    return trajectory

def ekf_test(subject, trial, side, heteroscedastic = False, kidnap = False, plot = False):
    print("EKF Test: ", subject, "/", trial, '/', side , "| Heteroscedastic R:", heteroscedastic)
    # load ground truth
    phases, phase_dots, step_lengths, ramps = Conti_state_vars(subject, trial, side)
    # load measurements
    globalThighAngle, globalThighVelocity, atan2, globalFootAngle, ankleMoment, tibiaForce = load_Conti_measurement_data(subject, trial, side)

    #### Joint Control ############################################################
    knee_angle, ankle_angle = load_Conti_joints_angles(subject, trial, side)
    ### Load reference trajectory
    refTrajectory = loadTrajectory(trajectory = 'walking')
    refAnk = refTrajectory["ankl"]
    refKne = refTrajectory["knee"]
    ################################################################################

    z_full = np.array([[globalThighAngle], [globalThighVelocity], [atan2], [globalFootAngle], [ankleMoment], [tibiaForce]])
    z_full = np.squeeze(z_full) 
    z = z_full[sensor_id, :]

    # build the system
    sys = myStruct()
    sys.f = process_model
    sys.A = A
    sys.h = m_model
    sys.Q = Q
    sys.R = R

    # initialize the state
    init = myStruct()
    init.x = np.array([[phases[0]], [phase_dots[0]], [step_lengths[0]], [ramps[0]]])
    #init.x = np.array([[phases[0]], [phase_dots[0]], [step_lengths[0]], [0]])
    init.Sigma = inital_Sigma

    ekf = extended_kalman_filter(sys, init)
    
    heel_strike_index = Conti_heel_strikes(subject, trial, side) - Conti_heel_strikes(subject, trial, side)[0]
    
    if kidnap != False:
        kidnap_index = np.random.randint(heel_strike_index[kidnap_stride, 0], heel_strike_index[kidnap_stride+1, 0]) # step at which kidnapping occurs
        #print("kidnap_index(%) = ", (kidnap_index - heel_strike_index[3, 0])/(heel_strike_index[4, 0]- heel_strike_index[3, 0])*100)
        phase_kidnap =  np.random.uniform(0, 1)
        phase_dot_kidnap = np.random.uniform(0, 5)
        step_length_kidnap = np.random.uniform(0, 2)
        ramp_kidnap = np.random.uniform(-45, 45)
        state_kidnap = np.array([[phase_kidnap], [phase_dot_kidnap], [step_length_kidnap], [ramp_kidnap]])
        print("state_kidnap = [%4.2f, %4.2f, %4.2f, %5.2f]" % (state_kidnap[0], state_kidnap[1], state_kidnap[2], state_kidnap[3]))

    total_step = int(heel_strike_index[total_strides, 0]) + 1
    phases = phases[0 : total_step]
    phase_dots = phase_dots[0 : total_step]
    step_lengths = step_lengths[0 : total_step]
    ramps = ramps[0 : total_step]

    x = np.zeros((total_step, 4))  # state estimate
    z_pred = np.zeros((total_step, len(sensors) + int(using_directRamp)))
    directRamp = np.zeros((total_step, 1))
    L_cop = np.zeros((total_step, 1))
    #Q_diag = np.zeros((total_step, 4))
    #Sigma_diag = np.zeros((total_step, 4))
    
    estimate_error = np.zeros((total_step, 4))
    #MD_residual = np.zeros((total_step, 1))
    #MD_estimate = np.zeros((total_step, 1))
    
    knee_angle_kmd = np.zeros((total_step, 1))
    ankle_angle_kmd = np.zeros((total_step, 1))
    knee_angle_cmd = np.zeros((total_step, 1))
    ankle_angle_cmd = np.zeros((total_step, 1))
    
    stance = False
    stance_prev = False
    stance_idxs = 0
    stance_idx1 = 0
    stance_idx2 = 0
    
    for i in range(total_step):
        if kidnap != False:
            if i == kidnap_index:
                ekf.x[kidnap] = state_kidnap[kidnap]
        
        ekf.prediction(dt)
        ekf.state_saturation(saturation_range)
        
        if using_ankleMoment or using_tibiaForce or using_footAngles or using_directRamp:
            # Compute L_cop; here, "stance" means that the foot is flat on the ground 
            if tibiaForce[i] <= tibiaForce_threshold:
                # Location of the centoer of pressure
                L_cop[i] = -ankleMoment[i] / tibiaForce[i]
                if stance_prev == False and L_cop[i] > L_cop_lower and L_cop[i] <= L_cop_upper and (i - stance_idxs) > 0.5/dt:
                    stance_idxs = i
                    stance = True
                    stance_prev = stance
                    
                elif stance_prev == True and L_cop[i] > L_cop_upper:
                    stance_idx2 = i
                    stance_idx1 = stance_idxs
                    stance = False
                    stance_prev = stance
            else:
                L_cop[i] = -1 # arbitraty number
                if stance_prev == True:
                    stance_idx2 = i
                    stance_idx1 = stance_idxs
                stance = False
                stance_prev = stance
            
            if stance_idx1 < stance_idx2:
                directRamp[i] = np.mean(globalFootAngle[stance_idx1:stance_idx2])
            else:
                directRamp[i] = 1e-4
            
            #if stance == True: # stance
            if tibiaForce[i] <= tibiaForce_threshold:
                ekf.h = m_model
                if heteroscedastic == True:
                    ekf.R = np.diag(hetero_cov[:, int(ekf.x[0, 0]*150)])
                    if using_directRamp:
                        ekf.R = np.diag(np.append(np.diag(ekf.R), R_directRamp))               
                else:
                    ekf.R = R

                if using_directRamp:
                    ekf.correction(z[:, i], Psi, using_atan2, steady_state_walking = True, direct_ramp = directRamp[i])
                else:
                    ekf.correction(z[:, i], Psi, using_atan2, steady_state_walking = True, direct_ramp = False)
                z_pred[i,:] = ekf.z_hat.T
            else: # swing
                ekf.h = m_model_swing
                if heteroscedastic == True:
                    ekf.R = np.diag(hetero_cov[sensor_swing_id, int(ekf.x[0, 0]*150)])
                    if using_directRamp:
                        ekf.R = np.diag(np.append(np.diag(ekf.R), R_directRamp))
                else:
                    ekf.R = R_swing
                
                if using_directRamp :
                    ekf.correction(z_full[sensor_swing_id, i], Psi_swing, using_atan2, steady_state_walking = True, direct_ramp = directRamp[i])
                    z_pred[i, np.append(sensor_swing_id, len(sensors))] = ekf.z_hat.T
                else:
                    ekf.correction(z_full[sensor_swing_id, i], Psi_swing, using_atan2, steady_state_walking = True, direct_ramp = False)
                    z_pred[i, sensor_swing_id] = ekf.z_hat.T
                
                #==============================================================================================================
        else:
            if heteroscedastic == True:
                ekf.R = np.diag(hetero_cov[:, int(ekf.x[0, 0]*150)])
            else:
                ekf.R = R
            ekf.correction(z[:, i], Psi, using_atan2, steady_state_walking = True)
            z_pred[i,:] = ekf.z_hat.T
        ekf.state_saturation(saturation_range)

        x[i,:] = ekf.x.T

        #Q_diag[i,:] = np.diag(ekf.Q)
        #Sigma_diag[i, :] = np.diag(ekf.Sigma)
        #MD_residual[i] = ekf.MD_residual

        estimate_error[i, :] = (ekf.x - np.array([[phases[i]], [phase_dots[i]], [step_lengths[i]], [ramps[i]]])).reshape(-1)
        if estimate_error[i, 0] > 0.5:
            estimate_error[i, 0] = estimate_error[i, 0] - 1
        elif estimate_error[i, 0] < -0.5:
            estimate_error[i, 0] = 1 + estimate_error[i, 0]

        #MD_estimate[i] = np.sqrt(estimate_error[i, :].T @ np.linalg.inv(ekf.Sigma) @ estimate_error[i, :])

        ## Joints control commands 
        # 1) generated by the kinematic model
        joint_angles = joints_control(x[i,0], x[i,1], x[i,2], x[i,3])
        knee_angle_kmd[i] = joint_angles[0]
        ankle_angle_kmd[i] = joint_angles[1]
        # 2) generated by Edgar's prescribed trajectories
        pv = int(ekf.x[0, 0] * 998)  # phase variable conversion (scaling)
        ankle_angle_cmd[i] = refAnk[pv]
        knee_angle_cmd[i] = refKne[pv]

    if kidnap == False:
        start_check_idx = int(3/np.average(phase_dots)/dt)
        SE_phase = np.sum(estimate_error[start_check_idx:, 0] ** 2)
        SE_phase_dot = np.sum(estimate_error[start_check_idx:, 1] ** 2)
        SE_step_length = np.sum(estimate_error[start_check_idx:, 2] ** 2)
        SE_ramp = np.sum(estimate_error[start_check_idx:, 3] ** 2)
        SE_directRamp = np.sum((directRamp[start_check_idx:, 0] - ramps[start_check_idx:]) ** 2)
        T = len(estimate_error[start_check_idx:, 0])

        RMSE_phase = np.sqrt(SE_phase/T)
        RMSE_phase_dot = np.sqrt(SE_phase_dot/T)
        RMSE_step_length = np.sqrt(SE_step_length/T)
        RMSE_ramp = np.sqrt(SE_ramp/T)
        RMSE_directRamp = np.sqrt(SE_directRamp/T) 
        
        print("RMSE phase = %5.3f" % RMSE_phase)
        print("RMSE phase_dot = %5.3f" % RMSE_phase_dot)
        print("RMSE step_length = %5.3f" % RMSE_step_length)
        #print("RMSE ramp = %5.3f" % RMSE_ramp)
        #print("RMSE direct ramp = %5.3f" % RMSE_directRamp)
        
        result = (SE_phase, SE_phase_dot, SE_step_length, SE_ramp, SE_directRamp, T)

    if plot == True:
        #th = heel_strike_index[0:25, 0].astype(int) # time step of heel strikes
        nu = np.sqrt(18.5)
        # plot results
        tt = dt * np.arange(len(phases))
        plt.figure("State Estimate")
        plt.subplot(411)
        plt.title('EKF Robustness Test')
        plt.plot(tt, phases, 'k-')
        plt.plot(tt, x[:, 0], 'r--')
        #plt.plot(tt, x[:, 0] + Sigma_diag[:, 0]*nu, 'b-')
        #plt.plot(tt, x[:, 0] - Sigma_diag[:, 0]*nu, 'g-')
        #plt.plot(th* dt, np.zeros((len(th), 1)), "rx")
        plt.legend(('ground truth', 'estimate'))
        #plt.legend(('ground truth', 'estimate'), bbox_to_anchor=(1, 1.05))
        plt.ylabel('$\phi$')
        plt.ylim([0, 2])
        plt.xlim([0, tt[-1]+0.1])
        plt.grid()
        plt.subplot(412)
        plt.plot(tt, phase_dots, 'k-')
        plt.plot(tt, x[:, 1], 'r--')
        #plt.plot(tt, x[:, 1] + Sigma_diag[:, 1]*nu, 'b-')
        #plt.plot(tt, x[:, 1] - Sigma_diag[:, 1]*nu, 'g-')
        plt.ylabel('$\dot{\phi}~(s^{-1})$')
        plt.xlim([0, tt[-1]+0.1])
        plt.ylim([0.5, 1.5])
        plt.grid()
        plt.subplot(413)
        plt.plot(tt, step_lengths, 'k-')
        plt.plot(tt, x[:, 2], 'r--')
        #plt.plot(tt, x[:, 2] + Sigma_diag[:, 2]*nu, 'b-')
        #plt.plot(tt, x[:, 2] - Sigma_diag[:, 2]*nu, 'g-')
        plt.ylabel('$l~(m)$')
        plt.xlim([0, tt[-1]+0.1])
        plt.ylim([0, 2])
        plt.grid()
        plt.subplot(414)
        plt.plot(tt, ramps, 'k-')
        plt.plot(tt, x[:, 3], 'r--')
        #plt.plot(tt, x[:, 3] + Sigma_diag[:, 3]*nu, 'b-')
        #plt.plot(tt, x[:, 3] - Sigma_diag[:, 3]*nu, 'g-')
        plt.ylabel('$\\alpha~(deg)$')
        plt.xlabel('time (s)')
        plt.xlim([0, tt[-1]+0.1])
        plt.ylim([-15, 15])
        plt.grid()

        plt.figure("Estimation Errors")
        plt.subplot(411)
        plt.title('Estimation Errors')
        plt.plot(tt, abs(estimate_error[:, 0].T))
        plt.ylabel('$\phi$ error')
        plt.ylim([0, 0.5])
        plt.xlim([0, tt[-1]+0.1])
        plt.grid()
        plt.subplot(412)
        plt.plot(tt, abs(estimate_error[:, 1].T))
        plt.ylabel('$\dot{\phi}$ error (1/s)')
        plt.xlim([0, tt[-1]+0.1])
        plt.grid()
        plt.subplot(413)
        plt.plot(tt, abs(estimate_error[:, 2].T))
        plt.ylabel('$l$ error (m)')
        plt.xlim([0, tt[-1]+0.1])
        plt.ylim([0, 1])
        plt.grid()
        plt.subplot(414)
        plt.plot(tt, abs(estimate_error[:, 3].T))
        plt.ylabel('$\\alpha$ error (deg)')
        plt.xlim([0, tt[-1]+0.1])
        #plt.ylim([5, 0.5])
        plt.xlabel('time (s)')
        plt.grid()

        """
        plt.figure("Mahalanobis Distance of Residuals")
        plt.title("Mahalanobis Distance of Residuals")
        plt.plot(tt, MD_residual)
        plt.ylabel('MD of Residuals')
        plt.xlabel('time (s)')
        
        plt.figure("Process Noise Covariance")
        plt.subplot(411)
        plt.title("diag Q")
        plt.plot(tt, Q_diag[:, 0])
        plt.ylabel('$Q(1,1)$')
        plt.subplot(412)
        plt.plot(tt, Q_diag[:, 1])
        plt.ylabel('$Q(2,2)$')
        plt.subplot(413)
        plt.plot(tt, Q_diag[:, 2])
        plt.ylabel('$Q(3,3)$')
        plt.subplot(414)
        plt.plot(tt, Q_diag[:, 3])
        plt.ylabel('$Q(4,4)$')
        plt.xlabel('time (s)')
        
        plt.figure("Mahalanobis Distance of State Estimate")
        plt.title("Mahalanobis Distance of State Estimate")
        plt.plot(tt, MD_estimate)
        plt.plot(tt, nu * np.ones((len(tt), 1)))
        plt.ylabel('MD of State Estimate')
        plt.xlabel('time (s)')
        """

        plt.figure("Control Commands: Joint Angles")
        plt.title("Control Commands: Joint Angles")
        plt.subplot(211)
        plt.plot(tt, knee_angle[0:total_step], 'k-')
        #plt.plot(tt, knee_angle_cmd, 'r-')
        plt.plot(tt, knee_angle_kmd, 'm-')
        plt.legend(('actual', 'kinematic model', 'Edgar\'s trajectory'))
        plt.ylabel('knee angle (deg)')
        plt.subplot(212)
        plt.plot(tt, ankle_angle[0:total_step], 'k-')
        #plt.plot(tt, ankle_angle_cmd, 'r-')
        plt.plot(tt, ankle_angle_kmd, 'm-')
        plt.legend(('actual', 'kinematic model', 'Edgar\'s trajectory'))
        plt.ylabel('ankle angle (deg)')
        plt.xlabel('time (s)')

        plt.figure("Measurements")
        for i in range(len(sensors) + int(using_directRamp)):
            plt.subplot(int(str(len(sensors) + int(using_directRamp)) + "1" + str(i+1)))
            if using_directRamp == True and i == len(sensors):
                plt.plot(tt, directRamp[0:total_step], 'k-')
            else:
                plt.plot(tt, z[i, 0:total_step], 'k-')
            plt.plot(tt, z_pred[:, i], 'r--')
            plt.xlim([0, tt[-1]+0.1])
            plt.grid()

            if i == 0:
                plt.title("Measurements")
                plt.legend(('actual', 'predicted'))
            elif i == len(sensors)-1:
                plt.xlabel("time (s)")

        plt.figure("Kinetics")
        plt.subplot(411)
        plt.title("Kinetics")
        plt.plot(tt, tibiaForce[0:total_step], 'k-')
        plt.plot(tt, tibiaForce_threshold*np.ones((len(tt),1)), 'b--')
        plt.xlim([0, tt[-1]+0.1])
        plt.ylabel('Tibia Axial Force')
        plt.grid()
        plt.subplot(412)
        plt.plot(tt, ankleMoment[0:total_step], 'k-')
        plt.xlim([0, tt[-1]+0.1])
        plt.ylabel('Ankle Moment')
        plt.grid()
        plt.subplot(413)
        plt.plot(tt, L_cop[0:total_step], 'k-')
        plt.xlim([0, tt[-1]+0.1])
        plt.ylabel('L cop')
        plt.grid()
        plt.subplot(414)
        plt.plot(tt, ramps, 'k-')
        plt.plot(tt, x[:, 3], 'r--')
        plt.plot(tt, globalFootAngle[0:total_step], 'm-')
        plt.plot(tt, directRamp, 'g-')
        plt.legend(('Ground truth ramp', 'EKF ramp est','foot angles','backup ramp'))
        plt.ylabel('Ramp Est & Foot Angle')
        plt.xlim([0, tt[-1]+0.1])
        plt.grid()
        
        plt.show()
    
    if kidnap == False:
        return result
    else:
        exit("The ekf_test program does not return anything for the kidnapping case.")

def ekf_bank_test(subject, trial, side, N = 30, heteroscedastic = False, kidnap = [0, 1, 2, 3], plot = True):
    # N: number of EKFs in the EKF-bank
    print("Monte-Carlo Test: ", subject, "/", trial, '/', side)

    # load ground truth
    phases, phase_dots, step_lengths, ramps = Conti_state_vars(subject, trial, side)
    # load measurements
    globalThighAngle, globalThighVelocity, atan2, globalFootAngle, ankleMoment, tibiaForce = load_Conti_measurement_data(subject, trial, side)

    z_full = np.array([[globalThighAngle], [globalThighVelocity], [atan2], [globalFootAngle], [ankleMoment], [tibiaForce]])
    z_full = np.squeeze(z_full)
    z = z_full[sensor_id, :]

    # build the system
    sys = myStruct()
    sys.f = process_model
    sys.A = A
    sys.h = m_model
    sys.Q = Q
    sys.R = R
    
    init = myStruct()

    heel_strike_index = Conti_heel_strikes(subject, trial, side) - Conti_heel_strikes(subject, trial, side)[0]
    total_step = int(heel_strike_index[total_strides, 0]) + 1
    
    # ground truth states
    phases = phases[0:total_step]
    phase_dots = phase_dots[0:total_step]
    step_lengths = step_lengths[0:total_step]
    ramps = ramps[0:total_step]
    
    kidnap_index = np.random.randint(heel_strike_index[kidnap_stride, 0], heel_strike_index[kidnap_stride+1, 0]) # step at which kidnapping occurs
    x = np.zeros((N+1, total_step, 4))  # state estimate
    estimate_error = np.zeros((N+1, total_step, 4))
    phase_converge_dist = np.zeros((N+1, total_step))
    r11 = 0
    r13 = 0
    r33 = 0
    r15 = 0
    r55 = 0

    for n in range(N+1):
        # n = 0: normal test w/0 kidnapping
        # initialize the state
        init.x = np.array([[phases[0]], [phase_dots[0]], [step_lengths[0]], [ramps[0]]])
        #init.x = np.array([[phases[0]], [phase_dots[0]], [step_lengths[0]], [0]])
        init.Sigma = inital_Sigma
        # build EKF
        ekf = extended_kalman_filter(sys, init)
        
        phase_kidnap = np.random.uniform(0, 1)
        phase_dot_kidnap = np.random.uniform(0, 5)
        step_length_kidnap = np.random.uniform(0, 2)
        ramp_kidnap = np.random.uniform(-45, 45)
        state_kidnap = np.array([[phase_kidnap], [phase_dot_kidnap], [step_length_kidnap], [ramp_kidnap]])

        directRamp = np.zeros((total_step, 1))
        L_cop = np.zeros((total_step, 1))
        stance = False
        stance_prev = False
        stance_idxs = 0
        stance_idx1 = 0
        stance_idx2 = 0
        for i in range(total_step):
            # kidnap
            if i == kidnap_index and n > 0:
                ekf.x[kidnap] = state_kidnap[kidnap]
            
            ekf.prediction(dt)
            ekf.state_saturation(saturation_range)

            if using_ankleMoment or using_tibiaForce or using_footAngles or using_directRamp:
                if tibiaForce[i] <= tibiaForce_threshold:
                    # Location of the centoer of pressure
                    L_cop[i] = -ankleMoment[i] / tibiaForce[i]
                    if stance_prev == False and L_cop[i] > L_cop_lower and L_cop[i] <= L_cop_upper and (i - stance_idxs) > 0.5/dt:
                        stance_idxs = i
                        stance = True
                        stance_prev = stance
                        
                    elif stance_prev == True and L_cop[i] > L_cop_upper:
                        stance_idx2 = i
                        stance_idx1 = stance_idxs
                        stance = False
                        stance_prev = stance
                else:
                    L_cop[i] = -1 # arbitraty number
                    if stance_prev == True:
                        stance_idx2 = i
                        stance_idx1 = stance_idxs
                    stance = False
                    stance_prev = stance
                
                if stance_idx1 < stance_idx2:
                    directRamp[i] = np.mean(globalFootAngle[stance_idx1:stance_idx2])
                else:
                    directRamp[i] = 1e-4
                
                #if stance == True:
                if tibiaForce[i] <= tibiaForce_threshold:
                    ekf.h = m_model
                    if heteroscedastic == True:
                        ekf.R = np.diag(hetero_cov[:, int(ekf.x[0, 0]*150)])
                        if using_directRamp:
                            ekf.R = np.diag(np.append(np.diag(ekf.R), R_directRamp))               
                    else:
                        ekf.R = R

                    if using_directRamp:
                        ekf.correction(z[:, i], Psi, using_atan2, steady_state_walking = True, direct_ramp = directRamp[i])
                    else:
                        ekf.correction(z[:, i], Psi, using_atan2, steady_state_walking = True, direct_ramp = False)
                else: # swing
                    ekf.h = m_model_swing
                    if heteroscedastic == True:
                        ekf.R = np.diag(hetero_cov[sensor_swing_id, int(ekf.x[0, 0]*150)])
                        if using_directRamp:
                            ekf.R = np.diag(np.append(np.diag(ekf.R), R_directRamp))
                    else:
                        ekf.R = R_swing
                    
                    if using_directRamp:
                        ekf.correction(z_full[sensor_swing_id, i], Psi_swing, using_atan2, steady_state_walking = True, direct_ramp = directRamp[i])
                    else:
                        ekf.correction(z_full[sensor_swing_id, i], Psi_swing, using_atan2, steady_state_walking = True, direct_ramp = False)       

            else:
                if heteroscedastic == True:
                    ekf.R = np.diag(hetero_cov[:, int(ekf.x[0, 0]*150)])
                else:
                    ekf.R = R
                ekf.correction(z[:, i], Psi, using_atan2, steady_state_walking = True)
            ekf.state_saturation(saturation_range)

            x[n, i,:] = ekf.x.T

            estimate_error[n, i, :] = (ekf.x - np.array([[phases[i]], [phase_dots[i]], [step_lengths[i]], [ramps[i]]])).reshape(-1)
            if estimate_error[n, i, 0] > 0.5:
                estimate_error[n, i, 0] = estimate_error[n, i, 0] - 1
            elif estimate_error[n, i, 0] < -0.5:
                estimate_error[n, i, 0] = 1 + estimate_error[n, i, 0]
        if n > 0:
            #idx_1 = int(kidnap_index + 1/np.average(phase_dots)/dt)
            idx_1 = int(heel_strike_index[kidnap_stride + 2, 0])
            #idx_3 = int(kidnap_index + 3/np.average(phase_dots)/dt)
            idx_3 = int(heel_strike_index[kidnap_stride + 4, 0])
            #idx_5 = int(kidnap_index + 5/np.average(phase_dots)/dt)
            idx_5 = int(heel_strike_index[kidnap_stride + 6, 0])

            # 1) Error mean approach
            """
            # Means of steady-state stride_length & ramp estimates before kidnapping
            step_length_est_error_mean = np.mean(estimate_error[n, int(heel_strike_index[2, 0]):int(kidnap_index-1), 2])
            ramp_est_error_mean = np.mean(estimate_error[n, int(heel_strike_index[2, 0]):int(kidnap_index-1), 3])
                
            phase_recover_1 = np.all(abs(estimate_error[n, idx_1:, 0]) < phase_recover_thr)
            step_length_recover_1 = np.all(x[n, idx_1:, 2] < step_lengths[idx_1:] + step_length_est_error_mean + step_length_recover_thr)\
                                    and np.all(x[n, idx_1:, 2] > step_lengths[idx_1:] + step_length_est_error_mean - step_length_recover_thr)
            ramp_recover_1 = np.all(x[n, idx_1:, 3] < ramps[idx_1:] + ramp_est_error_mean + ramp_recover_thr)\
                            and np.all(x[n, idx_1:, 3] > ramps[idx_1:] + ramp_est_error_mean - ramp_recover_thr)
            
            phase_recover_3 = np.all(abs(estimate_error[n, idx_3:, 0]) < phase_recover_thr)
            step_length_recover_3 = np.all(x[n, idx_3:, 2] < step_lengths[idx_3:] + step_length_est_error_mean + step_length_recover_thr)\
                                    and np.all(x[n, idx_3:, 2] > step_lengths[idx_3:] + step_length_est_error_mean - step_length_recover_thr)
            ramp_recover_3 = np.all(x[n, idx_3:, 3] < ramps[idx_3:] + ramp_est_error_mean + ramp_recover_thr)\
                            and np.all(x[n, idx_3:, 3] > ramps[idx_3:] + ramp_est_error_mean - ramp_recover_thr)
            """
            
            # 2) Absolute estimate error approach
            """
            #phase_recover_1 = np.all(abs(estimate_error[n, int(kidnap_index + 1/np.average(phase_dots)/dt):, 0]) < phase_recover_thr)
            #phase_recover_3 = np.all(abs(estimate_error[n, int(kidnap_index + 3/np.average(phase_dots)/dt):, 0]) < phase_recover_thr)
            #step_length_recover_1 = np.all(abs(estimate_error[n, int(kidnap_index + 1/np.average(phase_dots)/dt):, 2]) < step_length_recover_thr)
            #step_length_recover_3 = np.all(abs(estimate_error[n, int(kidnap_index + 3/np.average(phase_dots)/dt):, 2]) < step_length_recover_thr)
            #ramp_recover_1 = np.all(abs(estimate_error[n, int(kidnap_index + 1/np.average(phase_dots)/dt):, 3]) < ramp_recover_thr)
            #ramp_recover_3 = np.all(abs(estimate_error[n, int(kidnap_index + 3/np.average(phase_dots)/dt):, 3]) < ramp_recover_thr)
            """

            # 3) Convergence approach
            phase_converge_dist[n, :] = phase_error(x[n,:, 0], x[0,:, 0])
            phase_recover_1 = np.all(phase_converge_dist[n,idx_1:] < phase_recover_thr)
            step_length_recover_1 = np.all(abs(x[n, idx_1:, 2] - x[0, idx_1:, 2]) < step_length_recover_thr)
            ramp_recover_1 = np.all(abs(x[n, idx_1:, 3] - x[0, idx_1:, 3]) < ramp_recover_thr)
            phase_recover_3 = np.all(phase_converge_dist[n,idx_3:] < phase_recover_thr)
            step_length_recover_3 = np.all(abs(x[n, idx_3:, 2] - x[0, idx_3:, 2]) < step_length_recover_thr)
            ramp_recover_3 = np.all(abs(x[n, idx_3:, 3] - x[0, idx_3:, 3]) < ramp_recover_thr)
            phase_recover_5 = np.all(phase_converge_dist[n,idx_5:] < phase_recover_thr)
            step_length_recover_5 = np.all(abs(x[n, idx_5:, 2] - x[0, idx_5:, 2]) < step_length_recover_thr)
            ramp_recover_5 = np.all(abs(x[n, idx_5:, 3] - x[0, idx_5:, 3]) < ramp_recover_thr)

            track_11 = (phase_recover_1 and step_length_recover_1 and ramp_recover_1)
            track_13 = (phase_recover_1 and step_length_recover_3 and ramp_recover_3)
            track_33 = (phase_recover_3 and step_length_recover_3 and ramp_recover_3)
            track_15 = (phase_recover_1 and step_length_recover_5 and ramp_recover_5)
            track_55 = (phase_recover_5 and step_length_recover_5 and ramp_recover_5)
            print("n:", n)
            print("i, j | track_ij = phase_i & step_len_j & ramp_j")
            print("1, 1 |  ", track_11, "  =  ", phase_recover_1, " &    ", step_length_recover_1, "  & ", ramp_recover_1)
            print("1, 3 |  ", track_13, "  =  ", phase_recover_1, " &    ", step_length_recover_3, "  & ", ramp_recover_3)
            print("3, 3 |  ", track_33, "  =  ", phase_recover_3, " &    ", step_length_recover_3, "  & ", ramp_recover_3)
            print("1, 5 |  ", track_15, "  =  ", phase_recover_1, " &    ", step_length_recover_5, "  & ", ramp_recover_5)
            print("5, 5 |  ", track_55, "  =  ", phase_recover_5, " &    ", step_length_recover_5, "  & ", ramp_recover_5)
            
            if track_11:
                r11 += 1
            if track_13:
                r13 += 1
            if track_33:
                r33 += 1
            if track_15:
                r15 += 1
            if track_55:
                r55 += 1
        
    robustness_11 = r11 / N * 100
    robustness_13 = r13 / N * 100
    robustness_33 = r33 / N * 100
    robustness_15 = r15 / N * 100
    robustness_55 = r55 / N * 100

    print("R_11(%) = ", robustness_11, "|| R_13(%) = ", robustness_13, "|| R_15(%) = ",
          robustness_15, "|| R_33(%) = ", robustness_33, "|| R_55(%) = ", robustness_55)
    print("---------------------------------------------------------------------------")

    if plot == True:
        # plot results
        tt = dt * np.arange(len(phases))

        plt.figure("State Estimate")
        plt.subplot(411)
        plt.title('EKFs-Bank Test')
        plt.plot(tt, phases, 'k-', linewidth=2)
        plt.plot(tt,  x[1:, :, 0].T, '--', alpha = 0.5)#
        plt.plot(tt,  x[0, :, 0].T, 'r--', linewidth=2)
        plt.ylabel('$\phi$')
        plt.legend(('ground truth', 'estimate'))
        #plt.legend(('ground truth', 'estimate'), bbox_to_anchor=(1, 1.05))
        plt.ylim([0, 2])
        plt.xlim([0, tt[-1]+0.1])
        plt.grid()
        plt.subplot(412)
        plt.plot(tt, phase_dots, 'k-', linewidth=2)
        plt.plot(tt, x[1:, :, 1].T, '--', alpha = 0.5)
        plt.plot(tt, x[0, :, 1].T, 'r--', linewidth=2)
        plt.ylabel('$\dot{\phi}~(s^{-1})$')
        plt.xlim([0, tt[-1]+0.1])
        plt.ylim([0.2, 1.5])
        plt.grid()
        plt.subplot(413)
        plt.plot(tt, step_lengths, 'k-', linewidth=2)
        plt.plot(tt, x[1:, :, 2].T, '--', alpha = 0.5)
        plt.plot(tt, x[0, :, 2].T, 'r--', linewidth=2)
        #plt.plot(tt[idx_1:], step_lengths[idx_1:] + step_length_est_error_mean + step_length_recover_thr, 'k--')
        #plt.plot(tt[idx_1:], step_lengths[idx_1:] + step_length_est_error_mean - step_length_recover_thr, 'k--')
        plt.ylabel('$l~(m)$')
        plt.xlim([0, tt[-1]+0.1])
        plt.ylim([0, 2])
        plt.grid()
        plt.subplot(414)
        plt.plot(tt, ramps, 'k-', linewidth=2)
        plt.plot(tt, x[1:, :, 3].T, '--', alpha = 0.5)
        plt.plot(tt, x[0, :, 3].T, 'r--', linewidth=2)
        #plt.plot(tt[idx_1:], ramps[idx_1:] + ramp_est_error_mean + ramp_recover_thr, 'k--')
        #plt.plot(tt[idx_1:], ramps[idx_1:] + ramp_est_error_mean - ramp_recover_thr, 'k--')
        plt.ylabel('$\\alpha~(deg)$')
        plt.xlim([0, tt[-1]+0.1])
        plt.ylim([-10, 10])
        plt.grid()
        plt.xlabel('time (s)')

        plt.figure("Estimation Errors")
        plt.subplot(411)
        plt.title('Estimation Errors')
        plt.plot(tt, abs(estimate_error[:, :, 0].T))
        plt.ylabel('$\phi$ error')
        plt.ylim([0, 0.5])
        plt.xlim([0, tt[-1]+0.1])
        plt.grid()
        plt.subplot(412)
        plt.plot(tt, abs(estimate_error[:, :, 1].T))
        plt.ylabel('$\dot{\phi}$ error (1/s)')
        plt.xlim([0, tt[-1]+0.1])
        plt.grid()
        plt.subplot(413)
        plt.plot(tt, abs(estimate_error[:, :, 2].T))
        plt.ylabel('$l$ error (m)')
        plt.xlim([0, tt[-1]+0.1])
        plt.grid()
        plt.subplot(414)
        plt.plot(tt, abs(estimate_error[:, :, 3].T))
        plt.ylabel('$\\alpha$ error (deg)')
        plt.xlim([0, tt[-1]+0.1])
        plt.grid()
        plt.xlabel('time (s)')
        
        plt.figure("Covergence Distance")
        plt.subplot(411)
        plt.title('Covergence Distance')
        plt.plot(tt, phase_converge_dist[1:,:].T, '--')
        plt.plot(tt[idx_1:], phase_recover_thr*np.ones(len(tt[idx_1:])), 'k-', alpha = 0.4, linewidth=2)
        plt.plot(idx_1*dt*np.ones(5), np.linspace(0,1,5), 'k-', alpha = 0.4, linewidth=2)
        plt.plot(idx_3*dt*np.ones(5), np.linspace(0,1,5), 'b-', alpha = 0.4, linewidth=2)
        plt.plot(idx_5*dt*np.ones(5), np.linspace(0,1,5), 'r-', alpha = 0.4, linewidth=2)
        plt.ylabel('$\Delta \phi$')
        plt.ylim([0, 0.5])
        plt.xlim([0, tt[-1]+0.1])
        plt.grid()
        plt.subplot(412)
        plt.plot(tt, abs(x[1:,:,1] - x[0,:,1]).T, '--')
        plt.ylabel('$\Delta \dot{\phi}$ (1/s)')
        plt.xlim([0, tt[-1]+0.1])
        plt.grid()
        plt.subplot(413)
        plt.plot(tt, abs(x[1:,:,2] - x[0,:,2]).T, '--')
        plt.plot(tt[idx_1:], step_length_recover_thr*np.ones(len(tt[idx_1:])), 'k-', alpha = 0.4, linewidth=2)
        plt.plot(idx_1*dt*np.ones(5), np.linspace(0,2,5), 'k-', alpha = 0.4, linewidth=2)
        plt.plot(idx_3*dt*np.ones(5), np.linspace(0,2,5), 'b-', alpha = 0.4, linewidth=2)
        plt.plot(idx_5*dt*np.ones(5), np.linspace(0,2,5), 'r-', alpha = 0.4, linewidth=2)
        plt.ylabel('$\Delta l$ (m)')
        plt.xlim([0, tt[-1]+0.1])
        plt.grid()
        plt.subplot(414)
        plt.plot(tt, abs(x[1:,:,3] - x[0,:,3]).T, '--')
        plt.plot(tt[idx_1:], ramp_recover_thr*np.ones(len(tt[idx_1:])), 'k-', alpha = 0.4, linewidth=2)
        plt.plot(idx_1*dt*np.ones(5), np.linspace(0,20,5), 'k-', alpha = 0.4, linewidth=2)
        plt.plot(idx_3*dt*np.ones(5), np.linspace(0,20,5), 'b-', alpha = 0.4, linewidth=2)
        plt.plot(idx_5*dt*np.ones(5), np.linspace(0,20,5), 'r-', alpha = 0.4, linewidth=2)
        plt.ylabel('$\Delta \\alpha$ (deg)')
        plt.xlim([0, tt[-1]+0.1])
        plt.grid()
        plt.xlabel('time (s)')
        plt.show()
    
    return (robustness_11, robustness_13, robustness_33, robustness_15, robustness_55)

def ekf_robustness(kidnap = True, heteroscedastic = False):
    total_trials = 0
    robustness_11 = 0
    robustness_13 = 0 
    robustness_33 = 0
    robustness_15 = 0
    robustness_55 = 0

    SE_phase_total = 0
    SE_step_length_total = 0
    SE_ramp_total = 0
    SE_directRamp_total = 0
    T_total = 0

    poor_step_length_est = 0
    poor_ramp_est = 0
    poor_task_est = 0

    for subject in Conti_subject_names(): 
    #for subject in ['AB09']:
        for trial in Conti_trial_names(subject):
        #for trial in ['s1x2i10', 's1i0']:
            if trial == 'subjectdetails':
                continue
            for side in ['left']:
                if nan_dict[subject][trial][side] == False:
                    print(subject + "/"+ trial + "/"+ side+ ": Trial skipped!")
                    continue
                total_trials = total_trials + 1
                
                if kidnap != False:
                    (R_11, R_13, R_33, R_15, R_55) = ekf_bank_test(subject, trial, side, 1, heteroscedastic, kidnap, plot = False)
                    robustness_11 += R_11
                    robustness_13 += R_13
                    robustness_33 += R_33
                    robustness_15 += R_15
                    robustness_55 += R_55
                    print("**Current Average R_11 = %4.1f %%" % (robustness_11 / total_trials), 
                          "|| R_13 = %4.1f %%" % (robustness_13 / total_trials),
                          "|| R_15 = %4.1f %%" % (robustness_15 / total_trials),
                          "|| R_33 = %4.1f %%" % (robustness_33 / total_trials),
                          "|| R_55 = %4.1f %%" % (robustness_55 / total_trials))
                else:
                    SE_phase, _, SE_step_length, SE_ramp, SE_directRamp, T = ekf_test(subject, trial, side, heteroscedastic, kidnap, plot = False)
                    SE_phase_total += SE_phase
                    SE_step_length_total += SE_step_length
                    SE_ramp_total += SE_ramp
                    SE_directRamp_total += SE_directRamp
                    T_total += T

                    print("Current RMSE phase = %5.3f" % np.sqrt(SE_phase_total/T_total))
                    print("Current RMSE step_length = %5.3f" % np.sqrt(SE_step_length_total/T_total))
                    print("----------------------------------------------------------------")
                    #print("Current RMSE ramp = %5.3f" % np.sqrt(SE_ramp_total/T_total))
                    #print("Current RMSE directRamp = %5.3f" % np.sqrt(SE_directRamp_total/T_total))

                    """
                    if RMSE_phase > 0.05 or RMSE_step_length > 0.1 or RMSE_ramp > 2:
                        print(subject, "/", trial, '/', side, ": RMSE exceeds the threshold!")
                        print("RMSE phase = %5.3f" % RMSE_phase)
                        print("RMSE step_length = %5.3f" % RMSE_step_length)
                        print("RMSE ramp = %5.3f" % RMSE_ramp)
                        print("==================================================")
                    
                    # Number of trials whose task esimates are poor
                    if RMSE_step_length > 0.1:
                        poor_step_length_est += 1
                    if RMSE_ramp > 2:
                        poor_ramp_est += 1
                    if RMSE_step_length > 0.1 and RMSE_ramp > 2:
                        poor_task_est += 1
                    """

    if kidnap != False:
        robustness_11 = robustness_11 / total_trials
        robustness_13 = robustness_13 / total_trials
        robustness_33 = robustness_33 / total_trials
        robustness_15 = robustness_15 / total_trials
        robustness_55 = robustness_55 / total_trials

        print("==========================================")
        print("Overall Robustness_11 = %4.1f %%" % robustness_11)
        print("Overall Robustness_13 = %4.1f %%" % robustness_13)
        print("Overall Robustness_33 = %4.1f %%" % robustness_33)
        print("Overall Robustness_15 = %4.1f %%" % robustness_15)
        print("Overall Robustness_55 = %4.1f %%" % robustness_55)
    else:
        print("Total RMSE phase = %5.3f" % np.sqrt(SE_phase_total/T_total))
        print("Total RMSE step_length = %5.3f" % np.sqrt(SE_step_length_total/T_total))
        print("Total RMSE ramp = %5.3f" % np.sqrt(SE_ramp_total/T_total))
        print("Total RMSE directRamp = %5.3f" % np.sqrt(SE_directRamp_total/T_total))

        """
        print("Max RMSE phase = %5.3f" % np.max(RMSE_phase_list))
        print("Max RMSE step_length = %5.3f" % np.max(RMSE_step_length_list))
        print("Max RMSE ramp = %5.3f" % np.max(RMSE_ramp_list))

        print("Percentage of trials with poorly estimated step lengths: %4.2f %%" % (poor_step_length_est / total_trials * 100))
        print("Percentage of trials with poorly estimated ramps: %4.2f %%" % (poor_ramp_est / total_trials * 100))
        print("Percentage of trials with poorly estimated step lengths and ramps %4.2f %%" % (poor_task_est / total_trials * 100))

        plt.figure()
        plt.plot(RMSE_step_length_list, RMSE_ramp_list, 'r.')
        plt.plot(0.1 * np.ones(50), np.linspace(0,5,50), 'k--')
        plt.plot(np.linspace(0,0.5,50), 2 * np.ones(50), 'k--')
        plt.xlim([0, max(0.2, np.max(RMSE_step_length_list)+0.05)])
        plt.ylim([0, max(3, np.max(RMSE_ramp_list)+0.2)])
        plt.xlabel("RMSE step_length")
        plt.ylabel("RMSE ramp")
        plt.grid()
        plt.show()
        """

if __name__ == '__main__':
    subject = 'AB10'
    trial = 's1x2d5'
    side = 'right'

    if nan_dict[subject][trial][side] == False:
        print(subject + "/"+ trial + "/"+ side+ ": This trial should be skipped!")

    ekf_test(subject, trial, side, heteroscedastic = False, kidnap = False, plot = True)
    #ekf_bank_test(subject, trial, side, N = 5, heteroscedastic = False, kidnap = [0, 1, 2], plot = True)
    #ekf_robustness(kidnap = [0, 1, 2], heteroscedastic = False)
    #ekf_robustness(kidnap = False, heteroscedastic = False)
