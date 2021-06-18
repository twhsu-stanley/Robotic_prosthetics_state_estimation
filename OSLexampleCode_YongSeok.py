"""
Code to test drive the phase variable for level-ground walking 
"""
import sys, time, os
from flexsea import flexsea as flex
from flexsea import fxEnums as fxe
import matplotlib.pyplot as plt
import numpy as np
import csv
sys.path.append(r'/home/pi/yongseok_EKF/')   # CHANGE THIS PATH
from ekf_phase_estimator import EKF_ph_est
from NN_model_framework import evaluateNN
sys.path.append(r'/home/pi/OSL/locoAPI/')       # Path to Loco module
sys.path.append(r'/home/pi/.local/bin')
sys.path.append(r'/usr/share/python3-mscl/')    # Path of the MSCL - API for the IMU
import loco_OSL_fix as loco                          # Module from Locolab
import mscl as msl                              # Module from Microstrain

# -----------------TODO: change these constants to match your setup ----------------------
ANK_PORT = r'/dev/ttyACM1'
KNE_PORT = r'/dev/ttyACM0'
IMU_PORT = r'/dev/ttyUSB0'
TIMEOUT  = 500                          #Timeout in (ms) to read the IMU
fxs = flex.FlexSEA()

# ------------------ INITIALIZATION ------------------------------------------------------
# Connect with knee and ankle actuator packs
kneID = fxs.open(port = KNE_PORT, baud_rate = 230400, log_level = 0)
ankID = fxs.open(port = ANK_PORT, baud_rate = 230400, log_level = 0)

# Initialize IMU - The sample rate (SR) of the IMU controls the SR of the OSL 
connection = msl.Connection.Serial(IMU_PORT, 921600)
IMU = msl.InertialNode(connection)
IMU.setToIdle()
packets = IMU.getDataPackets(TIMEOUT)   # Clean the internal circular buffer.

# Set streaming        
IMU.resume()
fxs.start_streaming(kneID, freq = 1000, log_en = True)
fxs.start_streaming(ankID, freq = 1000, log_en = True)
time.sleep(1)                           # Healthy pause before using ActPacks or IMU

filename = '/home/pi/csv_log/log1.csv'
with open(filename,"w",newline="\n") as fd_l:
    writer_l = csv.writer(fd_l)
    print(writer_l)
    # ------------------ MAIN LOOP -----------------------------------------------------------
    try:
        # For gain details check https://dephy.com/wiki/flexsea/doku.php?id=controlgains
        # G_K = {"kp": 40, "ki": 400, "K": 600, "B": 300, "FF": 128}  # Knee controller gains
        G_K = {"kp": 40, "ki": 400, "K": 300, "B": 1600, "FF": 128}  # Knee controller gains
        # G_A = {"kp": 40, "ki": 400, "K": 600, "B": 300, "FF": 128}  # Ankle controller gains
        G_A = {"kp": 40, "ki": 400, "K": 300, "B": 1600, "FF": 128}  # Ankle controller gains
        kneSta  = fxs.read_device(kneID)
        ankSta  = fxs.read_device(ankID)
        IMUPac  = IMU.getDataPackets(TIMEOUT)
        fxs.set_gains(ankID, G_A["kp"], G_A["ki"], 0, G_A["K"], G_A["B"], G_A["FF"])
        fxs.set_gains(kneID, G_K["kp"], G_K["ki"], 0, G_K["K"], G_K["B"], G_K["FF"])
        
        # load trajectory and encoder map
        tra = loco.loadTrajectory(trajectory = 'walking')
        refAnk = tra["ankl"]
        refKne = tra["knee"]
        refHip = tra["hip_"]
        refPha = tra["phas"]
        
        # Create encoder map
        kneSta  = fxs.read_device(kneID)
        ankSta  = fxs.read_device(ankID)
        IMUPac  = IMU.getDataPackets(TIMEOUT)
        encMap  = loco.read_enc_map( loco.read_OSL(kneSta, ankSta, IMUPac) )
        dataOSL = loco.read_OSL(kneSta, ankSta, IMUPac)
        misclog = {'PV': [0.0, 'phase'], 'refAnk': [0.0,'deg'], 'refKnee': [0.0,'deg']}
        logger = loco.ini_log({**dataOSL,**misclog},sensors="all_sensors",trialName="PV_TwoStates_1000Hz")
        
        #############*****#############
        switch = 'K' # 'N' or 'K'
        initial_state = np.array([0, 0.86214417,1.16825315, 0,0,0,0])
        dataset_location = './local-storage/'
        ekf_est = EKF_ph_est(initial_state, switch, dataset_location)
        #############*****#############
        
        if input('\n\nAbout to walk. Would you like to continue? (y/n): ').lower() == 'y':
            print("\n Let's walk!")
            #############*****#############
            t0 = time.time()
            prev_t = t0
            time_memory = np.array([0])
            #############*****#############
        else:
            sys.exit("User stopped the execution")
        loops = 1
        ndx = 0
        while True:
            # Read OSL
            kneSta  = fxs.read_device(kneID)
            ankSta  = fxs.read_device(ankID)
            IMUPac  = IMU.getDataPackets(TIMEOUT)
            dataOSL = loco.read_OSL(kneSta, ankSta, IMUPac, logger['ini_time'], encMap)
            
            ankle_m = dataOSL['ankJoiPos'][0]*180/np.pi
            knee_m = dataOSL['kneJoiPos'][0]*180/np.pi
            thigh_m = dataOSL['ThighSagi'][0]*180/np.pi
            ankle_moment_m = dataOSL['ankJoiTor'][0]
            knee_moment_m = dataOSL['kneJoiTor'][0]
            # if ankle_m < -180:
            #     ankle_m += 360
            # if ankle_m > 180:
            #     ankle_m -= 360
            assert (ankle_m <= 180)
            assert (ankle_m >= -180)
            # # if knee_m < -180:
            #     knee_m += 360
            # if knee_m > 180:
            #     knee_m -= 360
            # assert (knee_m <= 180)
            # assert (knee_m >= -180)
            #convert to ekf coord.
            measurement =  np.array([-ankle_m, -knee_m, thigh_m]) #? sign of thigh
            
            #############*****#############
            # measurement: (3,) shape numpy array
            # [ankle, knee, thigh]
            current_t = time.time()
            dt = current_t - prev_t
            prev_t = current_t
            ekf_est.EKF.time_step_update(dt)
            ankle_angle_e, knee_angle_e, t, ankle_moment_e, knee_moment_e, state_e, thigh_angle_e = ekf_est.update(measurement)
            ankle_angle_e = - ankle_angle_e
            knee_angle_e = -knee_angle_e  
            ankle_moment_e = - ankle_moment_e
            knee_moment_e = - knee_moment_e
            thigh_angle_e = - thigh_angle_e
            ankle_angle_e[0] = np.clip(ankle_angle_e[0],-10.34,19.65)
            knee_angle_e[0] = np.clip(knee_angle_e[0],-70,0)
            #back to original coord.
            ndx += 1
            if ndx > 998:
                ndx =0
            writer_l.writerow([
                current_t-t0,
                ankle_angle_e[0],
                knee_angle_e[0],
                thigh_angle_e[0],
                ankle_moment_e[0],
                knee_moment_e[0],
                state_e[0],
                state_e[1],
                state_e[2],
                state_e[3],
                state_e[4],
                state_e[5],
                state_e[6],
                measurement[0],
                measurement[1],
                measurement[2],
                ankle_moment_m,
                knee_moment_m,
                refAnk[ndx],
                refKne[ndx],
                ekf_est.EKF.sigma[0,0],
                ekf_est.EKF.sigma[1,1],
                ekf_est.EKF.sigma[0,1]
            ])
            # time_memory = np.append(time_memory,current_t-t0)
            # ankle_angle_memory = np.append(ankle_angle_memory,ankle_angle)
            # knee_angle_memory = np.append(knee_angle_memory,knee_angle)
            # ankle_moment_memory = np.append(ankle_moment_memory,ankle_moment)
            # knee_moment_memory = np.append(knee_moment_memory,knee_moment)
            # #############*****#############
            # Estimate percentage within the gait cycle - Output from 0 to 998
            # misclog['PV'][0] = state[0]
            # misclog['refAnk'][0] = ankle_angle
            # misclog['refKnee'][0] = knee_angle
            # loco.log_OSL({**dataOSL,**misclog}, logger)
            if ndx%100 == 0:
                print('Ankle measurement: %.2f deg. Knee measurement: %.2f deg.'%(ankle_m,knee_m))
            dry_run = False
            if dry_run:
                ankMotCou, kneMotCou = loco.joi2motTic(encMap, knee_m, ankle_m)
                if np.abs(dataOSL['ankMotCur'][0]) > 10 or np.abs(dataOSL['kneMotCur'][0]) > 10:
                    err_string = 'Current exceed the limit.'
                    raise ValueError(err_string)
                if ndx%100 == 0:
                    print("Thigh: {:>10.4f} [deg] || LoadCellFz: {:>10.4f} [N] || Ph. Va.: {:10.4f} "
                    "|| Ref. Ankle: {:10.4f} [deg] || Ref. Knee: {:10.4f} [deg]".format(
                        thigh_m, dataOSL['loadCelFz'][0], state_e[0], 
                        ankle_m, knee_m ))
            else:
                ankMotCou, kneMotCou = loco.joi2motTic(encMap, knee_angle_e[0], ankle_angle_e[0])
                if ndx%100 == 0:
                    print("Thigh: {:>10.4f} [deg] || LoadCellFz: {:>10.4f} [N] || Ph. Va.: {:10.4f} "
                    "|| Ref. Ankle: {:10.4f} [deg] || Ref. Knee: {:10.4f} [deg]".format(
                        thigh_m, dataOSL['loadCelFz'][0], state_e[0], 
                        ankle_angle_e[0], knee_angle_e[0]) )
            # Move the OSL
            fxs.send_motor_command(ankID, fxe.FX_IMPEDANCE, ankMotCou)
            fxs.send_motor_command(kneID, fxe.FX_IMPEDANCE, kneMotCou)
    except KeyboardInterrupt:
            print('\n*** OSL shutting down ***\n')
    finally:        
        # Do anything but turn off the motors at the end of the program
        fxs.send_motor_command(ankID, fxe.FX_NONE, 0)
        fxs.send_motor_command(kneID, fxe.FX_NONE, 0)
        IMU.setToIdle()
        fxs.close(ankID)
        fxs.close(kneID)  
        print('Communication with ActPacks closed and IMU set to idle')
        #with open(filename,'wb') as file:
        #	pickle.dump(model,file)