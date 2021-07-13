"""
Holonomic Phase Variable for Stair Ascent.
"""
import sys, time, os
from flexsea import flexsea as flex
from flexsea import fxEnums as fxe
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(r'/home/pi/OSL/locoAPI/')       # Path to Loco module
sys.path.append(r'/usr/share/python3-mscl/')    # Path of the MSCL - API for the IMU
import locoOSL as loco                          # Module from Locolab
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
fxs.start_streaming(kneID, freq = 100, log_en = True)
fxs.start_streaming(ankID, freq = 100, log_en = True)
time.sleep(1)                           # Healthy pause before using ActPacks or IMU

try:    
    # For gain details check https://dephy.com/wiki/flexsea/doku.php?id=controlgains
    # G_K = {"kp": 40, "ki": 400, "K": 600, "B": 300, "FF": 128}  # Knee controller gains
    G_K = {"kp": 40, "ki": 400, "K": 300, "B": 1600, "FF": 128}  # Knee controller gains
    # G_A = {"kp": 40, "ki": 400, "K": 600, "B": 300, "FF": 128}  # Ankle controller gains
    G_A = {"kp": 40, "ki": 400, "K": 300, "B": 1600, "FF": 128}  # Ankle controller gains

    
    # load trajectory and encoder map
    tra = loco.loadTrajectory(trajectory = 'stair_ascent')
    refAnk = tra["ank30"]
    refKne = tra["kne30"]
    refHip = tra["hip30"]
    refPha = tra["phase30"]
    refTime = tra["time"]
    maxHip = np.amax(refHip)
    minHip = np.amin(refHip)
    encMap = loco.read_enc_map() 

    kneSta  = fxs.read_device(kneID)
    ankSta  = fxs.read_device(ankID)
    IMUPac  = IMU.getDataPackets(TIMEOUT)
    dataOSL = loco.read_OSL(kneSta, ankSta, IMUPac,encMap)
    misclog = {'PV': 0, 'refAnk': 0, 'refKnee': 0, 'State': 0}
    logger = loco.ini_log({**dataOSL,**misclog},sensors="all_sensors",trialName="PV_StairAscent")

    fxs.set_gains(ankID, G_A["kp"], G_A["ki"], 0, G_A["K"], G_A["B"], G_A["FF"])
    fxs.set_gains(kneID, G_K["kp"], G_K["ki"], 0, G_K["K"], G_K["B"], G_K["FF"])

    kne_buffer = []
    ank_buffer = []
    loadcell_z = []
    pv = 0

    print('\nCreating Buffers for Digital Signal Filters')
    time.sleep(2)
    for i in range(3):
        time.sleep(.5)
        ankMotCou, kneMotCou = loco.joi2motTic(encMap, 0, 0)
        kne_buffer.append(kneMotCou)
        ank_buffer.append(ankMotCou)
        loadcell_z.append(dataOSL['loadCelFz'][0])
    print('\nCAUTION: Moving the OSL to the initial configuration in 2 seconds')
    time.sleep(2)
    # Setting control mode and approaching initial position [Be careful fo high gains]
    ankMotCou = int(np.median(ank_buffer))
    kneMotCou = int(np.median(kne_buffer))
    load_z = int(np.median(loadcell_z))
    fxs.send_motor_command(ankID, fxe.FX_IMPEDANCE, ankMotCou)
    fxs.send_motor_command(kneID, fxe.FX_IMPEDANCE, kneMotCou)

    #Reparamatarize Trajectories

    refKnee = np.interp(refTime,refPha, refKne)
    refAnk = np.interp(refTime,refPha, refAnk)

    # print(ankMotCou)
    # print(ankSta.mot_ang)
    # print(kneMotCou)
    # print(kneSta.mot_ang)

    #Set PV Params
    prevState = 1
    prevPV = 0.3
    sm = 0.5
    qhm = -10
    mhf = False
   #getPhaseVariable_vStairAscent(dataOSL, prevState, prevPV, sm, qhm, maxHip, minHip, mhf, load_z, c = 0.53, qPo = -8.4, FCThres = 24):
    PV, prevState, sm, qhm, mhf = loco.getPhaseVariable_Stairs(dataOSL, prevState, prevPV, sm, qhm, maxHip, minHip, mhf, load_z, c = 0.53, qPo = -8.4, FCThres = 24)

    if input('\n\nAbout to walk. Would you like to continue? (y/n): ').lower() == 'y':
        print("\n Let's walk!")
    else:
        sys.exit("User stopped the execution")

    while True:
        # Read OSL
        kneSta  = fxs.read_device(kneID)
        ankSta  = fxs.read_device(ankID)
        IMUPac  = IMU.getDataPackets(TIMEOUT)  
        dataOSL = loco.read_OSL(kneSta, ankSta, IMUPac, logger['ini_time'],encMap)

        #Update Loadcell buffer and median filter
        if dataOSL['loadCelFz']>50:
            loadcell_z = [loadcell_z[0],loadcell_z[0], loadcell_z[1]]
        else:
            loadcell_z = [dataOSL['loadCelFz'],loadcell_z[0], loadcell_z[1]]

        dataOSL['loadCelFz'] =  int(np.median(loadcell_z))
        if prevState == 3 or prevState == 4:
            mhf,maxHip,temp_traj = loco.maximumHipFlexionDetection(dataOSL,mhf, maxHip, temp_traj,thresh = 0,mindist = 50)
            
        PV, prevState, sm, qhm, mhf = loco.getPhaseVariable_Stairs(dataOSL, prevState, PV, sm, qhm, maxHip, minHip, mhf, dataOSL['loadCelFz'])
        indxPV = PV*1000

        misclog['PV'] = PV
        misclog['refAnk'] = refAnk[PV]
        misclog['refKnee'] = refKne[PV]
        misclog['State'] = prevState
        loco.log_OSL({**dataOSL,**misclog}, logger)


        #Calculate Virtual Constraints

        
        

        # #Analytical
        # knee_pos, ankle_pos = loco.getVirtualConstraints(refKne,refAnk, PV, len(refKne))

        #Lookup Table
        knee_pos = refKne[indxPV]
        ankle_pos = refAnk[indxPV]

        #Update  motor tick buffers and median filter
        ankMotCou, kneMotCou = loco.joi2motTic(encMap, knee_pos, ankle_pos)
        kne_buffer = [kneMotCou,kne_buffer[0],kne_buffer[1]]
        ank_buffer = [ankMotCou,ank_buffer[0],ank_buffer[1]]

        #Move OSL
        ankMotCou = int(np.median(ank_buffer))
        kneMotCou = int(np.median(kne_buffer))
        fxs.send_motor_command(ankID, fxe.FX_IMPEDANCE, ankMotCou)
        fxs.send_motor_command(kneID, fxe.FX_IMPEDANCE, kneMotCou)

        
        print("Thigh: {:>10.4f} [deg] || LoadCellFz: {:>10.4f} [N] || Ph. Va.: {:10.4f} || Ankle: {:10.4f} [deg]"
            " || Knee: {:10.4f} [deg]".format(dataOSL['ThighSagi']*180/np.pi, dataOSL['loadCelFz'], indxPV, 
            refAnk[indxPV], refKne[indxPV]) )
        c = c+1

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