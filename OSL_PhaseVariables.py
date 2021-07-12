"""
Code to test drive the phase variable for level-ground walking 
"""
import sys, time, os
from flexsea import flexsea as flex
from flexsea import fxEnums as fxe
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(r'/home/pi/OSL/locoAPI/')       # Path to Loco module
#sys.path.append(r'/home/pi/.local/bin')
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
kneID = fxs.open(port = KNE_PORT, baud_rate = 230400, log_level = 6)
ankID = fxs.open(port = ANK_PORT, baud_rate = 230400, log_level = 6)

# Initialize IMU - The sample rate (SR) of the IMU controls the SR of the OSL 
connection = msl.Connection.Serial(IMU_PORT, 921600)
IMU = msl.InertialNode(connection)
IMU.setToIdle()
packets = IMU.getDataPackets(TIMEOUT)   # Clean the internal circular buffer.

# Set streaming        
IMU.resume()
fxs.start_streaming(kneID, freq = 100, log_en = False)
fxs.start_streaming(ankID, freq = 100, log_en = False)
time.sleep(1)                           # Healthy pause before using ActPacks or IMU

# Soft start
G_K = {"kp": 40, "ki": 400, "K": 10, "B": 0, "FF": 128}  # Knee controller gains
G_A = {"kp": 40, "ki": 400, "K": 10, "B": 0, "FF": 128}  # Ankle controller gains
fxs.set_gains(ankID, G_A["kp"], G_A["ki"], 0, G_A["K"], G_A["B"], G_A["FF"])
fxs.set_gains(kneID, G_K["kp"], G_K["ki"], 0, G_K["K"], G_K["B"], G_K["FF"])
fxs.send_motor_command(ankID, fxe.FX_IMPEDANCE, fxs.read_device(ankID).mot_ang)
fxs.send_motor_command(kneID, fxe.FX_IMPEDANCE, fxs.read_device(kneID).mot_ang)
time.sleep(2/100)

# ------------------ MAIN LOOP -----------------------------------------------------------
try:
    # For gain details check https://dephy.com/wiki/flexsea/doku.php?id=controlgains
    G_K = {"kp": 40, "ki": 400, "K": 30, "B": 160, "FF": 128}  # Knee controller gains
    G_A = {"kp": 40, "ki": 400, "K": 30, "B": 160, "FF": 128}  # Ankle controller gains

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
    maxHip = np.amax(refHip)
    minHip = np.amin(refHip)
    # Create encoder map
    kneSta  = fxs.read_device(kneID)
    ankSta  = fxs.read_device(ankID)
    IMUPac  = IMU.getDataPackets(TIMEOUT)
    encMap  = loco.read_enc_map( loco.read_OSL(kneSta, ankSta, IMUPac) ) 

    dataOSL = loco.read_OSL(kneSta, ankSta, IMUPac)
    misclog = {'PV': 0.0, 'refAnk': 0.0, 'refKnee': 0.0}
    logger = loco.ini_log({**dataOSL,**misclog},sensors="all_sensors",trialName="PV_TwoStates_1000Hz")

    kne_buffer = []
    ank_buffer = []
    loadcell_z = []
    pv = 0

    print('\nCreating Buffers for Digital Signal Filters')
    time.sleep(2)
    for i in range(3):
        time.sleep(.5)
        ankMotCou, kneMotCou = loco.joi2motTic(encMap, refKne[0], refAnk[0])
        kne_buffer.append(kneMotCou)
        ank_buffer.append(ankMotCou)
        loadcell_z.append(dataOSL['loadCelFz'])
    print('\nCAUTION: Moving the OSL to the initial configuration in 2 seconds')
    time.sleep(2)
    # Setting control mode and approaching initial position [Be careful fo high gains]
    ankMotCou = int(np.median(ank_buffer))
    kneMotCou = int(np.median(kne_buffer))
    dataOSL['loadCelFz'] = np.median(loadcell_z)
    fxs.send_motor_command(ankID, fxe.FX_IMPEDANCE, ankMotCou)
    fxs.send_motor_command(kneID, fxe.FX_IMPEDANCE, kneMotCou)
    # print(ankMotCou)
    # print(ankSta.mot_ang)
    # print(kneMotCou)
    # print(kneSta.mot_ang)
    
    if input('\n\nAbout to walk. Would you like to continue? (y/n): ').lower() == 'y':
        print("\n Let's walk!")
    else:
        sys.exit("User stopped the execution")
    loops = 1
    start_time = time.time()
    start_time_ns = time.perf_counter_ns()
    while True:        
        # Read OSL
        kneSta  = fxs.read_device(kneID)
        ankSta  = fxs.read_device(ankID)
        IMUPac  = IMU.getDataPackets(TIMEOUT)  
        dataOSL = loco.read_OSL(kneSta, ankSta, IMUPac, logger['ini_time'], encMap)

        #Update Loadcell buffer and median filter
        if dataOSL['loadCelFz']>50:
            loadcell_z = [loadcell_z[0],loadcell_z[0], loadcell_z[1]]
        else:
            loadcell_z = [dataOSL['loadCelFz'],loadcell_z[0], loadcell_z[1]]

        dataOSL['loadCelFz'] =  int(np.median(loadcell_z))
        
        # Estimate percentage within the gait cycle - Output from 0 to 998
        pv = int(loco.getPhaseVariable_vTwoStates(dataOSL, maxHip, minHip, dataOSL['loadCelFz'],FCThr = 24) * 998)
        # pv = 0

        misclog['PV'] = pv
        misclog['refAnk'] = refAnk[pv]
        misclog['refKnee'] = refKne[pv]
        loco.log_OSL({**dataOSL,**misclog}, logger)

        #Update  motor tick buffers and median filter
        ankMotCou, kneMotCou = loco.joi2motTic(encMap, refKne[pv], refAnk[pv])
        kne_buffer = [kneMotCou,kne_buffer[0],kne_buffer[1]]
        ank_buffer = [ankMotCou,ank_buffer[0],ank_buffer[1]]

        ankMotCou = int(np.median(ank_buffer))
        kneMotCou = int(np.median(kne_buffer))

        # Move the OSL        
        fxs.send_motor_command(ankID, fxe.FX_IMPEDANCE, ankMotCou)
        fxs.send_motor_command(kneID, fxe.FX_IMPEDANCE, kneMotCou)
        curTime = time.time()
        print(curTime - start_time)
        start_time = curTime
        print('Delta Time (ms):', (time.perf_counter_ns() - start_time_ns)/1e6)
        start_time_ns = time.perf_counter_ns()
        # print(kne_buffer)
        # print(kneMotCou)

        print("Thigh: {:>10.4f} [deg] || LoadCellFz: {:>10.4f} [N] ||LoadCellFx raw: {:>10.4f} [N] || Ph. Va.: {:10.4f} "
             "|| Ref. Ankle: {:10.4f} [deg] || Ref. Knee: {:10.4f} [deg] || Knee mot ang: {:>10.4f} [ticks]".format(
                 dataOSL['ThighSagi']*180/np.pi, dataOSL['loadCelFz'], dataOSL['loadCelFx'], pv, 
                 refAnk[pv], refKne[pv], kneMotCou) )
except KeyboardInterrupt:
        print('\n*** OSL shutting down ***\n')
finally:        
    # Do anything but turn off the motors at the end of the program
    fxs.send_motor_command(ankID, fxe.FX_NONE, 0)
    fxs.send_motor_command(kneID, fxe.FX_NONE, 0)
    IMU.setToIdle()
    time.sleep(0.5)    
    fxs.close(ankID)
    fxs.close(kneID)  
    print('Communication with ActPacks closed and IMU set to idle')