"""
Code to load walking trajectory from Winter's and follow it with a specific impedance 
"""
import sys, time
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
fxs.start_streaming(kneID, freq = 100, log_en = False)
fxs.start_streaming(ankID, freq = 100, log_en = False)
time.sleep(1)                           # Healthy pause before using ActPacks or IMU

# # Soft start
# G_K = {"kp": 40, "ki": 400, "K": 10, "B": 0, "FF": 128}  # Knee controller gains
# G_A = {"kp": 40, "ki": 400, "K": 10, "B": 0, "FF": 128}  # Ankle controller gains
# fxs.set_gains(ankID, G_A["kp"], G_A["ki"], 0, G_A["K"], G_A["B"], G_A["FF"])
# fxs.set_gains(kneID, G_K["kp"], G_K["ki"], 0, G_K["K"], G_K["B"], G_K["FF"])
# fxs.send_motor_command(ankID, fxe.FX_IMPEDANCE, fxs.read_device(ankID).mot_ang)
# fxs.send_motor_command(kneID, fxe.FX_IMPEDANCE, fxs.read_device(kneID).mot_ang)
# time.sleep(2/100)

# ------------------ MAIN LOOP -----------------------------------------------------------
# For gain details check https://dephy.com/wiki/flexsea/doku.php?id=controlgains
# G_K = {"kp": 40, "ki": 400, "K": 600, "B": 300, "FF": 128}  # Knee controller gains
G_K = {"kp": 40, "ki": 400, "K": 1000, "B": 3500, "FF": 1}  # Knee controller gains
# G_A = {"kp": 40, "ki": 400, "K": 600, "B": 300, "FF": 128}  # Ankle controller gains
G_A = {"kp": 40, "ki": 400, "K": 600, "B": 300, "FF": 128}  # Ankle controller gains
# load trajectory
tra = loco.loadTrajectory(trajectory = 'walking')
# steps and the factor to stretch time (e.g., factor = 2 makes a 0.5x speed motion)
ste = 1     # Steps
fac = 4     # Factor
sampleTime = 10e-3
samples = int( tra['time'][-1]/sampleTime )*fac*ste
fac = round(999/(samples/ste))

try: 
    # Get encoder map
    kneSta  = fxs.read_device(kneID)
    ankSta  = fxs.read_device(ankID)
    IMUPac  = IMU.getDataPackets(TIMEOUT)
    dataOSL = loco.read_OSL(kneSta, ankSta, IMUPac)
    encMap  = loco.read_enc_map( dataOSL ) 

    # Initial data from the OSL
    kneSta  = fxs.read_device(kneID)
    ankSta  = fxs.read_device(ankID)
    IMUPac  = IMU.getDataPackets(TIMEOUT)

    dataOSL = loco.read_OSL(kneSta, ankSta, IMUPac)
    logger = loco.ini_log(dataOSL,sensors="all_sensors",trialName="impedance")
    print('Log Initialized')
    fxs.set_gains(ankID, G_A["kp"], G_A["ki"], 0, G_A["K"], G_A["B"], G_A["FF"])
    fxs.set_gains(kneID, G_K["kp"], G_K["ki"], 0, G_K["K"], G_K["B"], G_K["FF"])

    print('\nCAUTION: Moving the OSL to the initial configuration in 2 seconds')
    time.sleep(2)
    # Setting control mode and approaching initial position [Be careful fo high gains]
    ankMotCou, kneMotCou = loco.joi2motTic(encMap, tra['knee'][0], tra['ankl'][0])
    fxs.send_motor_command(ankID, fxe.FX_IMPEDANCE, ankMotCou)
    fxs.send_motor_command(kneID, fxe.FX_IMPEDANCE, kneMotCou)
    
    if input('\n\nWould you like to continue? (y/n): ').lower() == 'y':
        print('\nReady to walk!')
    else:
        sys.exit("User stopped the execution")

    ank_ref = []
    ank_mea = []
    kne_ref = []
    kne_mea = []
    initialTime = 0.0
    for i in range(samples - 1):
        iniTime = time.perf_counter()
        # Loading reference trajectories
        ank_i = tra['ankl'][ (i*fac) % 998 ]
        kne_i = tra['knee'][ (i*fac) % 998 ] 
        # Sending reference trajectories
        ankMotCou, kneMotCou = loco.joi2motTic(encMap, kne_i, ank_i)
        fxs.send_motor_command(ankID, fxe.FX_IMPEDANCE, ankMotCou)
        fxs.send_motor_command(kneID, fxe.FX_IMPEDANCE, kneMotCou)
        # Read and log OSL data        
        kneSta  = fxs.read_device(kneID)
        ankSta  = fxs.read_device(ankID)
        IMUPac  = IMU.getDataPackets(TIMEOUT)
        dataOSL = loco.read_OSL(kneSta, ankSta, IMUPac, logger['ini_time'],encMap)
        loco.log_OSL(dataOSL, logger)
        # loco.log_OSL(dataOSL,OSL["log"])
        ank_ref.append( ank_i )
        ank_mea.append(dataOSL['ankMotPos'])
        kne_ref.append( kne_i )
        kne_mea.append(dataOSL['kneMotPos'])        
        deltaTime = time.perf_counter() - iniTime    

    fig, ax = plt.subplots(2,1)
    ax[0].plot(ank_ref, label = 'Ank. ref.')
    ax[0].plot(ank_mea, label = 'Ank. mea.')
    # ax[0].set_xlabel('Sample')
    ax[0].set_ylabel('Ankle angle [deg]')
    ax[0].set_ylim([-30,20])
    ax[0].legend()
    ax[0].set_title('Ankle-knee Motion')
    ax[1].plot(kne_ref, label = 'Kne. ref.')
    ax[1].plot(kne_mea, label = 'Kne. mea.')
    ax[1].set_ylim([-70,10])
    ax[1].set_xlabel('Sample (10ms sample time)')
    ax[1].set_ylabel('Knee angle [deg]')
    ax[1].legend()
    plt.savefig('ankle-knee_position.png')

    fxs.send_motor_command(ankID, fxe.FX_NONE, 0)
    fxs.send_motor_command(kneID, fxe.FX_NONE, 0)
finally:        
    # Do anything but turn off the motors at the end of the program
    fxs.send_motor_command(ankID, fxe.FX_NONE, 0)
    fxs.send_motor_command(kneID, fxe.FX_NONE, 0)
    IMU.setToIdle()
    time.sleep(0.5) 
    fxs.close(ankID)
    fxs.close(kneID)  
    print('Communication with ActPacks closed and IMU set to idle')