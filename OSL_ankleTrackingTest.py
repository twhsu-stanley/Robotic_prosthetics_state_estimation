"""
Code to test ankle position tracking
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
fxs.start_streaming(kneID, freq = 500, log_en = False)
fxs.start_streaming(ankID, freq = 500, log_en = False)
time.sleep(1)                           # Healthy pause before using ActPacks or IMU

# Soft start
G_K = {"kp": 40, "ki": 400, "K": 10, "B": 0, "FF": 128}  # Knee controller gains
G_A = {"kp": 40, "ki": 400, "K": 10, "B": 0, "FF": 128}  # Ankle controller gains
fxs.set_gains(ankID, G_A["kp"], G_A["ki"], 0, G_A["K"], G_A["B"], G_A["FF"])
fxs.set_gains(kneID, G_K["kp"], G_K["ki"], 0, G_K["K"], G_K["B"], G_K["FF"])
fxs.send_motor_command(ankID, fxe.FX_IMPEDANCE, fxs.read_device(ankID).mot_ang)
fxs.send_motor_command(kneID, fxe.FX_IMPEDANCE, fxs.read_device(kneID).mot_ang)
time.sleep(0.5)

## Set controller gains
# For gain details check https://dephy.com/wiki/flexsea/doku.php?id=controlgains
# 1) Small gians
#G_K = {"kp": 40, "ki": 400, "K": 60, "B": 0, "FF": 1}  # Knee controller gains
#G_A = {"kp": 40, "ki": 400, "K": 60, "B": 0, "FF": 1}  # Ankle controller gains

# 2) Medium gains
#G_K = {"kp": 40, "ki": 400, "K": 300, "B": 150, "FF": 60}  # Knee controller gains
#G_A = {"kp": 40, "ki": 400, "K": 300, "B": 150, "FF": 60}  # Ankle controller gains

# 3) Large  gains
G_K = {"kp": 40, "ki": 400, "K": 1000, "B": 3500, "FF": 1}  # Knee controller gains
G_A = {"kp": 40, "ki": 400, "K": 600, "B": 300, "FF": 128}  # Ankle controller gains

try:
    # Get encoder map
    kneSta  = fxs.read_device(kneID)
    ankSta  = fxs.read_device(ankID)
    IMUPac  = IMU.getDataPackets(TIMEOUT)
    encMap  = loco.read_enc_map(loco.read_OSL(kneSta, ankSta, IMUPac))

    # Initial data from the OSL
    kneSta  = fxs.read_device(kneID)
    ankSta  = fxs.read_device(ankID)
    IMUPac  = IMU.getDataPackets(TIMEOUT)

    dataOSL = loco.read_OSL(kneSta, ankSta, IMUPac)
    logger = loco.ini_log(dataOSL, sensors="all_sensors", trialName="ankleTrackingTest")
    print('Log Initialized')
    fxs.set_gains(ankID, G_A["kp"], G_A["ki"], 0, G_A["K"], G_A["B"], G_A["FF"])
    fxs.set_gains(kneID, G_K["kp"], G_K["ki"], 0, G_K["K"], G_K["B"], G_K["FF"])

    # Read inital OSL data
    #dataOSL = loco.read_OSL(kneSta, ankSta, IMUPac, logger['ini_time'], encMap)
    # Use buffers to avoid spikes in the initial measuremts
    kne_buffer = []
    ank_buffer = []
    for i in range(3):
        dataOSL = loco.read_OSL(kneSta, ankSta, IMUPac, logger['ini_time'], encMap)
        kne_buffer.append(dataOSL['kneJoiPos'] * 180 / np.pi)
        ank_buffer.append(dataOSL['ankJoiPos'] * 180 / np.pi)

    ## Commands settings
    # Knee angle limits (deg)
    knee_max = -5
    knee_min = -55
    # Knee initial position
    knee_fixed = np.median(kne_buffer) # dataOSL['kneJoiPos'] * 180 / np.pi # 

    # Ankle angle limits (deg)
    ankle_max = 18
    ankle_min = -10
    # Natural frequency of the sine wave (rad/s)
    freq = 2 * np.pi * 0.2 # 0.2 Hz
    # DC offset of the sine wave
    ankle_offset_initial = np.median(ank_buffer) # dataOSL['ankJoiPos'] * 180 / np.pi # 
    ankle_offset_final = (ankle_max + ankle_min) / 2
    # Amplitude of the sine wave
    ankle_amplitude_initial = 0
    ankle_amplitude_final = (ankle_max - ankle_min) / 2

    # Fade-in time (sec)
    fade_in_time = 3
    
    # Double check the values before starting the main loop
    print("Knee initial position: %.2f deg" % knee_fixed)
    print("Ankle initial position: %.2f deg" % ankle_offset_initial)

    if input('\n\nWould you like to continue? (y/n): ').lower() == 'y':
        print('\nReady to walk!')
    else:
        sys.exit("User stopped the execution")

    ## Main Loop #####################################################################################
    ankle_ref = []
    ankle_mea = []
    knee_ref = []
    knee_mea = []
    t_0 = time.perf_counter() # sec
    t = 0
    while(t < 10):
        t = time.perf_counter() - t_0

        # 1) Sinusoidal ankle command with fade-in effect (deg)
        if t < fade_in_time:
            amplitude = ankle_amplitude_initial + (ankle_amplitude_final - ankle_amplitude_initial) * t / fade_in_time
            ankle_offset = ankle_offset_initial + (ankle_offset_final - ankle_offset_initial) * t / fade_in_time
        elif t >= fade_in_time:
            amplitude = ankle_amplitude_final
            ankle_offset = ankle_offset_final

        ankle_cmd = amplitude * np.sin(freq * t) + ankle_offset
        
        # Saturation for ankle command
        if ankle_cmd > ankle_max: 
            ankle_cmd = ankle_max
        elif ankle_cmd < ankle_min:
            ankle_cmd = ankle_min
        
        # 2) Constant knee command (deg)
        knee_cmd = knee_fixed
        
        # Saturation for knee command
        if knee_cmd > knee_max: 
            knee_cmd = knee_max
        elif knee_cmd < knee_min:
            knee_cmd = knee_min
        
        # Sending commands  
        ankMotCou, kneMotCou = loco.joi2motTic(encMap, knee_cmd, ankle_cmd)
        fxs.send_motor_command(ankID, fxe.FX_IMPEDANCE, ankMotCou)
        fxs.send_motor_command(kneID, fxe.FX_IMPEDANCE, kneMotCou)
        
        # Read and log OSL data
        kneSta  = fxs.read_device(kneID)
        ankSta  = fxs.read_device(ankID)
        IMUPac  = IMU.getDataPackets(TIMEOUT)
        dataOSL = loco.read_OSL(kneSta, ankSta, IMUPac, logger['ini_time'], encMap)
        loco.log_OSL(dataOSL, logger)

        ankle_ref.append(ankle_cmd)
        ankle_mea.append(dataOSL['ankJoiPos'] * 180 / np.pi)
        knee_ref.append(knee_cmd)
        knee_mea.append(dataOSL['kneJoiPos'] * 180 / np.pi)
    
    #=======================================================================================================#

    # Convert list to numpy array
    #ankle_ref = np.array(ankle_ref).reshape(-1)
    #ankle_mea = np.array(ankle_mea).reshape(-1)
    #knee_ref = np.array(knee_ref).reshape(-1)
    #knee_mea = np.array(knee_mea).reshape(-1)

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

    # RMSE of joints positions
    #ankle_rmse = np.sqrt(np.square(ankle_ref - ankle_mea).mean())
    #knee_rmse = np.sqrt(np.square(knee_ref - knee_mea).mean())
    #print("RMSE of ankle position: ", ankle_rmse)
    #print("RMSE of knee position: ", knee_rmse)

    # Plot and save figures
    fig, ax = plt.subplots(2,1)
    ax[0].plot(ankle_ref, label = 'commanded')
    ax[0].plot(ankle_mea, label = 'measured')
    # ax[0].set_xlabel('Sample')
    ax[0].set_ylabel('Ankle angle (deg)')
    ax[0].set_ylim([-30,20])
    ax[0].legend()
    ax[0].set_title('Ankle-knee Motion')
    
    ax[1].plot(knee_ref, label = 'commanded')
    ax[1].plot(knee_mea, label = 'measured')
    ax[1].set_ylim([-70,10])
    ax[1].set_xlabel('Sample (10ms sample time)')
    ax[1].set_ylabel('Knee angle (deg)')
    ax[1].legend()
    plt.savefig('ankleTrackingTest.png')