""" Code to read sensors on board and print them in the console """
import sys, time, os
import numpy as np
from flexsea import flexsea as flex
from flexsea import fxEnums as fxe

sys.path.append(r'/home/pi/OSL/locoAPI/')       # Path to Loco module
sys.path.append(r'/usr/share/python3-mscl/')    # Path of the MSCL - API for the IMU
import locoOSL as loco                          # Module from Locolab
import mscl as msl                              # Module from Microstrain

from scipy.signal import butter, lfilter, lfilter_zi

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

# ------------------ MAIN LOOP -----------------------------------------------------------
try:
    """
    for i in range(1000):
        iniTime         = time.perf_counter_ns() 
        state           = fxs.read_device(kneID)
        IMUState        = IMU.getDataPackets(TIMEOUT)
        loadCellDict    = loco.state2LoCe(state)
        print( loadCellDict )
        deltaTime       = time.perf_counter_ns() - iniTime
    """
    # Create encoder map
    kneSta  = fxs.read_device(kneID)
    ankSta  = fxs.read_device(ankID)
    IMUPac  = IMU.getDataPackets(TIMEOUT)
    encMap  = loco.read_enc_map( loco.read_OSL(kneSta, ankSta, IMUPac) )

    # configure low-pass filter (1-order)
    nyq = 0.5 * 100
    normal_cutoff = 2 / nyq
    order = 1
    b_lp, a_lp = butter(1, normal_cutoff, btype='low', analog=False)
    z_lp = lfilter_zi(b_lp,  a_lp)
    # configure band-pass filter (2-order)
    nyq = 0.5 * 100
    normal_lowcut = 0.5 / nyq
    normal_highcut = 2 / nyq
    b_bp, a_bp = butter(2, [normal_lowcut, normal_highcut], btype='band', analog=False)
    z_bp = lfilter_zi(b_bp,  a_bp)

    
    t_p = time.time()
    loops = 0
    while True:
        # Read OSL
        kneSta  = fxs.read_device(kneID)
        ankSta  = fxs.read_device(ankID)
        IMUPac  = IMU.getDataPackets(TIMEOUT)
        dataOSL = loco.read_OSL(kneSta, ankSta, IMUPac, encoderMap = encMap)

        # Read measurement data
        GlobThighY = dataOSL['ThighSagi'][0] * 180 / np.pi
        #AnkleForceZ = dataOSL[][0]
        #AnkleForceX = dataOSL[][0]
        #AnkleMomentY =dataOSL[][0]
        
        # time
        t = time.time()
        dt = t - t_0 
        t_0 = t

        ## Compute global thigh angle velocity
        if loops == 0:
            GlobThighVel = 0 
        else:
            GlobThighVel = (GlobThigh - GlobThigh_0) / dt
            # low-pass filtering
            GlobThighVel_lp, z_lp = lfilter(b_lp, a_lp, GlobThighVel, zi = z_lp) # low-pass filtering
        loops += 1
        GlobThigh_0 = GlobThigh

        ## Compute atan2
        # band-pass filtering
        GlobThigh_bp, z_bp = lfilter(b_bp, a_bp, GlobThigh, zi = z_bp) 
        Atan2 = np.arctan2(-GlobThighVel_lp / (2*np.pi*0.8), GlobThigh_bp)
        if Atan2 < 0:
            Atan2 = Atan2 + 2 * np.pi
        
        AnkleY = dataOSL['ankJoiPos'][0] * 180 / np.pi
        KneeY = dataOSL['kneJoiPos'][0] * 180 / np.pi

        measurements = np.array([[GlobThighY], [GlobThighVelY], [Atan2]])


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