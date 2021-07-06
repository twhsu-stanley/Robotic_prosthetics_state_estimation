'''
Module to interface with the locolab version of the OSL, LocoOSL. 
The locoOSL is the default OSL plus a thigh-IMU.
Python 3.7.3, rpi4, Dephy Actuator Package v0.2(B) (Firmware v5-20210121-ActPack-rigid0.2) 
Vamsi, Ross, and Edgar - March 2021
'''
import sys, time, os, csv,math
import numpy as np
sys.path.append(r'/usr/share/python3-mscl/')                # Path of the MSCL - API for the IMU
sys.path.append(r'/home/pi/OSL/Reference_Trajectories')     # Path to reference trajectories
import mscl as ms
import peakutils
from flexsea import flexsea as flex
from flexsea import fxEnums as fxe
from scipy import interpolate
from scipy import fftpack

# TODO
# Check loadcell scale. It is reading about 140N when Edgar is stading only in the OSL. (We should be able to measure something close to 700N)
# Fix 0 reading of joint encoder velocity. Follow up with Dephy's email. Check units of speed too.

# OPTIONAL
# Write routine to identify ports for ActPacks, IMU
# Modify ini_microstrain to allow programable IMUSampleTime
# Unify these functions and data into an object -CAUTION: compare realtime performance when making this transition.
# Minimum-jerk trajectory controller or trajectory generator

TIMEOUT = 500
# Motor Parameters
T_motor = {
    'jm'        : 0.12e-3,                   # Kg m^2 [LeePanRouse2019IROS]
    'bm'        : 0.16e-3,                   # Nm s / rad [LeePanRouse2019IROS]
    'k_tau'     : 0.146,                     # Nm/A [LeePanRouse2019IROS]
    'k_emf'     : 0.146,                     # Vs/rad [LeePanRouse2019IROS]     
    'r_phase'   : 3./2.*186e-3,              # Ohms [LeePanRouse2019IROS]
    'r'         : 49.4,                      # Reduction ratio knee, the reduction of the ankle is 58.4 ± 16.0
    'k_mot'     : 0.146/np.sqrt(3./2.*186e-3),
    }

OSL_units = {
    'Time': 's',
    'kneMotPos': 'rad',
    'kneMotTic': 'Ticks',
    'kneMotVel': 'rad/s',
    'kneMotAcc': 'rad/s^2',
    'kneMotCur': 'A',
    'kneBatVol': 'V',
    'kneBatCur': 'A',
    'kneGyrXax': 'rad/s',
    'kneGyrYax': 'rad/s',
    'kneGyrZax': 'rad/s',
    'kneAccXax': 'rad/s^2',
    'kneAccYax': 'rad/s^2',
    'kneAccZax': 'rad/s^2',
    'kneJoiPos': 'rad',
    'kneJoiVel': 'rad/s',
    'kneTimeSta':'s',
    'kneMotTor': 'Nm',
    'kneJoiTor': 'Nm',
    'ankMotPos': 'rad',
    'ankMotTic': 'Ticks',
    'ankMotVel': 'rad/s',
    'ankMotAcc': 'rad/s^2',
    'ankMotCur': 'A',
    'ankBatVol': 'V',
    'ankBatCur': 'A',
    'ankGyrXax': 'rad/s',
    'ankGyrYax': 'rad/s',
    'ankGyrZax': 'rad/s',
    'ankAccXax': 'rad/s^2',
    'ankAccYax': 'rad/s^2',
    'ankAccZax': 'rad/s^2',
    'ankJoiPos': 'rad',
    'ankJoiVel': 'rad/s',
    'ankTimeSta':'s',
    'ankMotTor': 'Nm',
    'ankJoiTor': 'Nm',
    'loadCelFx': 'N',
    'loadCelFy': 'N',
    'loadCelFz': 'N',
    'loadCelMx': 'Nm',
    'loadCelMy': 'Nm',
    'loadCelMz': 'Nm',
    'IMURolXax': 'rad',
    'IMUPitYax': 'rad',
    'IMUYawZax': 'rad',
    'IMUGyrXax': 'rad/s',
    'IMUGyrYax': 'rad/s',
    'IMUGyrZax': 'rad/s',
    'IMUAccXax': 'rad/s^2',
    'IMUAccYax': 'rad/s^2',
    'IMUAccZax': 'rad/s^2',
    'IMUTimSta': 's',
    'IMUDelTim': 's',
    'ThighSagi': 'rad',
    'ThighCoro': 'rad',
    'ThighTran': 'rad'
}

# UPDATED
def state2Dict(actPackState, prefix = '', iniJoint = 0, biasJoint = 0):
    """
    Convert state of ActPack to SI units and wrap it into a dictionary
    """       
    # Auxiliary conversions
    tick2rad = lambda x: x*(2*np.pi)/(2**14 - 1)    #14 bit encoder ticks to radians
    bits2rps = lambda x: x/32.8*(2*np.pi)/(360)     #Gyro bits to radians per second
    bit2mps2 = lambda x: x/8192*9.80665             #Accelerometer bits to m/s**2

    sensorOSL = dict([(prefix+'MotPos', tick2rad(actPackState.mot_ang))])
    sensorOSL[prefix+'MotTic'] = actPackState.mot_ang
    sensorOSL[prefix+'MotVel'] = tick2rad(actPackState.mot_vel)*1e3
    sensorOSL[prefix+'MotAcc'] = actPackState.mot_acc
    sensorOSL[prefix+'MotCur'] = actPackState.mot_cur*1e-3                        
    sensorOSL[prefix+'BatVol'] = actPackState.batt_volt*1e-3
    sensorOSL[prefix+'BatCur'] = actPackState.batt_curr*1e-3
    sensorOSL[prefix+'GyrXax'] = bits2rps(actPackState.gyrox)
    sensorOSL[prefix+'GyrYax'] = bits2rps(actPackState.gyroy)
    sensorOSL[prefix+'GyrZax'] = bits2rps(actPackState.gyroz)
    sensorOSL[prefix+'AccXax'] = bit2mps2(actPackState.accelx)
    sensorOSL[prefix+'AccYax'] = bit2mps2(actPackState.accely)
    sensorOSL[prefix+'AccZax'] = bit2mps2(actPackState.accelz)
    sensorOSL[prefix+'JoiPos'] = -(tick2rad(actPackState.ank_ang) + iniJoint) \
                                    + biasJoint 
    
    sensorOSL[prefix+'JoiVel'] = -actPackState.ank_vel
    sensorOSL[prefix+'TimSta'] = actPackState.SystemTime/1000
    # Estimated signals
    sensorOSL[prefix+'MotTor'] = sensorOSL[prefix+'MotCur']*T_motor['k_tau']
    sensorOSL[prefix+'JoiTor'] = sensorOSL[prefix+'MotTor']\
                                -sensorOSL[prefix+'MotAcc']*T_motor['jm']\
                                -sensorOSL[prefix+'MotVel']*T_motor['bm']
    return sensorOSL

def state2LoCe(actPackState):
    """
    Read genvars from ActPack and return loadcell measurements in the 6 axes as a dictionary with SI units
    """
    ampGain = 125
    EXC = 5
    offset = (2**12)/2
    rangeADC = 2**12 - 1    # Range from 0-4095 (12 bit ADC)
    rawLoadCell = np.array([actPackState.genvar_0, actPackState.genvar_1,
                            actPackState.genvar_2, actPackState.genvar_3, 
                            actPackState.genvar_4, actPackState.genvar_5])
    scaledLoadCell = (rawLoadCell- offset)/rangeADC*EXC
    scaledLoadCell = scaledLoadCell*1000/(EXC*ampGain)      #[mv]    
    # Decoupling matrix for LoadCell SN: 3675        
    DM = np.array([[3.59592, -400.67846, 3.96142, 8.88532, -4.79989, 396.76782],
        [7.89532, -241.50443, -2.55021, 468.02965, -7.84030, -228.10704],
        [-216.99768, -3.93163, -222.92478, -3.55461, -212.63961, 0.84577],
        [4.10269, -0.08257, -0.23597, 0.34608, -4.35220, -0.18607],
        [-2.57000, -0.48576, 5.08661, -0.00205, -2.48033, 0.42842],
        [0.04800, -5.56485, -0.08480, -5.00595, -0.03852, -5.69261]])      
    # # Decoupling matrix for LoadCell SN: 11257        
    # DM = np.array([[-32.67033, -1554.71169, 14.88415, 31.35056, 14.59218, 1525.95494],
    #     [3.77389, 923.35243, -16.81418, -1810.81131, 26.43635, 861.68852],
    #     [-901.66605, 15.77789, -917.98461, 5.76605, -887.18961, -11.67373],
    #     [17.64411, -0.25516, -0.20905, 0.53613, -18.17071, -0.41057],
    #     [-10.50416, -0.63003, 21.52759, -0.20044, -10.04536, 0.70095],
    #     [-0.29794, -22.48821, 0.05804, -22.73965, -0.71915, -23.13224]])       
    loadCellDecoupled = DM@scaledLoadCell 
    """
    Check get_bias_LoadCell for details - Loadcell offset when unloaded and resting on a table
    FX  - Mean: 5.3543      Standard dev.: 0.3970
    FY  - Mean: 28.4064     Standard dev.: 0.4568
    FZ  - Mean: -25.2324    Standard dev.: 0.4076
    MX  - Mean: 0.2068      Standard dev.: 0.0073
    MY  - Mean: 0.0118      Standard dev.: 0.0054
    MZ  - Mean: -0.3679     Standard dev.: 0.0071
    """
    """
    ############## Experiment in 210311 After updated wiring from Vamsi
    FX  - Mean: -1.7483     Standard dev.: 0.4668
    FY  - Mean: 26.5773     Standard dev.: 0.5045
    FZ  - Mean: -27.2637    Standard dev.: 0.3006
    MX  - Mean: 0.1900      Standard dev.: 0.0055
    MY  - Mean: -0.0101     Standard dev.: 0.0058
    MZ  - Mean: -0.5148     Standard dev.: 0.0074
    """
    meanVector = -np.array([-1.7483, 26.5773, -27.2637, 0.1900, -0.0101, -0.5148])
    """
    ############## Experiment in 210311 After updated wiring from Vamsi
    FX  - Mean: -11.6776    Standard dev.: 1.5474
    FY  - Mean: -115.6159   Standard dev.: 1.9460
    FZ  - Mean: -117.3587   Standard dev.: 1.2226
    MX  - Mean: 0.9551      Standard dev.: 0.0203
    MY  - Mean: -0.0365     Standard dev.: 0.0256
    MZ  - Mean: -2.0245     Standard dev.: 0.0274
    # Run 2
    FX  - Mean: -13.6664    Standard dev.: 1.8705
    FY  - Mean: -113.5456   Standard dev.: 2.1668
    FZ  - Mean: -116.6638   Standard dev.: 1.2058
    MX  - Mean: 0.9557      Standard dev.: 0.0211
    MY  - Mean: -0.0394     Standard dev.: 0.0242
    MZ  - Mean: -2.0320     Standard dev.: 0.0318
    """
    sensorOSL = {'loadCelFx': int(loadCellDecoupled[0]+meanVector[0])}
    sensorOSL['loadCelFy'] = int(loadCellDecoupled[1]+meanVector[1])
    sensorOSL['loadCelFz'] = int(loadCellDecoupled[2]+meanVector[2])
    sensorOSL['loadCelMx'] = loadCellDecoupled[3]+meanVector[3]
    sensorOSL['loadCelMy'] = loadCellDecoupled[4]+meanVector[4]
    sensorOSL['loadCelMz'] = loadCellDecoupled[5]+meanVector[5]
    return sensorOSL

def IMUPa2Dict(IMU_packets):
    """
    Convert IMU packets into dictionary with SI units
    """     
    if(len(IMU_packets)):
        # Read all the information from the first packet as float.
        microstrainData = {dataPoint.channelName(): dataPoint.as_float() for dataPoint in IMU_packets[0].data()}
    else:
        raise Exception('The IMU did not send messages within the timeout.')

    d2r = lambda x: x*np.pi/180                     #degrees to radians
    #Parsing information from IMU
    #print(microstrainData['estRoll'])
    OSLSensors = {
    'IMURolXax': microstrainData['estRoll'],
    'IMUPitYax': microstrainData['estPitch'],
    'IMUYawZax': microstrainData['estYaw'],
    'IMUGyrXax': microstrainData['estAngularRateX'],
    'IMUGyrYax': microstrainData['estAngularRateY'],
    'IMUGyrZax': microstrainData['estAngularRateZ'],
    'IMUAccXax': microstrainData['estLinearAccelX'],
    'IMUAccYax': microstrainData['estLinearAccelY'],
    'IMUAccZax': microstrainData['estLinearAccelZ'],
    'IMUTimSta': microstrainData['estFilterGpsTimeTow'],
    'IMUDelTim': microstrainData['estFilterGpsTimeTow'],
    'ThighSagi': microstrainData['estRoll'] + d2r( 39.38 ),    # Adding offset rotation from IMU mount
    'ThighCoro': microstrainData['estPitch'],
    'ThighTran': microstrainData['estYaw'],
    }
    return OSLSensors

def read_OSL(kneeState, ankleState, IMUPackets, initime = 0, encoderMap = None):
    """
    Parse actpacks and IMU to same-old same-old dataOSL dictionary
    """
    d2r = lambda x: x*np.pi/180     # Auxiliary function
    if encoderMap == None:
        iniAnk, biasAnk = 0, 0
        iniKne, biasKne = 0, 0
    else:
        iniAnk, biasAnk = encoderMap['ankJoiIni'], encoderMap['ankJoiBia']
        iniKne, biasKne = encoderMap['kneJoiIni'], encoderMap['kneJoiBia']
    timeDic = {'Time': time.perf_counter()-initime}
    kneDic = state2Dict(kneeState, 'kne', iniKne, biasKne)
    ankDic = state2Dict(ankleState,'ank', iniAnk, biasAnk)
    # print("IniJoint: {:>10.4f} [rad]|BiasJoint: {:>10.4f} [rad]|".format(iniAnk,biasAnk)) 
    loaDic = state2LoCe(kneeState)        # State of the ActPack connected to the loadcell    
    IMUDic = IMUPa2Dict(IMUPackets)
    return {**timeDic, **kneDic, **loaDic, **ankDic, **IMUDic}

def read_enc_map( OSLdic):
    """
    Read encoder map from a .csv file. The 'home' function generates and stores this .csv
    file in the SD card. Then, we read the absolute joint encoder to know the offset in 
    the relative motor encoder and create the joint to motor encoder map.
    """
    # Auxiliary functions
    d2r      = lambda x: x*np.pi/180     
    tick2rad = lambda x: x*(2*np.pi)/(2**14 - 1)    #14 bit encoder ticks to radians
    
    # Read homing data from the csv file
    try:
        with open('/home/pi/OSL/locoAPI/joint2motorAnkle.csv', 'r') as f:
            list_file = list(csv.reader(f))
            ankMotPosMap_plan2dor = np.array([float(item[0]) for item in list_file if not (item == list_file[0])])
            ankJoiPosMap_plan2dor = np.array([float(item[1]) for item in list_file if not (item == list_file[0])])
        with open('/home/pi/OSL/locoAPI/joint2motorKnee.csv', 'r') as f:
            list_file = list(csv.reader(f))
            kneMotPosMap_flex2ext = np.array([float(item[0]) for item in list_file if not (item == list_file[0])])
            kneJoiPosMap_flex2ext = np.array([float(item[1]) for item in list_file if not (item == list_file[0])])
    except FileNotFoundError:
        raise Exception('The csv files are not available in the folder /home/pi/OSL/locoAPI/. Run the homing routine after a power cycle.')

    # Select only the values of the vector that are increasing. This is mandatory for the interpolation
    indxing = lambda x: (np.diff(x, append = x[-1]) > 0 )
    
    # Indexes for values that are only monotonically increasing 
    ankindx = indxing( -ankJoiPosMap_plan2dor )
    kneindx = indxing( kneJoiPosMap_flex2ext )

    # Enforcing that the map is monotonically increasing to use np.diff
    ankMotPosMap_plan2dor = ankMotPosMap_plan2dor[ ankindx ]
    ankJoiPosMap_plan2dor = ankJoiPosMap_plan2dor[ ankindx ]
    kneMotPosMap_flex2ext = kneMotPosMap_flex2ext[ kneindx ]
    kneJoiPosMap_flex2ext = kneJoiPosMap_flex2ext[ kneindx ]

    # Get the offset of the motor encoders - Read values in radians
    ankJoiEnc = OSLdic['ankJoiPos']      # Joint encode in radians
    ankMotEnc = OSLdic['ankMotTic']      # Motor encoder in ticks
    kneJoiEnc = OSLdic['kneJoiPos']
    kneMotEnc = OSLdic['kneMotTic']

    # Use old encoder map to obtain old motor position - Ankle negative b/c increasing
    motAnkOldMap = -np.interp(-ankJoiEnc, -ankJoiPosMap_plan2dor, -ankMotPosMap_plan2dor) 
    motKneOldMap =  np.interp( kneJoiEnc,  kneJoiPosMap_flex2ext,  kneMotPosMap_flex2ext) 
    offsetAnk   = ankMotEnc - int(motAnkOldMap)
    offsetKne   = kneMotEnc - int(motKneOldMap)

    # Make the old encoder map valid for the current motor encoder setup
    ankMotPosMap_plan2dor = ankMotPosMap_plan2dor + offsetAnk
    kneMotPosMap_flex2ext = kneMotPosMap_flex2ext + offsetKne
    
    # Vector from joint to motor encoders - Setting the initial biases
    encMap = {  'ankMot': ankMotPosMap_plan2dor,
                #Ankle is 19.65 deg. at max. plantarflexion
                'ankJoiIni': ankJoiPosMap_plan2dor[0],
                'ankJoiBia': d2r(19.65),
                'ankJoi': ankJoiPosMap_plan2dor - ankJoiPosMap_plan2dor[0] + d2r(19.65), 
                'kneMot': kneMotPosMap_flex2ext,   
                #Knee is zero at max. extension           
                'kneJoi': kneJoiPosMap_flex2ext - kneJoiPosMap_flex2ext[-1],
                'kneJoiIni': kneJoiPosMap_flex2ext[-1],
                'kneJoiBia': d2r(0),}             

    return encMap

def joi2motTic(encoderMap, kneeAngle, anklAngle, anglesInDegrees = True):    
    """
    Returns absolute motor position in encoder ticks for a given joint position (in degrees or radians).
    - The knee angle is 0 deg at max. extension and -120 deg at max. flexion. ROM is 120 degress.
    - The ankle angle is +19.65deg at max. plantar flexion, 0 with foot perpendicular to the shank, and -10.35 deg
    at max. dorsiflexion. ROM of the ankle is about 30 degrees.
    """
    d2r = lambda x: x*np.pi/180
    r2d = lambda x: x*180/np.pi
    if anglesInDegrees:        
        anklAngle = d2r(anklAngle)
        kneeAngle = d2r(kneeAngle)
    # Is the angle within the adequate ROM? - Software safety feature
    if d2r(-116) >= kneeAngle:
        raise ValueError('Desired knee angle: {:.2f} [deg] is too low (less than -116 [deg]).'.format(r2d(kneeAngle))) 
    if d2r(0) < kneeAngle:
        raise ValueError('Desired knee angle: {:.2f} [deg] is too high (greater than 0 [deg]).'.format(r2d(kneeAngle)))
    if d2r(-10.35) >= anklAngle:
        raise ValueError('Desired ankle angle: {:.2f} [deg] is too low (less than -10.35 [deg]).'.format(r2d(anklAngle))) 
    if d2r(19.65) < anklAngle:
        raise ValueError('Desired ankle angle: {:.2f} [deg] is too high (greater than 19.65 [deg]).'.format(r2d(anklAngle))) 
    # if not ( ( d2r(-116) < kneeAngle <= d2r(0) ) and ( d2r(-10.35) < anklAngle <= d2r(19.65) ) ):
    #     raise ValueError('The desired knee: {:.2f} [deg] or the ankle: {:.2f}[deg] are out of the ROM. Please check the angle convention.'.format(r2d(kneeAngle), r2d(anklAngle) ) )
    
    # Angular position of motor in ticks
    # Negative because the x-coordinate sequence is expected to be increasing

    
    # ankMotTick = -np.interp(-anklAngle, -encoderMap['ankJoi'], -encoderMap['ankMot']) 
    # kneMotTick = np.interp(kneeAngle, encoderMap['kneJoi'], encoderMap['kneMot'])

    ankInterp = interpolate.interp1d(-encoderMap['ankJoi'], -encoderMap['ankMot'],kind = 'linear')
    kneInterp = interpolate.interp1d(encoderMap['kneJoi'], encoderMap['kneMot'],kind = 'linear')

    ankMotTick = -ankInterp(-anklAngle)
    kneMotTick = kneInterp(kneeAngle)

    # Angular position of the motor in encoder ticks
    ankMotCou = int( ankMotTick )
    kneMotCou = int( kneMotTick )   
    return ankMotCou, kneMotCou

def loadTrajectory(trajectory = 'walking'):
    """
    Creates a dictionary with the ankle-knee trajectories from recorded csv files
    """
    # Create path to the reference csv trajectory
    if trajectory.lower() == 'walking':
        # walking data uses convention from D. A. Winter, “Biomechanical Motor Patterns in Normal Walking,”  
        # J. Mot. Behav., vol. 15, no. 4, pp. 302–330, Dec. 1983.
        pathFile = r'/home/pi/OSL/Reference_Trajectories/walkingWinter_deg.csv'
        # Gains to scale angles to OSL convention
        ankGain = -1
        ankOffs = -0.15 # [deg] Small offset to take into accoun that ankle ROM is -10 deg< ankle < 19.65 deg
        kneGain = -1
        kneOffs = 0
        hipGain = 1
        hipOffs = 0

    elif trajectory.lower() == 'stair_ascent':
        pathFile = r'/home/pi/OSL/Reference_Trajectories/stairAscentReznick_deg.csv'
        # Gains to scale angles to OSL convention
        ankGain = -1
        ankOffs = -0.15 # [deg] Small offset to take into accoun that ankle ROM is -10 deg< ankle < 19.65 deg
        kneGain = -1
        kneOffs = 0
        hipGain = 1
        hipOffs = 0
    elif trajectory.lower() == 'stair_descent':
        pathFile = r'/home/pi/OSL/Reference_Trajectories/stairAscentReznick_deg.csv'
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

    if trajectory.lower() == 'walking':
        
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
    elif trajectory.lower() == 'stair_ascent':
        
        # Parse data to knee-ankle trajectories using OSL angle convention (+ ankle = plantarflexion. + knee = flexion)
        
        #Pose
        trajectory = dict(ank20 = ankGain*data[0] + ankOffs)
        trajectory["ank25"] = ankGain*data[1]
        trajectory["ank30"] = ankGain*data[2]
        trajectory["ank35"] = ankGain*data[3]

        trajectory["kne20"] = kneGain*data[4] + kneOffs
        trajectory["kne25"] = kneGain*data[5] + kneOffs
        trajectory["kne30"] = kneGain*data[6] + kneOffs
        trajectory["kne35"] = kneGain*data[7] + kneOffs

        trajectory["hip20"] = hipGain*data[8] + hipOffs
        trajectory["hip25"] = hipGain*data[9] + hipOffs
        trajectory["hip30"] = hipGain*data[10] + hipOffs
        trajectory["hip35"] = hipGain*data[11] + hipOffs

        #Velocity
        trajectory["ankd20"] = ankGain*data[12]
        trajectory["ankd25"] = ankGain*data[13]
        trajectory["ankd30"] = ankGain*data[14]
        trajectory["ankd35"] = ankGain*data[15]

        trajectory["kned20"] = kneGain*data[16]
        trajectory["kned25"] = kneGain*data[17]
        trajectory["kned30"] = kneGain*data[18]
        trajectory["kned35"] = kneGain*data[19]

        trajectory["hipd20"] = hipGain*data[20]
        trajectory["hipd25"] = hipGain*data[21]
        trajectory["hipd30"] = hipGain*data[22]
        trajectory["hipd35"] = hipGain*data[23]

        #Acceleration
        trajectory["ankdd20"] = ankGain*data[24]
        trajectory["ankdd25"] = ankGain*data[25]
        trajectory["ankdd30"] = ankGain*data[26]
        trajectory["ankdd35"] = ankGain*data[27]

        trajectory["knedd20"] = kneGain*data[28]
        trajectory["knedd25"] = kneGain*data[29]
        trajectory["knedd30"] = kneGain*data[30]
        trajectory["knedd35"] = kneGain*data[31]

        trajectory["hipdd20"] = hipGain*data[32]
        trajectory["hipdd25"] = hipGain*data[33]
        trajectory["hipdd30"] = hipGain*data[34]
        trajectory["hipdd35"] = hipGain*data[35]

        trajectory["phase20"] = data[36]
        trajectory["phase25"] = data[37]
        trajectory["phase30"] = data[38]
        trajectory["phase35"] = data[39]

        trajectory["time"] = data[40]
        
    elif trajectory.lower() == 'stair_descent':
            
            # Parse data to knee-ankle trajectories using OSL angle convention (+ ankle = plantarflexion. + knee = flexion)
            
            #Pose
            trajectory = dict(ank20 = ankGain*data[0] + ankOffs)
            trajectory["ank25"] = ankGain*data[1]
            trajectory["ank30"] = ankGain*data[2]
            trajectory["ank35"] = ankGain*data[3]

            trajectory["kne20"] = kneGain*data[4] + kneOffs
            trajectory["kne25"] = kneGain*data[5] + kneOffs
            trajectory["kne30"] = kneGain*data[6] + kneOffs
            trajectory["kne35"] = kneGain*data[7] + kneOffs

            trajectory["hip20"] = hipGain*data[8] + hipOffs
            trajectory["hip25"] = hipGain*data[9] + hipOffs
            trajectory["hip30"] = hipGain*data[10] + hipOffs
            trajectory["hip35"] = hipGain*data[11] + hipOffs

            #Velocity
            trajectory["ankd20"] = ankGain*data[12]
            trajectory["ankd25"] = ankGain*data[13]
            trajectory["ankd30"] = ankGain*data[14]
            trajectory["ankd35"] = ankGain*data[15]

            trajectory["kned20"] = kneGain*data[16]
            trajectory["kned25"] = kneGain*data[17]
            trajectory["kned30"] = kneGain*data[18]
            trajectory["kned35"] = kneGain*data[19]

            trajectory["hipd20"] = hipGain*data[20]
            trajectory["hipd25"] = hipGain*data[21]
            trajectory["hipd30"] = hipGain*data[22]
            trajectory["hipd35"] = hipGain*data[23]

            #Acceleration
            trajectory["ankdd20"] = ankGain*data[24]
            trajectory["ankdd25"] = ankGain*data[25]
            trajectory["ankdd30"] = ankGain*data[26]
            trajectory["ankdd35"] = ankGain*data[27]

            trajectory["knedd20"] = kneGain*data[28]
            trajectory["knedd25"] = kneGain*data[29]
            trajectory["knedd30"] = kneGain*data[30]
            trajectory["knedd35"] = kneGain*data[31]

            trajectory["hipdd20"] = hipGain*data[32]
            trajectory["hipdd25"] = hipGain*data[33]
            trajectory["hipdd30"] = hipGain*data[34]
            trajectory["hipdd35"] = hipGain*data[35]

            trajectory["phase20"] = data[36]
            trajectory["phase25"] = data[37]
            trajectory["phase30"] = data[38]
            trajectory["phase35"] = data[39]

            trajectory["time"] = data[40]
    return trajectory

def getVirtualConstraints(refKnee,refAnkle, pv, N):
    L = len(refKnee)

    #Calculate DFT
    X_k = fftpack.fft(refKnee)/L
    X_a = fftpack.fft(refAnkle)/L

    #Real One sided DFT Coefficients
    pk_real = np.real(X_k[range(L/2)])
    pa_real = np.real(X_a[range(L/2)])

    #Imaginary One sided DFT Coefficients
    pk_imag = np.imag(X_k[range(L/2)])
    pa_imag = np.imag(X_a[range(L/2)])

    # Calculating VC
    knee_traj =  .5*pk_real(0) + .5*pk_real(N/2)*math.cos(math.pi*N*pv);   
    ankle_traj =  .5*pa_real(0) + .5*pa_real(N/2)*math.cos(math.pi*N*pv);   
    for i in int(range(N)/2):
        knee_traj += pk_real(i+1)*math.cos(2*math.pi*(i)*pv)-pk_imag(i+1)*math.sin(2*math.pi*(i)*pv)
        ankle_traj += pa_real(i+1)*math.cos(2*math.pi*(i)*pv)-pa_imag(i+1)*math.sin(2*math.pi*(i)*pv)


    return knee_traj,ankle_traj


def getPhaseVariable_vTwoStates(OSLdic,maxHip, minHip, load_z, FCThr = 10):
    """
    Compute the walking phase variable using the thigh angle during only two monotonic regions: stance and swing.
    Note: With this phase variable the subject may need to modify the gait pattern to obtain better results. 
    Talk to Edgar about human compass walking.
    To do: Implement [1] D. J. Villarreal, D. Quintero, and R. D. Gregg, “Piecewise and unified phase variables in 
    the control of a powered prosthetic leg,” in 2017 International Conference on Rehabilitation Robotics (ICORR),
     2017, pp. 1425–1430. This controller has the same principles behind the current implementation.
    """
    r2d = lambda x: x*180/np.pi

    # Load walking trajectory and min. max. hip values for interpolation
    # wlkTrj = loadTrajectory(trajectory = 'walking')
    # maxHip = 30
    # minHip = -10
    # maxHip = np.amax(wlkTrj['hip_'])        # Close to hip at HS
    # minHip = np.amin(wlkTrj['hip_'])        # Max. hip extension

    # Read data from sensors
    thigh = r2d( OSLdic['ThighSagi'] )     # Sagittal-plane thigh angle [degrees]

    s = 0.57

    # Interpolate phase variable depending on loading condition (add + 0.5 to phase variable during swing)
    if load_z >= FCThr:
        pv = ((thigh - minHip) / (maxHip - minHip))*(1-s) + s
        if pv > 1:
            pv = 1
        elif pv < s:
            pv = s
    else:        
        pv =  ( (thigh - maxHip) / (minHip - maxHip)  )*s
        if pv > s:
            pv = s
        elif pv < 0:
            pv = 0
    
    return pv

def getPhaseVariable_vSiavash(dataOSL, prevState, prevPV, sm, qhm, maxHip,minHip,HS_Hip,load_z,c = 0.53, qPo = -8.4, FCThres = 0):
    """
    Compute the walking phase variable using the thigh angle during only two monotonic regions: stance and swing.
    Note: The version of this phase variable corresponds to the definition in Section II-B of 
    [RQDRGG_IEEEAccess2019] S. Rezazadeh, D. Quintero, N. Divekar, E. Reznick, L. Gray, and R. D. Gregg, “A Phase 
    Variable Approach for Improved Rhythmic and Non-Rhythmic Control of a Powered Knee-Ankle Prosthesis,” 
    vol. x, pp. 1–16, 2019.
    """
    r2d = lambda x: x*180/np.pi

    # Load walking trajectory and min. max. hip values for interpolation
    # wlkTrj = loadTrajectory(trajectory = 'walking')
    # maxHip = np.amax(wlkTrj['hip_'])
    # minHip = np.amin(wlkTrj['hip_'])
    # HS_Hip = wlkTrj['hip_'][0]

    # Read thigh angle in Sagittal plane [degrees]
    thigh  = r2d( dataOSL['ThighSagi'])    
    thighd = r2d( dataOSL['IMUGyrXax'])
    # Define if Foot contact is True or False
    if (load_z > FCThres):
        FC = False
    else:
        FC = True

    # Transition conditions. Check Fig. 2 of RQDRGG_IEEEAccess2019 to evaluate conditions.
    if (prevState == 1 and (thigh < qPo and FC) ):
        currState = 2        
    elif (prevState == 2 and thighd > 0):
        currState = 3
        sm = prevPV
        qhm = thigh
    elif (prevState == 4 and FC):
        currState = 1
    elif (not FC):      # Regardless of the state, if there is no foot contact go to state 4.
        currState = 4
    else:               # There is not transition
        currState = prevState
    #print(currState)
    # Compute phase variable
    if ( currState == 1 or currState == 2 ):
        currPV = (HS_Hip - thigh) / (HS_Hip - minHip)*c
    elif ( currState == 3 or currState == 4 ):
        currPV = 1 + (1 - sm)/(HS_Hip - qhm)*(thigh - HS_Hip)

    # Compute unidirectional filter for state 3
    if ( currState == 3 and (prevPV > currPV) ):
        currPV = prevPV         # Enforce s(k-1) <= s(k)

    # Saturate phase variable as a safety feature
    if currPV > 1:
        currPV = 1
    elif currPV < 0:
        currPV = 0

    return currPV, prevState, sm, qhm, FC

def getPhaseVariable_Stairs(dataOSL, prevState, prevPV, sm, qhm, maxHip, minHip, mhf, load_z, c = 0.61, qPo = 10.0, FCThres = 24):
    
    r2d = lambda x: x*180/np.pi

    # Load walking trajectory and min. max. hip values for interpolation
    # stairTrj = loadTrajectory(trajectory = 'stair_ascent')
    # maxHip = np.amax(stairTrj['hip_'])
    # minHip = np.amin(stairTrj['hip_'])
    

    # Read thigh angle in Sagittal plane [degrees]
    thigh  = r2d( dataOSL['ThighSagi'])    
    thighd = r2d( dataOSL['IMUGyrXax'])
    # Define if Foot contact is True or False
    if (load_z > FCThres):
        FC = False
    else:
        FC = True

    # Transition conditions. Check Fig. 2 of RQDRGG_IEEEAccess2019 to evaluate conditions.
    if (prevState == 1 and (thigh < qPo and FC) ):
        currState = 2        
    elif (prevState == 2 and thighd > 0):
        currState = 3
        sm = prevPV
        qhm = thigh
    elif (prevState == 4 and (mhf or FC)):
        currState = 1
        mhf = False
    elif (not FC):      # Regardless of the state, if there is no foot contact go to state 4.
        currState = 4
    else:               # There is not transition
        currState = prevState
    
    # Compute phase variable
    if ( currState == 1 or currState == 2 ):
        currPV = (maxHip - thigh) / (maxHip - minHip)*c
    elif ( currState == 3 or currState == 4 ):
        currPV = 1 + (1 - sm)/(maxHip - qhm)*(thigh - maxHip)

    # Compute unidirectional filter for state 3
    if ( currState == 3 and (prevPV > currPV) ):
        currPV = prevPV         # Enforce s(k-1) <= s(k)

    # Saturate phase variable as a safety feature
    if currPV > 1:
        currPV = 1
    elif currPV < 0:
        currPV = 0

    return currPV, prevState, sm, qhm, mhf

def maximumHipFlexionDetection(dataOSL,mhf, maxHip, temp_traj, minHipThresh = .3, maxHipThresh = 1.3, hipSwingThresh = .16, peakThresh = .8,minPeakDist = 100, FCThres = -50):
    thigh = dataOSL['ThighSagi'][0]

    if (dataOSL['loadCelFz'][0] < FCThres):
        if thigh > hipSwingThresh:
            if(not mhf or temp_traj != []):
                temp_traj.append(thigh)

            #Start Peak Search Algorithm
            if len(temp_traj) > 0:
                indexes = peakutils.indexes(np.array(temp_traj), thres = peakThresh,min_dist = minPeakDist)
                #print(indexes)
                if len(indexes) > 0:
                    if indexes[-1] == (len(temp_traj)-2):
                        if temp_traj[len(temp_traj)-2] > minHipThresh and temp_traj[len(temp_traj)-2] < maxHipThresh:
                            mhf = True
                            maxHip = temp_traj[len(temp_traj)-2]

    return mhf, maxHip, temp_traj
    
def home_joint(fxs, actPackID, IMU, joint, jointVolt = 1000, motTorThr = 0.35):
    """ 
    Move joint until sensing a toque/current limit. Record data in a csv.  
    """
    def moveUntilTorqueLimit(motVol):
        """
        Move using open voltage control until hitting torque threshold (neg. -> pos.). 
        """
        time.sleep(0.1)         # Healthy pause before starting
        keepGoing = True
        i = 0

        # -----------------------------Set the joint in a know position to start recording
        while keepGoing:                               
            #Move in one direction
            fxs.send_motor_command(actPackID, fxe.FX_VOLTAGE, -motVol) 
            # Read IMU to control sample time
            joiDict = state2Dict( fxs.read_device(actPackID) )
            IMUDict = IMUPa2Dict( IMU.getDataPackets(TIMEOUT) )      
            print('Calculated motor torque: %3.3f'%joiDict['MotTor'])  

            if ( np.abs( joiDict['MotTor'] ) >= motTorThr ):
                keepGoing = False
                fxs.send_motor_command(actPackID, fxe.FX_VOLTAGE, 0)
                IMU.getDataPackets(TIMEOUT)     # Wait one sample so that torque goes to 0
            elif (i > 6000):
                fxs.send_motor_command(actPackID, fxe.FX_VOLTAGE, 0)
                raise Exception(f'Stopping homing routine after {i} samples')
            i += 1           

        # --------Move until finding the torque limit and collect joint and motor position
        # Initialize variables for the loop
        deltaTime = 0
        keepGoing = True
        i = 0
        torqueArray = np.array([])
        motPosArray = np.array([])
        joiPosArray = np.array([])

        # Read initial state
        joiDict = state2Dict( fxs.read_device(actPackID) ) 
        IMUDict = IMUPa2Dict( IMU.getDataPackets(TIMEOUT) )

        torqueArray = np.append( torqueArray, joiDict['MotTor'])
        motPosArray = np.append( motPosArray, joiDict['MotTic'])
        joiPosArray = np.append( joiPosArray, joiDict['JoiPos'])

        while keepGoing:
            #Initial time to estimate the sample time per cycle            
            iniTime = time.perf_counter_ns()                    
            #Move in one direction
            fxs.send_motor_command(actPackID, fxe.FX_VOLTAGE, motVol)
            # Read and append data
            joiDict = state2Dict( fxs.read_device(actPackID) )
            IMUDict = IMUPa2Dict( IMU.getDataPackets(TIMEOUT) )
            torqueArray = np.append( torqueArray, joiDict['MotTor'])
            motPosArray = np.append( motPosArray, joiDict['MotTic'])
            joiPosArray = np.append( joiPosArray, joiDict['JoiPos'])           
            
            print('Calculated motor torque: %3.3f'%joiDict['MotTor'])

            if ( np.abs( joiDict['MotTor'] ) >= motTorThr ):
                keepGoing = False
                fxs.send_motor_command(actPackID, fxe.FX_VOLTAGE, 0)
                # Read and append data
                joiDict = state2Dict( fxs.read_device(actPackID) )
                IMUDict = IMUPa2Dict( IMU.getDataPackets(TIMEOUT) )
                torqueArray = np.append( torqueArray, joiDict['MotTor'])
                motPosArray = np.append( motPosArray, joiDict['MotTic'])
                joiPosArray = np.append( joiPosArray, joiDict['JoiPos'])   
            elif (i > 2000):
                raise Exception(f'Stopping homing routine after {i} samples')
            print('Delta of time in the cycle: %4.4f [ms]'%deltaTime) 
            i += 1
            deltaTime = (time.perf_counter_ns() - iniTime)/1e6
        time.sleep(0.1)     # Healthy pause before finishing
        return motPosArray, joiPosArray
    try:
        fxs.send_motor_command(actPackID, fxe.FX_NONE, 0)
        # Execute homing task               
        motPosArray, joiPosArray = moveUntilTorqueLimit( jointVolt ) 
        if joint == 'ankle':
            strFile = 'joint2motorAnkle.csv'
        elif joint == 'knee':
            strFile = 'joint2motorKnee.csv'
        else:
            raise Exception('Provide valid string for "joint" in homing routine')

        with open('/home/pi/OSL/locoAPI/'+strFile, 'w') as f:
            f.write("motor_encoder_[tick], joint_encoder_[rad]\n")
            for Mo, Jo in zip(motPosArray, joiPosArray): 
                f.write(f"{Mo}, {Jo}\n") 
    except KeyboardInterrupt:
        print('\n***Homing routine stopped by user***\n')
# OUTDATED

def log_OSL(dataOSL, logger,degrees=True):
    """
    Log the OSL sensor information and its corresponding units.
    Parameters
    ----------
    dataOSL : dict
        Output dictionary of the function read_OSL 
    logger : dict
        
    degrees : bool
        If True prints the joint and motor angular position, velocity, and acceleration in degrees instead of rads
    """
    sensors = logger['sensors']


    # Create a list with the appropiate keys to print always includes internal time
    if sensors == 'all_sensors':
        keys = list(dataOSL.keys())
    elif sensors == 'ankleActPack':
        keys = [key for key in list(dataOSL.keys()) if (key.lower().startswith('tim') or key.lower().startswith('ank'))] 
    elif sensors == 'kneeActPack':
        keys = [key for key in list(dataOSL.keys()) if (key.lower().startswith('tim') or key.lower().startswith('kne'))]  
    elif sensors == 'IMU':
        keys = [key for key in list(dataOSL.keys()) if (key.lower().startswith('tim') or key.lower().startswith('imu'))]
    elif sensors == 'loadCell':
        keys = [key for key in list(dataOSL.keys()) if (key.lower().startswith('tim') or key.lower().startswith('loa'))]        
    else:
        raise Exception('Non-valid set of keys. Please check "if" within log_OSL')
    
    
    val_list = []
    
    #Makes a list of all data that will be logged based on the keys used
    for k in keys:
        if degrees and ( k.endswith('MotPos') or k.endswith('MotVel') or k.endswith('MotAcc') or \
                         k.endswith('JoiPos') or k.endswith('JoiVel') ):

            val_list.append(dataOSL[k]*180/np.pi)
             
        else:
            val_list.append(dataOSL[k])
            

    #Logs data into .csv file
    with open(logger['log'], 'a') as csvfile:
        writer = csv.writer(csvfile)
        # writing data rows
        writer.writerow(val_list) 

def print_OSL(dataOSL, sensors = 'all_sensors', clearConsole = False, data2print = None, degrees = True):
    """
    Print the OSL sensor information and its corresponding units.
    Parameters
    ----------
    dataOSL : dict
        Output dictionary of the function read_OSL   
    sensors : str
        String to define the set of sensors to print, e.g., 'all_sensors', 'ankleActPack'
    clearConsole : bool
        If True it clears the console screen before printing
    data2print : list
        List with all the keys of the dataOSL dictionary that are desired to print
    degrees : bool
        If True prints the joint and motor angular position, velocity, and acceleration in degrees instead of rads
    """
    if clearConsole:
        # This works in raspbian OS/Debian. Windows is not supported to improve real-time.
        os.system('clear')

    # Create a list with the appropiate keys to print
    if sensors == 'all_sensors':
        keys = list(dataOSL.keys())
    elif sensors == 'ankleActPack':
        keys = [key for key in list(dataOSL.keys()) if key.lower().startswith('ank')] 
    elif sensors == 'kneeActPack':
        keys = [key for key in list(dataOSL.keys()) if key.lower().startswith('kne')]  
    elif sensors == 'IMU':
        keys = [key for key in list(dataOSL.keys()) if key.lower().startswith('imu')]
    elif sensors == 'loadCell':
        keys = [key for key in list(dataOSL.keys()) if key.lower().startswith('loa')]        
    else:
        raise Exception('Non-valid set of keys. Please check "if" within print_OSL')
    if data2print is not None:
        keys = data2print
    # Ship it!    
    for k in keys:
        if degrees and ( k.endswith('MotPos') or k.endswith('MotVel') or k.endswith('MotAcc') or \
                         k.endswith('JoiPos') or k.endswith('JoiVel') ):
            print(k, ': ', "{: >15.4f}".format(dataOSL[k][0]*180/np.pi), "[degrees%s]"%(dataOSL[k][1][3:]) ) 
        else:
            print(k, ': ', "{: >15.4f}".format(dataOSL[k][0]), "[%s]"%dataOSL[k][1])       

def ini_log(dataOSL, sensors = "all_sensors",relativePath = "", trialName = ""):
    """
    Initialize log file
    """    
    #Create logfile
    logFile = relativePath + time.strftime("%y%m%d_%H%M%S")

    #Check if log file has a duplicate and append number
    i = 2
    temp = logFile+"_"+trialName
    while True:
        if os.path.exists(logFile+".csv"):
            temp = logFile+"_"+ str(i)
            i = i+1
        else:
            logFile = temp+".csv"
            break

    #Get initial time for log
    initial_time = time.perf_counter()

    if sensors == 'all_sensors':
        keys = list(dataOSL.keys())
    elif sensors == 'ankleActPack':
        keys = [key for key in list(dataOSL.keys()) if (key.lower().startswith('tim') or key.lower().startswith('ank'))] 
    elif sensors == 'kneeActPack':
        keys = [key for key in list(dataOSL.keys()) if (key.lower().startswith('tim') or key.lower().startswith('kne'))]  
    elif sensors == 'IMU':
        keys = [key for key in list(dataOSL.keys()) if (key.lower().startswith('tim') or key.lower().startswith('imu'))]
    elif sensors == 'loadCell':
        keys = [key for key in list(dataOSL.keys()) if (key.lower().startswith('tim') or key.lower().startswith('loa'))]        
    else:
        raise Exception('Non-valid set of keys. Please check "if" within log_OSL')

    with open(logFile, 'w') as csvfile:
        writer = csv.writer(csvfile)
        # writing data rows
        writer.writerow(keys) 

    return {'log': logFile,'sensors': sensors, 'ini_time': initial_time}

# EXAMPLE FUNCTIONS
def example_read(fxs, ankID, kneID, IMU):
    """
    Example function to read the loadcell
    """
    for i in range(10000):
        iniTime         = time.perf_counter_ns() 
        state           = fxs.read_device(kneID)
        loadCellDict    = state2LoCe(state)
        IMUState        = IMU.getDataPackets(TIMEOUT)
        print("Fx: {:>10.4f} [N] ||   Fy: {:>10.4f} [N] ||   Fz: {:10.4f} [N] ||   "
              "Mx: {:10.4f} [Nm] ||   My: {:10.4f} [Nm] ||   Mz: {:10.4f} [Nm]".format(
                loadCellDict['loadCelFx'], loadCellDict['loadCelFy'],
                loadCellDict['loadCelFz'], loadCellDict['loadCelMx'], 
                loadCellDict['loadCelMy'], loadCellDict['loadCelMz'] ) )
        deltaTime       = time.perf_counter_ns() - iniTime
    return

def test_motion(fxs, ankID, kneID, IMU):
    """
    Move the knee and ankle to a know position and send it back to 0 configuration
    """
    # Set controller gains https://dephy.com/wiki/flexsea/doku.php?id=controlgains
    G_K = {"kp": 40, "ki": 400, "K": 60, "B": 0, "FF": 1}  # Knee controller gains
    G_A = {"kp": 40, "ki": 400, "K": 60, "B": 0, "FF": 1}  # Ankle controller gains

    # Initial data from the OSL
    kneSta  = fxs.read_device(kneID)
    ankSta  = fxs.read_device(ankID)
    IMUPac  = IMU.getDataPackets(TIMEOUT)
    dataOSL = read_OSL(kneSta, ankSta, IMUPac)
    logger = ini_log(dataOSL,sensors="all_sensors",trialName="test_motion")

    encMap  = read_enc_map(dataOSL)

    fxs.set_gains(ankID, G_A["kp"], G_A["ki"], 0, G_A["K"], G_A["B"], G_A["FF"])
    fxs.set_gains(kneID, G_K["kp"], G_K["ki"], 0, G_K["K"], G_K["B"], G_K["FF"])

    print('\nCAUTION: Moving the OSL in 3 seconds')
    def moveJoint (kneAngle, ankAngle):
        "Move knee and ankle joint to a position in degrees."
        # Setting control mode and approaching initial position [Be careful fo high gains]
        ankMotCou, kneMotCou = joi2motTic(encMap, kneAngle, ankAngle)
        fxs.send_motor_command(ankID, fxe.FX_IMPEDANCE, ankMotCou)
        fxs.send_motor_command(kneID, fxe.FX_IMPEDANCE, kneMotCou)
        print(f'Moving the knee to {kneAngle} deg and ankle to {ankAngle} deg and waiting for 3 seconds')
        time.sleep(3)
        kneSta  = fxs.read_device(kneID)
        ankSta  = fxs.read_device(ankID)
        IMUPac  = IMU.getDataPackets(TIMEOUT)
        dataOSL = read_OSL(kneSta, ankSta, IMUPac, logger['ini_time'],encMap)
        log_OSL(dataOSL, logger)
        
    # moveJoint(kneAngle = -5,  ankAngle = 0)
    moveJoint(kneAngle = -5,  ankAngle = 0)
    moveJoint(kneAngle = -45,  ankAngle = 19)
    moveJoint(kneAngle = -90,  ankAngle = -10)
    # moveJoint(kneAngle = -5,  ankAngle = 0)
    # moveJoint(kneAngle = -5,  ankAngle = 10)
    # moveJoint(kneAngle = -5,  ankAngle = 0)
    # moveJoint(kneAngle = -5,  ankAngle = -10)
    # moveJoint(kneAngle = -5,  ankAngle = 9)    
    moveJoint(kneAngle = -5,  ankAngle = 0)    
    return

if __name__ == "__main__":
    """
    Test drive of the locoOSL. Call with "home" from the prompt for home routine.
    """
    # ----------------TODO: change these constants to match your setup -------------------
    ANK_PORT = r'/dev/ttyACM1'
    KNE_PORT = r'/dev/ttyACM0'
    IMU_PORT = r'/dev/ttyUSB0'
    TIMEOUT  = 500                  #Timeout in (ms) to read the IMU
    fxs = flex.FlexSEA()
    # ---------------- INITIALIZATION ----------------------------------------------------
    # Connect with knee and ankle actuator packs
    kneID = fxs.open(port = KNE_PORT, baud_rate = 230400, log_level = 0)
    ankID = fxs.open(port = ANK_PORT, baud_rate = 230400, log_level = 0)

    # Initialize IMU - The sample rate (SR) of the IMU controls the SR of the OS
    connection = ms.Connection.Serial(IMU_PORT, 921600)
    #connection = ms.Connection.Serial(IMU_PORT, 230400)
    IMU = ms.InertialNode(connection)
    IMU.setToIdle()
    packets = IMU.getDataPackets(TIMEOUT)       # Clean the internal circular buffer.

    # Set streaming        
    IMU.resume()
    fxs.start_streaming(kneID, freq = 100, log_en = True)
    fxs.start_streaming(ankID, freq = 100, log_en = True)
    time.sleep(0.01)                            # Healthy pause before using OSL

    # ----------------------------- MAIN LOOP --------------------------------------------
    try:
        if ( len(sys.argv) > 1 and sys.argv[1] == "home" ):
            home_joint(fxs, kneID, IMU, "knee" , jointVolt = 2000, motTorThr = 0.45)  # flex2ext
            home_joint(fxs, ankID, IMU, "ankle", jointVolt = -1500, motTorThr = 0.40) # plan2dor
        elif(  len(sys.argv) > 1 and sys.argv[1] == "move" ):
            test_motion(fxs, ankID, kneID, IMU)
        else:
            example_read(fxs, ankID, kneID, IMU)
            # test_motion(fxs, ankID, kneID, IMU)
    finally:        
        # Do anything but turn off the motors at the end of the program
        fxs.send_motor_command(ankID, fxe.FX_NONE, 0)
        fxs.send_motor_command(kneID, fxe.FX_NONE, 0)
        IMU.setToIdle()
        time.sleep(0.02)
        fxs.close(ankID)
        fxs.close(kneID)    
        print('Communication with ActPacks closed and IMU set to idle')    