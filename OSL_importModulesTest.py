import sys, time, os
from flexsea import flexsea as flex
from flexsea import fxEnums as fxe
#import matplotlib.pyplot as plt
import numpy as np

sys.path.append(r'/home/pi/OSL-master/locoAPI/') # Path to Loco module
#sys.path.append(r'/home/pi/.local/bin')
sys.path.append(r'/usr/share/python3-mscl/')     # Path of the MSCL - API for the IMU
import locoOSL as loco                           # Module from Locolab
import mscl as msl                               # Module from Microstrain

sys.path.append(r'/home/pi/prosthetic_phase_estimation/')
from EKF import *
from model_framework import *
from scipy.signal import butter, lfilter, lfilter_zi
import sender_test as sender   # for real-time plotting