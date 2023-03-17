import numpy as np
from model_framework import *

## Load model coefficients
def load_Psi():
    with open('Psi/Psi_globalThighAngles', 'rb') as file:
        Psi_globalThighAngles = pickle.load(file)
    with open('Psi/Psi_globalThighVelocities', 'rb') as file:
        Psi_globalThighVelocities = pickle.load(file)
    with open('Psi/Psi_atan2', 'rb') as file:
        Psi_atan2 = pickle.load(file)
    with open('Psi/Psi_globalFootAngles', 'rb') as file:
        Psi_globalFootAngles = pickle.load(file)
    Psi = {'globalThighAngles': Psi_globalThighAngles, 'globalThighVelocities': Psi_globalThighVelocities,
           'atan2': Psi_atan2, 'globalFootAngles': Psi_globalFootAngles}
    return Psi
