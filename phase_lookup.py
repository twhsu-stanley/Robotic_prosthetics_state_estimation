from matplotlib import interactive
import numpy as np
import matplotlib.pyplot as plt
from model_framework import *
from load_Psi import load_Psi 
import time


def phase_lookup_table(plot = False):
    ## Create a lookup table that mpas global thigh angles to phases
    #  at normalized_stride_length = 1.5
    m_model = model_loader('Measurement_model_globalThighAngles_globalThighVelocities_atan2_globalFootAngles.pickle')
    Psi = load_Psi()

    res = 1000
    phase = np.linspace(0, 0.5, res)

    step_length = 1.5
    globalThighAngle_pred = model_prediction(m_model.models[0], Psi['globalThighAngles'],
                                            phase, np.zeros(res), step_length*np.ones(res), np.zeros(res))
    #globalThighAngle_max = np.amax(globalThighAngle_pred)
    #globalThighAngle_min = np.amin(globalThighAngle_pred)

    # inverse mapping
    inv_res = 1000
    #globalThighAngle = np.linspace(globalThighAngle_min, globalThighAngle_max, inv_res)
    globalThighAngle = np.linspace(-180, 180, inv_res)
    inv_phase = np.zeros(inv_res)
    for i in range(inv_res):
        min_error = 10000
        for j in range(res):
            if abs(globalThighAngle[i] - globalThighAngle_pred[j]) < min_error:
                min_error = abs(globalThighAngle[i] - globalThighAngle_pred[j])
                inv_phase[i] = phase[j]
    
    if plot == True:
        plt.figure()
        plt.plot(globalThighAngle, inv_phase)
        plt.plot(globalThighAngle_pred, phase, 'r--')
        plt.xlabel("globalThighAngle")
        plt.ylabel("phase (stance)")
        plt.grid()
        plt.xlim((-180, 180))
        #plt.show()
    
    return inv_phase

if __name__ == '__main__': 
    inv_phase = phase_lookup_table(plot = True)

    # an example of using the lookup table
    theta_th = -12.5
    k = int( (theta_th + 180) / (360/(1000-1)) )
    pv = inv_phase[k]
    print(pv)
    plt.show()
