import numpy as np
import h5py
from data_generators import *
import matplotlib.pyplot as plt

subject_names = get_subject_names()

with open('Reaction_wrench.npz', 'rb') as file:
    RW = np.load(file)
    #for subject in subject_names:
        #print(np.shape(RW[subject]))
    
    d_s = np.shape(RW['AB07'][0])
    plt.figure(1)
    for i in range(100):
        plt.plot(RW['AB07'][1,i,:])

    #plt.show()

with open('Global_thigh_angle.npz', 'rb') as file:
    GT = np.load(file)
    #for subject in subject_names:
        #print(np.shape(GT[subject]))
    plt.figure(2)
    for i in np.linspace(500, 700):
        plt.plot(GT['AB07'][2,i,:])
    plt.show()