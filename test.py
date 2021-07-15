import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import time
import pickle
from continuous_data import *

a = [0,1,2,3,4]
b = [0,1,4]
for z in zip(a,b):
    print(z)


with open('Continuous_data/GlobalThighAngles_with_Nan.pickle', 'rb') as file:
#with open('Continuous_data/KneeAngles_with_Nan.pickle', 'rb') as file:
        nan_dict = pickle.load(file)
    
for subject in Conti_subject_names():
        for trial in Conti_trial_names(subject):
            if trial == 'subjectdetails':
                continue
            for side in ['left', 'right']:
                if nan_dict[subject][trial][side] == False:
                    print(subject + "/"+ trial + "/"+ side+ ": Trial skipped!")
                    