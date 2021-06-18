import numpy as nop
from model_fit import *
from model_framework import *

c_model = model_loader('Control_model.pickle')

with open('PikPsi_knee_G.pickle', 'rb') as file:
    Psi_knee = pickle.load(file)
with open('PikPsi_ankle_G.pickle', 'rb') as file:
    Psi_ankle = pickle.load(file)

def joints_control(phases, phase_dots, step_lengths, ramps):
    knee_angle = model_prediction(c_model.models[0], Psi_knee, phases, phase_dots, step_lengths, ramps)
    ankle_angle = model_prediction(c_model.models[1], Psi_ankle, phases, phase_dots, step_lengths, ramps)
    return [knee_angle, ankle_angle]