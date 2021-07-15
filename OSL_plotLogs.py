import numpy as np
import matplotlib.pyplot as plt
from EKF import joints_control
from model_framework import *
import csv

logFile = r"OSL_walking_data/210714_113523_OSL_benchtop_test.csv"
datatxt = np.genfromtxt(logFile , delimiter=',', names = True)

#Actual trajectory obtained from log files
actualTrajectory = {
    "ThighSagi": datatxt["ThighSagi"],
    #"PV": datatxt['PV'],
    "AnkleAngle": datatxt["ankJoiPos"],
    "KneeAngle": datatxt["kneJoiPos"]
}

#reference trajectory obtained from datasets
referenceTrajectory = {
    "AnkleRef": datatxt["refAnk"],
    "KneeRef": datatxt["refKnee"]
}

# EKF data
ekfEstimates = {
    "phase": datatxt["phase"],
    "phase_dot": datatxt["phase_dot"],
    "stride_length": datatxt["stride_length"],
    "ramp": datatxt["ramp"],
    
    # NOTICE: "_bf" here is for correcting the error in the csv file
    "thigh_angle_pred": datatxt["thigh_angle_pred"], 
    "thigh_angle_vel_pred": datatxt["thigh_angle_vel_pred"],
    "atan2_pred": datatxt["atan2_pred"],
    
    "thigh_angle_vel": datatxt["thigh_angle_vel"],
    "atan2": datatxt["atan2"]
}

## Generating joints angles using the kinematics model
knee_angle_kmodel = np.zeros((len(ekfEstimates['phase']), 1))
ankle_angle_kmodel = np.zeros((len(ekfEstimates['phase']), 1))
for i in range(len(ekfEstimates['phase'])):
    joint_angles = joints_control(ekfEstimates['phase'][i], ekfEstimates['phase_dot'][i],
                                  ekfEstimates['stride_length'][i], ekfEstimates['ramp'][i])          
    knee_angle_kmodel[i] = joint_angles[0]
    ankle_angle_kmodel[i] = joint_angles[1]

############## Plotting and Saving #################
ranA = 0
ranB = len(datatxt["Time"])
xindex = datatxt["Time"]

## Figure 1
fig, axs = plt.subplots(3, 1)
axs[0].set_title("Joint Angles Commands")
axs[0].set_ylabel('EKF phase')
#axs[0].plot(xindex, actualTrajectory['PV'][ranA:ranB]/998)
axs[0].plot(xindex, ekfEstimates['phase'][ranA:ranB], 'r-')
#axs[0].legend(["PV/998","EKF Phase"])

axs[1].set_ylabel('Ankle angle (Deg)')
axs[1].plot(xindex, actualTrajectory['AnkleAngle'][ranA:ranB])
axs[1].plot(xindex, referenceTrajectory['AnkleRef'][ranA:ranB], 'r-')
#axs[1].plot(xindex, ankle_angle_kmodel[ranA:ranB], 'm-')
#axs[1].set_ylim([-30,20])
axs[1].legend(["Measured", "Commanded: Edgar\'s trajectory", "Command: kinematic model"])

axs[2].set_xlabel('Time(s)')
axs[2].set_ylabel('Knee angle (Deg)')
axs[2].plot(xindex, actualTrajectory['KneeAngle'][ranA:ranB])
axs[2].plot(xindex, referenceTrajectory['KneeRef'][ranA:ranB], 'r-')
#axs[2].plot(xindex, knee_angle_kmodel[ranA:ranB], 'm-')
axs[2].set_ylim([-70,10])
axs[2].legend(["Measured", "Commanded: Edgar\'s trajectory", "Command: kinematic model"])

fig.set_size_inches(22, 13)
plt.savefig(logFile + 'Joints_Commands.png', dpi=100)

## Figure 2
fig, axs = plt.subplots(4, 1)
axs[0].set_ylabel('EKF Phase')
axs[0].plot(xindex, ekfEstimates['phase'][ranA:ranB], 'r-')
#axs[0].set_ylim([-30,20])
axs[0].set_title("EKF State Estimates")
# axs[0].legend(["Actual trajectory","Reference Trajectory"])
#axs[0].legend(["Commanded Value","Measured Value"])

axs[1].set_ylabel('EKF Phase Rate (1/s)')
axs[1].plot(xindex, ekfEstimates['phase_dot'][ranA:ranB], 'r-')
#axs[1].set_ylim([0,100])

axs[2].set_ylabel('EKF Stride Length (m)')
axs[2].plot(xindex, ekfEstimates['stride_length'][ranA:ranB], 'r-')

axs[3].set_xlabel('Time(s)')
axs[3].set_ylabel('EKF Ramp Angle (deg)')
axs[3].plot(xindex, ekfEstimates['ramp'][ranA:ranB], 'r-')

fig.set_size_inches(22, 13)
plt.savefig(logFile + 'EKF_estimates.png', dpi=100)

## Figure 3
fig, axs = plt.subplots(3, 1)
axs[0].set_title("Measurements")
axs[0].set_ylabel('Global Thigh Angle (deg)')
axs[0].plot(xindex, actualTrajectory["ThighSagi"][ranA:ranB] * 180 / np.pi, 'k-')
axs[0].plot(xindex, ekfEstimates["thigh_angle_pred"][ranA:ranB], 'r-')
axs[0].legend(["Actual", "EKF Predicted"])

axs[1].set_ylabel('Global Thigh Angle Vel (deg/s)')
axs[1].plot(xindex, ekfEstimates["thigh_angle_vel"][ranA:ranB], 'k-')
axs[1].plot(xindex, ekfEstimates["thigh_angle_vel_pred"][ranA:ranB], 'r-')
axs[1].legend(["Actual", "EKF Predicted"])

axs[2].set_xlabel('Time(s)')
axs[2].set_ylabel('Atan2')
axs[2].plot(xindex, ekfEstimates["atan2"][ranA:ranB], 'k-')
axs[2].plot(xindex, ekfEstimates["atan2_pred"][ranA:ranB], 'r')
axs[2].legend(["Actual", "EKF Predicted"])

fig.set_size_inches(22, 13)
plt.savefig(logFile + 'EKF_measurements.png', dpi=100)

plt.show()