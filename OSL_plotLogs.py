import numpy as np
import matplotlib.pyplot as plt
from EKF import joints_control
from model_framework import *
import csv

logFile = r"OSL_walking_data/211014_130906_OSL_benchtop_swing_test.csv"
# 211014_130906_OSL_benchtop_swing_test
# 211014_130313_OSL_benchtop_swing_test
# 211014_124556_OSL_benchtop_swing_test

datatxt = np.genfromtxt(logFile , delimiter=',', names = True)

fs = 1 / np.average(np.diff(datatxt["Time"]))
print("Average fs = %4.2f Hz" % fs)

#Actual trajectory obtained from log files
actualTrajectory = {
    "global_thigh_angle": datatxt["ThighSagi"],
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
    
    "global_thigh_angle_pred": datatxt["global_thigh_angle_pred"], 
    "global_thigh_vel_pred": datatxt["global_thigh_vel_pred"],
    "atan2_pred": datatxt["atan2_pred"],
    
    #"global_thigh_vel_lp": datatxt["global_thigh_vel_lp"],

    "global_thigh_angle_lp": datatxt["global_thigh_angle_lp"],
    "global_thigh_vel_lp_2": datatxt["global_thigh_vel_lp_2"],
    "global_thigh_angle_max": datatxt["global_thigh_angle_max"],
    "global_thigh_angle_min": datatxt["global_thigh_angle_min"],
    "global_thigh_vel_max": datatxt["global_thigh_vel_max"],
    "global_thigh_vel_min": datatxt["global_thigh_vel_min"],
    "phase_x": datatxt["phase_x"],
    "phase_y": datatxt["phase_y"],
    "radius": datatxt["radius"],
    "atan2": datatxt["atan2"],
    
    "MD_residual": datatxt["MD_residual"],
    "lost": datatxt["lost"],
    "hold": datatxt["hold"],
    "peg": datatxt["peg"]
}

## Generating joints angles using the kinematics model ============================================
knee_angle_kmodel = np.zeros((len(ekfEstimates['phase']), 1))
ankle_angle_kmodel = np.zeros((len(ekfEstimates['phase']), 1))
for i in range(len(ekfEstimates['phase'])):
    joint_angles = joints_control(ekfEstimates['phase'][i], ekfEstimates['phase_dot'][i],
                                  ekfEstimates['stride_length'][i], ekfEstimates['ramp'][i])          
    knee_angle_kmodel[i] = joint_angles[0]
    ankle_angle_kmodel[i] = joint_angles[1]

## Plotting and Saving =============================================================================
ranA = 0
ranB = len(datatxt["Time"])
time = datatxt["Time"]

print("Average fs = %4.2f Hz" % (1 / np.average(np.diff(time)) ) )
print("Average dt = %4.2f Hz" % np.average(np.diff(time)))

## Figure 1
fig, axs = plt.subplots(3, 1)
axs[0].set_title("Joint Angles Commands")
axs[0].set_ylabel('EKF phase')
#axs[0].plot(time, actualTrajectory['PV'][ranA:ranB]/998)
axs[0].plot(time, ekfEstimates['phase'][ranA:ranB], 'r-')
#axs[0].legend(["PV/998","EKF Phase"])
axs[0].grid()

axs[1].set_ylabel('Ankle angle (Deg)')
#axs[1].plot(time, actualTrajectory['AnkleAngle'][ranA:ranB], label = "Measured")
#axs[1].plot(time, referenceTrajectory['AnkleRef'][ranA:ranB], 'r-', label = "Commanded: Prescribed trajectory")
axs[1].plot(time, ankle_angle_kmodel[ranA:ranB], 'm-', label = "Command: kinematic model")
axs[1].set_ylim([-20,30])
axs[1].legend()
axs[1].grid()

axs[2].set_xlabel('Time(s)')
axs[2].set_ylabel('Knee angle (Deg)')
#axs[2].plot(time, actualTrajectory['KneeAngle'][ranA:ranB], label = "Measured")
#axs[2].plot(time, referenceTrajectory['KneeRef'][ranA:ranB], 'r-', label = "Commanded: Prescribed trajectory")
axs[2].plot(time, knee_angle_kmodel[ranA:ranB], 'm-', label = "Command: kinematic model")
axs[2].set_ylim([-70,10])
axs[2].legend()
axs[2].grid()

fig.set_size_inches(22, 13)
plt.savefig(logFile + 'Joints_Commands.png', dpi=100)

## Figure 2
fig, axs = plt.subplots(4, 1)
axs[0].set_ylabel('EKF Phase')
axs[0].plot(time, ekfEstimates['phase'][ranA:ranB], 'r-')
#axs[0].set_ylim([-30,20])
axs[0].set_title("EKF State Estimates")
# axs[0].legend(["Actual trajectory","Reference Trajectory"])
#axs[0].legend(["Commanded Value","Measured Value"])

axs[1].set_ylabel('EKF Phase Rate (1/s)')
axs[1].plot(time, ekfEstimates['phase_dot'][ranA:ranB], 'r-')
#axs[1].set_ylim([0,100])

axs[2].set_ylabel('EKF Normalized Stride Length (m)')
axs[2].plot(time, ekfEstimates['stride_length'][ranA:ranB], 'r-')

axs[3].set_xlabel('Time(s)')
axs[3].set_ylabel('EKF Ramp Angle (deg)')
axs[3].plot(time, ekfEstimates['ramp'][ranA:ranB], 'r-')

fig.set_size_inches(22, 13)
plt.savefig(logFile + 'EKF_estimates.png', dpi=100)

## Figure 3
fig, axs = plt.subplots(3, 1)
axs[0].set_title("Measurements")
axs[0].set_ylabel('Global Thigh Angle (deg)')
axs[0].plot(time, actualTrajectory["global_thigh_angle"][ranA:ranB] * 180 / np.pi, 'k-')
axs[0].plot(time, ekfEstimates["global_thigh_angle_pred"][ranA:ranB], 'r-')
axs[0].legend(["Actual", "EKF Predicted", "Band-pass filtered"])

axs[1].set_ylabel('Global Thigh Angle Vel (deg/s)')
axs[1].plot(time, ekfEstimates["global_thigh_vel_lp_2"][ranA:ranB], 'k-')
axs[1].plot(time, ekfEstimates["global_thigh_vel_pred"][ranA:ranB], 'r-')
axs[1].legend(["Actual", "EKF Predicted"])

axs[2].set_xlabel('Time(s)')
axs[2].set_ylabel('Atan2')
axs[2].plot(time, ekfEstimates["atan2"][ranA:ranB], 'k-')
axs[2].plot(time, ekfEstimates["atan2_pred"][ranA:ranB], 'r')
axs[2].legend(["Actual", "EKF Predicted"])

fig.set_size_inches(22, 13)
plt.savefig(logFile + 'EKF_measurements.png', dpi=100)

## Figure 4
fig, axs = plt.subplots(3, 1)
axs[0].set_title("Atan2 Computation")
axs[0].set_ylabel('Global Thigh Angle (deg)')
axs[0].plot(time, ekfEstimates["global_thigh_angle_lp"][ranA:ranB], 'k-', label = 'low-passed')
axs[0].plot(time, actualTrajectory["global_thigh_angle"][ranA:ranB] * 180 / np.pi, 'm-', alpha = 0.4, label = 'raw')
axs[0].plot(time, ekfEstimates["global_thigh_angle_max"][ranA:ranB], 'r-', label = 'max')
axs[0].plot(time, ekfEstimates["global_thigh_angle_min"][ranA:ranB], 'b-', label = 'min')

axs[1].set_ylabel('Global Thigh Velocity (deg/s)')
axs[1].plot(time, ekfEstimates["global_thigh_vel_lp_2"][ranA:ranB], 'k-', label = 'for atan2 computation')
#axs[1].plot(time, ekfEstimates["global_thigh_vel_lp"][ranA:ranB], 'm-', label = 'for vel measurement')
axs[1].plot(time, ekfEstimates["global_thigh_vel_max"][ranA:ranB], 'r-', label = 'max')
axs[1].plot(time, ekfEstimates["global_thigh_vel_min"][ranA:ranB], 'b-', label = 'min')

axs[2].set_xlabel('Time(s)')
axs[2].set_ylabel('Atan2')
axs[2].plot(time, ekfEstimates["atan2"][ranA:ranB], 'k-')

fig.set_size_inches(22, 13)
plt.savefig(logFile + 'EKF_atan2_computation.png', dpi=100)

## Figure 5
plt.figure("Atan2 Phase Portrait")
plt.plot(ekfEstimates["phase_x"][ranA:ranB], ekfEstimates["phase_y"][ranA:ranB])
plt.xlabel("Phase X")
plt.xlabel("Phase Y")
plt.grid()
plt.savefig(logFile + 'EKF_atan2_phasePortrait.png', dpi=100)

## Figure 6
plt.figure("Failure and Position-holding Detectors")
plt.subplot(411)
plt.plot(time, ekfEstimates['MD_residual'][ranA:ranB])
plt.ylabel("MD_residual")
plt.grid()
plt.subplot(412)
plt.plot(time, ekfEstimates['lost'][ranA:ranB])
plt.ylabel("lost (T/F)")
plt.grid()
plt.subplot(413)
plt.plot(time, ekfEstimates['hold'][ranA:ranB])
plt.ylabel("hold (T/F)")
plt.grid()
plt.subplot(414)
plt.plot(time, ekfEstimates['peg'][ranA:ranB])
plt.ylabel("peg (T/F)")
plt.grid()
plt.xlabel("Time (s)")

plt.show()