import numpy as np
import matplotlib.pyplot as plt
from EKF import joints_control
from model_framework import *
import csv

logFile = r"OSL_walking_data/211101_131638_OSL_benchtop_swing_test.csv"
# 211101_131638_OSL_benchtop_swing_test
# 211101_130912_OSL_benchtop_swing_test
# 211101_125910_OSL_benchtop_swing_test

# 211021_184718_OSL_benchtop_swing_test
# 211021_184343_OSL_benchtop_swing_test

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
    
    "global_thigh_angle_lp": datatxt["global_thigh_angle_lp"],
    "global_thigh_vel_lp": datatxt["global_thigh_vel_lp"],
    #"global_thigh_vel_lp_2": datatxt["global_thigh_vel_lp_2"],
    "global_thigh_angle_max": datatxt["global_thigh_angle_max"],
    "global_thigh_angle_min": datatxt["global_thigh_angle_min"],
    "global_thigh_vel_max": datatxt["global_thigh_vel_max"],
    "global_thigh_vel_min": datatxt["global_thigh_vel_min"],
    # "global_thigh_angle_cline": datatxt[global_thigh_angle_cline,
    "phase_x": datatxt["phase_x"],
    "phase_y": datatxt["phase_y"],
    "radius": datatxt["radius"],
    "atan2": datatxt["atan2"],
    "walk": datatxt["walk"],

    "MD": datatxt["MD"],
    "lost": datatxt["lost"]
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

## Figure: Joints control
fig, axs = plt.subplots(6, 1)
axs[0].set_title("Joint Controls")
axs[0].set_ylabel('EKF phase')
#axs[0].plot(time, actualTrajectory['PV'][ranA:ranB]/998)
axs[0].plot(time, ekfEstimates['phase'][ranA:ranB], 'r-')
#axs[0].legend(["PV/998","EKF Phase"])
axs[0].grid()

axs[1].set_ylabel('$l~(m)$')
axs[1].plot(time, ekfEstimates['stride_length'][ranA:ranB], 'r-')
axs[1].grid()

axs[2].set_ylabel('radius & walk')
axs[2].plot(time, ekfEstimates['radius'][ranA:ranB])
axs[2].axhline(y=1, color='r', linestyle='-')
axs[2].plot(time, ekfEstimates['walk'][ranA:ranB] * 10, 'k-')
axs[2].grid()

axs[2].set_ylabel('MD & lost')
axs[3].plot(time, ekfEstimates['MD'][ranA:ranB])
axs[3].axhline(y=40, color='r', linestyle='-')
axs[3].plot(time, ekfEstimates['lost'][ranA:ranB] * 40, 'k-')
axs[3].grid()

axs[4].set_ylabel('Ankle angle (Deg)')
axs[4].plot(time, actualTrajectory['AnkleAngle'][ranA:ranB], label = "Measured")
axs[4].plot(time, referenceTrajectory['AnkleRef'][ranA:ranB], 'r-', label = "Commanded")
#axs[1].plot(time, ankle_angle_kmodel[ranA:ranB], 'm-', label = "Command: kinematic model")
axs[4].set_ylim([-20,30])
axs[4].legend()
axs[4].grid()

axs[5].set_xlabel('Time(s)')
axs[5].set_ylabel('Knee angle (Deg)')
axs[5].plot(time, actualTrajectory['KneeAngle'][ranA:ranB], label = "Measured")
axs[5].plot(time, referenceTrajectory['KneeRef'][ranA:ranB], 'r-', label = "Commanded")
#axs[5].plot(time, knee_angle_kmodel[ranA:ranB], 'm-', label = "Command: kinematic model")
axs[5].set_ylim([-70,10])
axs[5].legend()
axs[5].grid()

fig.set_size_inches(22, 13)
#plt.savefig(logFile + 'Joints_Commands.png', dpi=100)

## Figure: Overall 
fig, axs = plt.subplots(8, 1)
axs[0].set_ylabel('$\phi$')
axs[0].plot(time, ekfEstimates['phase'][ranA:ranB], 'r-')
axs[0].set_title("EKF")
axs[0].grid()

axs[1].set_ylabel('$\dot{\phi}~(s^{-1})$')
axs[1].plot(time, ekfEstimates['phase_dot'][ranA:ranB], 'r-')
axs[1].grid()

axs[2].set_ylabel('$l~(m)$')
axs[2].plot(time, ekfEstimates['stride_length'][ranA:ranB], 'r-')
axs[2].grid()

#axs[3].set_xlabel('Time(s)')
#axs[3].set_ylabel('EKF Ramp Angle (deg)')
#axs[3].plot(time, ekfEstimates['ramp'][ranA:ranB], 'r-')

axs[3].set_ylabel('$\\theta_{th}~(deg)$')
axs[3].plot(time, actualTrajectory["global_thigh_angle"][ranA:ranB] * 180 / np.pi, 'k-')
axs[3].plot(time, ekfEstimates["global_thigh_angle_pred"][ranA:ranB], 'r-')
axs[3].legend(["Actual", "EKF Predicted", "Band-pass filtered"])
axs[3].grid()

axs[4].set_ylabel('$\dot{\\theta_{th}}~(deg/s)$')
axs[4].plot(time, ekfEstimates["global_thigh_vel_lp"][ranA:ranB], 'k-')
axs[4].plot(time, ekfEstimates["global_thigh_vel_pred"][ranA:ranB], 'r-')
axs[4].legend(["Actual", "EKF Predicted"])
axs[4].grid()

axs[5].set_ylabel('Atan2')
axs[5].plot(time, ekfEstimates["atan2"][ranA:ranB], 'k-')
axs[5].plot(time, ekfEstimates["atan2_pred"][ranA:ranB], 'r')
axs[5].legend(["Actual", "EKF Predicted"])
axs[5].grid()

axs[6].set_ylabel('radius')
axs[6].plot(time, ekfEstimates['radius'][ranA:ranB])
axs[6].axhline(y=1, color='r', linestyle='-', label = '1')
axs[6].grid()

axs[7].set_xlabel('Time(s)')
axs[7].set_ylabel('Knee angle (Deg)')
axs[7].plot(time, actualTrajectory['KneeAngle'][ranA:ranB], label = "Measured")
axs[7].plot(time, referenceTrajectory['KneeRef'][ranA:ranB], 'r-', label = "Commanded")
axs[7].set_ylim([-70,10])
axs[7].legend()
axs[7].grid()

fig.set_size_inches(22, 13)
#plt.savefig(logFile + 'EKF_estimates.png', dpi=100)

## Figure: Atan2 Computation
fig, axs = plt.subplots(6, 1)
axs[0].set_title("Atan2 Computation")
axs[0].set_ylabel('Global Thigh Angle (deg)')
axs[0].plot(time, ekfEstimates["global_thigh_angle_lp"][ranA:ranB], 'k-')
#axs[0].plot(time, actualTrajectory["global_thigh_angle"][ranA:ranB] * 180 / np.pi, 'm-', alpha = 0.4)
axs[0].plot(time, ekfEstimates["global_thigh_angle_max"][ranA:ranB], 'r-')
axs[0].plot(time, ekfEstimates["global_thigh_angle_min"][ranA:ranB], 'b-')
#axs[0].plot(time, ekfEstimates["global_thigh_angle_cline"][ranA:ranB], 'k-', label = 'center')
axs[0].grid()

axs[1].set_ylabel('Global Thigh Velocity (deg/s)')
axs[1].plot(time, ekfEstimates["global_thigh_vel_lp"][ranA:ranB], 'k-')
axs[1].plot(time, ekfEstimates["global_thigh_vel_max"][ranA:ranB], 'r-')
axs[1].plot(time, ekfEstimates["global_thigh_vel_min"][ranA:ranB], 'b-')
axs[1].grid()

axs[2].set_ylabel('phase_x')
axs[2].plot(time, ekfEstimates["phase_x"][ranA:ranB], 'k-')
axs[2].grid()

axs[3].set_ylabel('phase_y')
axs[3].plot(time, ekfEstimates["phase_y"][ranA:ranB], 'k-')
axs[3].grid()

axs[4].set_ylabel('Atan2')
axs[4].plot(time, ekfEstimates["atan2"][ranA:ranB], 'k-')
axs[4].grid()

axs[5].set_ylabel('r')
axs[5].plot(time, ekfEstimates["radius"][ranA:ranB], 'b-')
axs[5].axhline(y = 1, color = 'r', linestyle = '-')
axs[5].set_xlabel('Time(s)')
axs[5].grid()

fig.set_size_inches(22, 13)
#plt.savefig(logFile + 'EKF_atan2_computation.png', dpi=100)

## Figure: Phase portrait
plt.figure("Atan2 Phase Portrait")
plt.plot(ekfEstimates["phase_x"][ranA:ranB], ekfEstimates["phase_y"][ranA:ranB])
plt.xlabel("Phase X")
plt.xlabel("Phase Y")
plt.grid()
#plt.savefig(logFile + 'EKF_atan2_phasePortrait.png', dpi=100)

plt.show()