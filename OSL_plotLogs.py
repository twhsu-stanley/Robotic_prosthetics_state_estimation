import numpy as np
import matplotlib.pyplot as plt
import csv
logFile = r"OSL_walking_data/210617_122334_PV_Siavash_walk_500_2500.csv"
datatxt = np.genfromtxt(logFile , delimiter=',', names = True)

#Actual trajectory obtained from log files
actualTrajectory = {
    "ThighSagi": datatxt["ThighSagi"],
    "PV": datatxt['PV'],
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
    
    "thigh_angle_pred": datatxt["thigh_angle_pred"],
    "thigh_angle_vel_pred": datatxt["thigh_angle_vel_pred"],
    "atan2_pred": datatxt["atan2_pred"],
    
    "thigh_angle_vel": datatxt["thigh_angle_vel"],
    "atan2": datatxt["atan2"]
}

############## Plotting and Saving #################
ranA = 0
ranB = len(datatxt["Time"])
xindex = datatxt["Time"]

## Figure 1
fig, axs = plt.subplots(3, 1)
axs[0].set_xlabel('Time(s)')
axs[0].set_ylabel('Phase Variable')
axs[0].plot(xindex, actualTrajectory['PV'][ranA:ranB]/998)
axs[0].plot(xindex, ekfEstimates['phase'][ranA:ranB])
axs[0].legend(["PV/998","EKF Phase"])

axs[1].set_xlabel('Time(s)')
axs[1].set_ylabel('Ankle Angle (Deg) (+Plantar.)')
axs[1].plot(xindex, referenceTrajectory['AnkleRef'][ranA:ranB])
axs[1].plot(xindex, actualTrajectory['AnkleAngle'][ranA:ranB])
axs[1].set_ylim([-30,20])
axs[1].set_title(logFile)
axs[1].legend(["Commanded Value","Measured Value"])

axs[2].set_xlabel('Time(s)')
axs[2].set_ylabel('Knee Angle (Deg) (- Flex.)')
axs[2].plot(xindex, referenceTrajectory['KneeRef'][ranA:ranB])
axs[2].plot(xindex, actualTrajectory['KneeAngle'][ranA:ranB])
axs[2].set_ylim([-70,10])
axs[2].legend(["Commanded Value","Measured Value"])

fig.set_size_inches(22, 13)
plt.savefig(logFile + 'Joints_Commands.png', dpi=100)

## Figure 2
fig, axs = plt.subplots(4, 1)
axs[0].set_xlabel('Time(s)')
axs[0].set_ylabel('Phase')
axs[0].plot(xindex, ekfEstimates['phase'][ranA:ranB])
#axs[0].set_ylim([-30,20])
axs[0].set_title("EKF Estimates")
# axs[0].legend(["Actual trajectory","Reference Trajectory"])
#axs[0].legend(["Commanded Value","Measured Value"])

axs[1].set_xlabel('Time(s)')
axs[1].set_ylabel('Phase Rate (1/s)')
axs[1].plot(xindex, ekfEstimates['phase_dot'][ranA:ranB])
#axs[1].set_ylim([0,100])

axs[2].set_xlabel('Time(s)')
axs[2].set_ylabel('Stride Length (m)')
axs[2].plot(xindex, ekfEstimates['stride_length'][ranA:ranB])

axs[3].set_xlabel('Time(ms)')
axs[3].set_ylabel('Ramp Angle (deg)')
axs[3].plot(xindex, ekfEstimates['ramp'][ranA:ranB])

fig.set_size_inches(22, 13)
plt.savefig(logFile + 'EKF_estimates.png', dpi=100)

## Figure 3
fig, axs = plt.subplots(3, 1)
axs[0].set_xlabel('Time(s)')
axs[0].set_ylabel('Global Thigh Angle (deg)')
axs[0].plot(xindex, actualTrajectory["ThighSagi"][ranA:ranB])
axs[0].plot(xindex, ekfEstimates["thigh_angle_pred"][ranA:ranB])
axs[0].legend(["Measured","EKF Predicted"])

axs[1].set_xlabel('Time(s)')
axs[1].set_ylabel('Global Thigh Angle Vel (deg/s)')
axs[1].plot(xindex, ekfEstimates["thigh_angle_vel"][ranA:ranB])
axs[1].plot(xindex, ekfEstimates["thigh_angle_vel_pred"][ranA:ranB])
axs[1].legend(["Measured","EKF Predicted"])

axs[2].set_xlabel('Time(s)')
axs[2].set_ylabel('Atan2')
axs[2].plot(xindex, ekfEstimates["atan2"][ranA:ranB])
axs[2].plot(xindex, ekfEstimates["atan2_pred"][ranA:ranB])
axs[2].legend(["Measured","EKF Predicted"])

fig.set_size_inches(22, 13)
plt.savefig(logFile + 'EKF_measurements.png', dpi=100)

plt.show()