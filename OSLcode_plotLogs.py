import numpy as np
import matplotlib.pyplot as plt
import csv
logFile = r"/home/pi/OSL/210617_121732_PV_Siavash_walk.csv"



datatxt = np.genfromtxt(logFile , delimiter=',', names=True)


#names to be tracked from the log file
var1 = "ankMotTic"
var2 = "kneMotTic"
var3 = "ThighSagi"
var4 = "loadCelFz"
var5 = "FKnetics"
var6 = "FAnktics"
var7 = "ankJoiPos"
var8 = "kneJoiPos"
var9 = "ankMotTor"
var10 = "kneMotTor"

#Actual trajectory obtained from log files
actualTrajectory={
    'AnkleAct':datatxt[var1],
    'KneeAct':datatxt[var2],
    "ThighSagi":datatxt[var3],
    "loadCelFz":datatxt[var4],
    "PV":datatxt['PV'],
    'FKnetics':datatxt[var5],
    'FAnktics':datatxt[var6],
    'AnkleAngle':datatxt[var7],
    'KneeAngle':datatxt[var8],
    'AnkleTorque':datatxt[var9],
    'KneeTorque':datatxt[var10]
}

#reference trajectory obtained from datasets
referenceTrajectory= {
    'AnkleRef':datatxt['refAnk'],
    'KneeRef':datatxt['refKnee']
}


    
#plotting the graphs and saving them as png

ranA =500#2420#1100# Range A
ranB = 3750#len(referenceTrajectory['AnkleRef'])-1#2600#3650 Range B

xindex = np.array(range(ranB-ranA))*10



# ranA = 0 # Range A
# ranB = 5000 # Range B

fig, axs = plt.subplots(3, 1)

axs[0].set_xlabel('Time(ms)')
axs[0].set_ylabel('Ankle Angle (Degress) (+Plantar.)')
axs[0].plot(xindex,referenceTrajectory['AnkleRef'][ranA:ranB])
axs[0].plot(xindex,actualTrajectory['AnkleAngle'][ranA:ranB])
axs[0].set_ylim([-30,20])
axs[0].set_title(logFile)
# axs[0].legend(["Actual trajectory","Reference Trajectory"])
axs[0].legend(["Commanded Value","Measured Value"])
axs[1].set_xlabel('Time(ms)')
axs[1].set_ylabel('Knee (Angle Degrees) (- Flex.)')
axs[1].plot(xindex,referenceTrajectory['KneeRef'][ranA:ranB])
axs[1].plot(xindex,actualTrajectory['KneeAngle'][ranA:ranB])
axs[1].set_ylim([-70,10])
# axs[1].legend(["Actual trajectory","Reference Trajectory"])
axs[1].legend(["Commanded Value","Measured Value"])
axs[2].set_xlabel('Time(ms)')
axs[2].set_ylabel('Phase Variable')
axs[2].plot(xindex,actualTrajectory['PV'][ranA:ranB])
fig.set_size_inches(22, 13)
plt.savefig(logFile + '.png', dpi=100)
plt.show()


# Ankle torque plot
fig, axs = plt.subplots(2, 1)
axs[0].set_xlabel('Time(ms)')
axs[0].set_ylabel('Ankle Angle (Degress) (+Plantar.)')
axs[0].plot(xindex,referenceTrajectory['AnkleRef'][ranA:ranB])
axs[0].plot(xindex,actualTrajectory['AnkleAngle'][ranA:ranB])
axs[0].set_ylim([-30,20])
axs[0].set_title(logFile + "Ankle Joint Angle and Torque")
# axs[0].legend(["Actual trajectory","Reference Trajectory"])
axs[0].legend(["Commanded Value","Measured Value"])

axs[1].set_xlabel('Time(ms)')
axs[1].set_ylabel('Ankle Joint Torque (Nm)')
axs[1].plot(xindex,actualTrajectory['AnkleTorque'][ranA:ranB]*58)
axs[1].set_ylim([0,100])
# axs[1].plot(actualTrajectory['AnkleTorque'][ranA:ranB])
#axs[1].set_ylim([-70,10])
# axs[1].legend(["Actual trajectory","Reference Trajectory"])
# axs[1].legend(["Commanded Value","Measured Value"])

fig.set_size_inches(22, 13)
plt.savefig(logFile + '_ankle_torque.png', dpi=100)
plt.show()

# Knee torque plot
fig, axs = plt.subplots(2, 1)
axs[0].set_xlabel('Time(ms)')
axs[0].set_ylabel('Kne Angle (Degress) (+Plantar.)')
axs[0].plot(xindex,referenceTrajectory['KneeRef'][ranA:ranB])
axs[0].plot(xindex,actualTrajectory['KneeAngle'][ranA:ranB])
axs[0].set_ylim([-70,10])
axs[0].set_title(logFile+ "Knee Joint Angle and Torque")
# axs[0].legend(["Actual trajectory","Reference Trajectory"])
axs[0].legend(["Commanded Value","Measured Value"])

axs[1].set_xlabel('Time(ms)')
axs[1].set_ylabel('Knee Joint Torque (Nm)')
axs[1].plot(xindex,actualTrajectory['KneeTorque'][ranA:ranB]*49.4)
axs[1].set_ylim([-100,100])
# axs[1].plot(actualTrajectory['AnkleTorque'][ranA:ranB])
#axs[1].set_ylim([-70,10])
# axs[1].legend(["Actual trajectory","Reference Trajectory"])
# axs[1].legend(["Commanded Value","Measured Value"])

fig.set_size_inches(22, 13)
plt.savefig(logFile + '_knee_torque.png', dpi=100)
plt.show()