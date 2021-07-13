import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import time


# Ankle angle limits (deg)
ankle_max = 18
ankle_min = -10
# Natural frequency of the sine wave (rad/s)
freq = 2 * np.pi * 0.2 # 0.2 Hz
# DC offset of the sine wave
dc_offset_initial = 10
dc_offset_final = (ankle_max + ankle_min) / 2
# Amplitude of the sine wave
amplitude_initial = 0
amplitude_final = (ankle_max - ankle_min) / 2

# Fade-in time (sec)
fade_in_time = 3

## Main Loop #####################################################################################
ankle_ref = []
t_0 = time.perf_counter() # sec
t = 0
while(t < 15):
    t = time.perf_counter() - t_0

    # 1) Sinusoidal ankle command with fade-in effect (deg)
    if t < fade_in_time:
        amplitude = amplitude_initial + (amplitude_final - amplitude_initial) * t / fade_in_time
        dc_offset = dc_offset_initial + (dc_offset_final - dc_offset_initial) * t / fade_in_time
    elif t >= fade_in_time:
        amplitude = amplitude_final
        dc_offset = dc_offset_final

    ankle_cmd = amplitude * np.sin(freq * t) + dc_offset
        
    # Saturation for ankle command
    if ankle_cmd > ankle_max: 
        ankle_cmd = ankle_max
    elif ankle_cmd < ankle_min:
        ankle_cmd = ankle_min

    ankle_ref.append(ankle_cmd)

ankle_ref = np.array(ankle_ref)
plt.figure()
plt.plot(ankle_ref)
plt.show()






print("Knee initial position: %.2f deg" % -5)

a = []
b = []

for i in range(10):
    a.append(1)
    b.append(0)
a = np.array(a)
b = np.array(b)
rmse = np.sqrt(np.square(a - b).mean())
print(rmse)

t_0 = time.perf_counter() # sec
t = 0
while(t < 7):
    t = time.perf_counter() - t_0
    print(t)
    time.sleep(1)

dict1  = {"a": 1, 'b':2}

print(len(dict1))


mat = scipy.io.loadmat('OSL_walking_data/Treadmill_speed1_incline0_file1.mat')
thighY = mat['ThighIMU'][0, 0]['ThetaY']

plt.figure()
plt.plot(mat['ThighIMU'][0, 0]['ThetaY'])
plt.plot(mat['ThighIMU'][0, 0]['ThetaX'])
plt.plot(mat['ThighIMU'][0, 0]['ThetaZ'])
plt.legend(('Y', 'X', 'Z'))

plt.figure()
tt = np.cumsum(mat['ControllerOutputs'][0, 0]['dt']).reshape(-1)

plt.plot(np.diff(tt))
plt.plot(mat['ControllerOutputs'][0, 0]['dt'])
plt.show()

