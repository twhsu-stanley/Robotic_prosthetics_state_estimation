import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import time

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

