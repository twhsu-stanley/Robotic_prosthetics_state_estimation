import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import numpy as np

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

