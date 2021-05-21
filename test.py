import numpy as np

R = np.ones((8,8)) * 2
print(R)
U = np.diag([1, 1, 1, 1, 1/2, 1/2, 1/2, 1])
R = U @ R @ U.T
print(R)

R = np.ones((8,8)) * 1
R = R * 4
R1 = U @ R @ U.T
print(R)
R = np.ones((8,8)) * 1
U = np.diag([2, 2, 2, 2, 1, 1, 1, 2])
R2 = U @ R @ U.T
print(R2-R1)