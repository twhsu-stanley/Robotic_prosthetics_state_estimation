import numpy as np

R = np.ones((8,8)) * 2
print(R)
U = np.diag([1, 1, 1, 1, 1/2, 1/2, 1/2, 1])
R = U @ R @ U.T
print(R)

