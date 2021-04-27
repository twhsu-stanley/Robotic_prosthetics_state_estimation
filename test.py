import numpy as np

s = [1,3]

Y = np.arange(30).reshape(5,6)
Y[-1,0] += 20
print('full Y', Y)
print(Y[np.ix_(s,s)])
print(Y[s, :])

P = np.arange(8)
print('P:', P)

print(P[s])