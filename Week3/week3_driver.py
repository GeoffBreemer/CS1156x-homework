import numpy as np

eps = 0.05
M = 100

N = np.log( 0.03/(2*M)) / (-2 * eps * eps)
print('N = {}'.format(N))