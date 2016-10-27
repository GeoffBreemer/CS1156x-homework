import numpy as np
from Models import regression as reg

# q8:
N = 100
N_test = 500
d = 2
minAxis = -1
maxAxis = 1
lr = 0.01
numIter = 100
threshold  = 0.01
cross_entropy = []
epoch_counts = []

# https://github.com/lytemar/Caltech-CS1156x-Learning-from-Data/blob/master/HW5_probs8_9.m
for i in range(numIter):
    # Create target data set D, target function f, and values y
    D = np.random.uniform(minAxis, maxAxis, size=(N, d))  # create input vector x
    f_points = np.random.uniform(minAxis, maxAxis, size=(2, 2))  # find two random points
    f_slope, f_intercept = np.polyfit(f_points[:, 0], f_points[:, 1], 1)  # and use them to create target function f
    y = np.sign(D[:, 0] * f_slope - D[:, 1] + f_intercept)  # determine y (= f evaluated on x)

    D = np.hstack((np.ones((N, 1)), D))  # add x0 to x
    y = y.reshape((len(D), 1))

    # Run Logistic Regression with Stochastic Gradient Descent
    LogReg = reg.LogisticRegression(lr, threshold)
    numEpochs = LogReg.fit(D, y)
    epoch_counts.append(numEpochs)

    # Evaluate error on a separate set of data points and calculate the cross-entropy error
    tmpXOrg = np.random.uniform(minAxis, maxAxis, size=(N_test, d))           # create input vector
    yActual = np.sign(tmpXOrg[:,0] * f_slope - tmpXOrg[:, 1] + f_intercept)   # determine y (= f evaluated on tmpXOrg)

    tmpX = np.hstack((np.ones((N_test, 1)), tmpXOrg))                         # add x0 to tmpXOrg
    yActual = yActual.reshape((len(yActual), 1))

    # error = LogReg.cross_entropy(tmpX, yActual)
    error = (1/len(tmpX)) * np.sum(  np.log(1+np.exp(-yActual.dot(LogReg.predict(tmpX))))   )
    cross_entropy.append(error)

    print('iter: {}, num epochs: {}, error: {}\n'.format(i, numEpochs, error))

# Print overall summary
print('                Average cross-entropy: {}'.format(np.mean(cross_entropy)))
print('                  Average # of epochs: {}'.format(np.mean(epoch_counts)))


exit()

# q1: print(0.1**2 * (1- (8+1)/100))
# q4: 2*(ue^v-2ve^-u)*(u*e^v+2*v*e^-u)
# q5+q6:
u = 1
v = 1
N = 1
lr = 0.1

numIter = 10  # 1, 3, 5, 10, 17

for i in range(numIter):
    grad_du = 2*(np.exp(v)+2*v*np.exp(-u))*(u*np.exp(v)-2*v*np.exp(-u))
    grad_dv = 2*(u*np.exp(v)-2*v*np.exp(-u))*(u*np.exp(v) -2*np.exp(-u)  )

    u = u - lr * grad_du
    v = v - lr * grad_dv
    error = np.square(u*np.exp(v)-2*v*np.exp(-u))

    print("(u,v) = ({}, {}), error={}, grad=({}, {})".format(u, v, error, grad_du, grad_dv))

# q7 - coordinate descent:
u = 1
v = 1
N = 1
lr = 0.1

numIter = 15

for i in range(numIter):
    grad_du = 2*(np.exp(v)+2*v*np.exp(-u))*(u*np.exp(v)-2*v*np.exp(-u))
    u = u - lr * grad_du

    grad_dv = 2*(u*np.exp(v)-2*v*np.exp(-u))*(u*np.exp(v) -2*np.exp(-u)  )
    v = v - lr * grad_dv

    error = np.square(u*np.exp(v)-2*v*np.exp(-u))

    print("(u,v) = ({}, {}), error={}, grad=({}, {})".format(u, v, error, grad_du, grad_dv))

