import numpy as np
from Models import regression as reg

# q8+q9:
N = 100                     # number of training points
N_test = 5000               # number of test points
d = 2                       # number of dimensions
minAxis = -1                # min value for each dimension
maxAxis = 1                 # max value for each dimension
lr = 0.01                   # learning rate
numIter = 100               # number of iterations
threshold  = 0.01           # weight update stopping threshold

test_cross_entropy = []     # test cross-entropy score for each iteration
epoch_counts = []           # number of epochs for each iteration

# Instantiate new Logistic Regression object
logReg = reg.LogisticRegression(lr, threshold)

for i in range(numIter):
    # Create target data set X, target function f, and values y
    f_points = np.random.uniform(minAxis, maxAxis, size=(2, 2))                 # find two random points
    f_slope, f_intercept = np.polyfit(f_points[:, 0], f_points[:, 1], 1)        # and use them to create target function f
    X = np.random.uniform(minAxis, maxAxis, size=(N, d))                        # create input vector X
    y = np.sign(X[:, 0] * f_slope - X[:, 1] + f_intercept)                      # determine y (= f evaluated on X)
    X = np.hstack((np.ones((N, 1)), X))                                         # add bias to X

    # Fit the model using Stochastic Gradient Descent, store the number of epochs
    epoch_counts.append(logReg.fit(X, y))

    # Evaluate E-out on a separate set of data points
    XTest = np.random.uniform(minAxis, maxAxis, size=(N_test, d))               # create input vector XTest
    yTest = np.sign(XTest[:, 0] * f_slope - XTest[:, 1] + f_intercept)          # determine yTest (= f evaluated on XTest)
    XTest = np.hstack((np.ones((N_test, 1)), XTest))                            # add bias to XTest

    # Calculate and store the error
    test_cross_entropy.append(logReg.cross_entropy(XTest, yTest))

    # Print info about each iteration
    print('Iter: {}, # epochs: {}, test cross-entropy error: {}'.format(i, epoch_counts[-1], test_cross_entropy[-1]))

# Print overall summary
print('\n                Average test cross-entropy: {}'.format(np.mean(test_cross_entropy)))
print('                       Average # of epochs: {}'.format(np.mean(epoch_counts)))

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

