import numpy as np
from Models import regression as reg

def read_and_transform(filename):
    '''Read the data, split into X and y, apply non-linear transformation'''
    X = np.genfromtxt(filename, delimiter="  ", dtype=None)

    # Split into X and y
    y = X[:, 2]
    X = X[:, 0:2]

    # Transform
    X = np.hstack((np.ones((len(X), 1)),
                   X[:, 0].reshape(len(X), 1),
                   X[:, 1].reshape(len(X), 1),
                   np.square(X[:, 0]).reshape(len(X), 1),
                   np.square(X[:, 1]).reshape(len(X), 1),
                   (X[:, 0] * X[:, 1]).reshape(len(X), 1),
                   np.abs(X[:, 0] - X[:, 1]).reshape(len(X), 1),
                   np.abs(X[:, 0] + X[:, 1]).reshape(len(X), 1),
                   ))

    return (X, y)

# Read X and y and apply non-linear transformation for both data sets
X, y = read_and_transform("in.dta")

# Q1 and Q2:
XTrain = X[:25]
yTrain = y[:25]
XVal = X[25:]
yVal = y[25:]

# Q3 and Q4:
# XVal= X[:25]
# yVal= y[:25]
# XTrain = X[25:]
# yTrain = y[25:]

del X, y    # to avoid confusion
XTest, yTest = read_and_transform("out.dta")

linReg = reg.LinearRegression()

# Loop through various models (k determines which features to use)
krange = [3, 4, 5, 6, 7]
for k in krange:
    # Fit linear regression model
    w = linReg.fit(XTrain[:, 0:k+1], yTrain)

    # Print error for each set for each k
    print("k = {:3d} -> Training: {:8.8f}, Validation: {:8.8f}, Test: {:8.8f}".format(k,
                                                                            linReg.score(XTrain[:, 0:k+1], yTrain),
                                                                            linReg.score(XVal[:, 0:k + 1], yVal),
                                                                            linReg.score(XTest[:, 0:k+1], yTest)))
