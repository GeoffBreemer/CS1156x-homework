import numpy as np
from Models import regression as reg

def read_and_transform(filename):
    '''
    Read the data, split into X and y, apply non-linear transformation

    Note: input file needs to be comma separated, not spaces
    '''
    X = np.genfromtxt(filename, delimiter=",", dtype=None)

    y = X[:, 2]
    X = X[:, 0:2]
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

# Read training X and y, then apply non-linear transformation
X, y = read_and_transform("in.dta")
XTest, yTest = read_and_transform("out.dta")

linReg = reg.LinearRegression()

# Loop through various values of lambda/k
krange = np.linspace(-3, 3, 7, dtype=int)
for k in krange:
    l = 10 ** k
    # l = 0 # uncomment this line for Q2, comment out this line for all other questions

    # Fit linear regression model
    w = linReg.fit(X, y, reg=l)

    # Determine the in-sample E-aug
    yPred = np.sign(np.dot(X, w))
    missedIn = np.sum((yPred != y)) / len(X)  + l / len(X) * np.dot(w.T, w)

    # And for the out-of-sample E-aug
    yPredTest = np.sign(np.dot(XTest, w))
    missedOut = np.sum((yPredTest != yTest)) / len(XTest)  + l / len(XTest) * np.dot(w.T, w)

    # Print info for each k
    print("k = {:3d} -> E-in: {:8.4f}, E-out: {:8.4f}".format(k, missedIn, missedOut))
