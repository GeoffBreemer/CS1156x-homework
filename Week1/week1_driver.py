# Week 1, Perceptron, Q7-Q10
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Models import regression as reg

# ---------------- Start ----------------

if __name__ == "__main__":
    # ---------------- Constants ----------------
    minAxis = -1                # grid size
    maxAxis = 1
    d = 2                       # number of dimensions
    N = 10                      # number of training points
    num_iter = 250              # maximum number of iterations per run
    num_runs = 1000             # number of runs
    numProb = 1000              # number of out of sample points used to predict P(f <> g)

    #  For q7-8: N =  10, num_runs = 1000
    # For q9-10: N = 100, num_runs = 1000

    numIterations = []
    probabilities = []

    # Execute the Perceptron algorithm
    for run in np.arange(num_runs):
        # Initialise run specific variables
        numIter = 1
        w = np.zeros((1, d + 1))                                        # reset weights vector to zero

        # Create random observations
        xOrg = np.random.uniform(minAxis, maxAxis, size=(N, d))         # create input vector x
        fPts = np.random.uniform(minAxis, maxAxis, size=(2,2))          # find two random points
        f_slope, f_intercept = np.polyfit(fPts[:, 0], fPts[:, 1], 1)    # and use them to create target function f
        y = np.sign(xOrg[:,0] * f_slope - xOrg[:, 1] + f_intercept)     # determine y (= f evaluated on x)
        x = np.hstack((np.ones((N, 1)), xOrg))                          # add x0 to x

        # Run the Perceptron Learning Algorithm
        for iter in np.arange(num_iter):
            hypothesis = np.sign(np.dot(x, w.T).T)                      # calculate the hypothesis: x * w.T
            missed = (y!=hypothesis)                                    # determine misclassified points

            if np.sum(missed) != 0:
                # Pick a random missclassified point
                idx = [i for i, ele in enumerate(missed[0]) if ele==True]   # indices of misclassified points
                tmpIndex = np.random.randint(0, len(idx))                   # pick one of the indices
                rndIndex = idx[tmpIndex]                                    # determine the index in x

                w = w + y[rndIndex] * x[rndIndex]                           # update the weights
                numIter += 1
            else:
                # no misclassifications: done with this run
                break

        # Calculate P(f <> g) for a large number of out of sample points
        tmpXOrg = np.random.uniform(minAxis, maxAxis, size=(numProb, d))          # create input vector
        yActual = np.sign(tmpXOrg[:,0] * f_slope - tmpXOrg[:, 1] + f_intercept)   # determine y (= f evaluated on tmpXOrg)
        tmpX = np.hstack((np.ones((numProb, 1)), tmpXOrg))                  # add x0 to tmpXOrg
        yPred = np.sign(np.dot(tmpX, w.T).T)                                # evaluate the hypothesis g on x
        missed = np.sum((yPred != yActual))                                 # number of misclassified points
        probabilities.append(missed / numProb)                              # keep track of the P(f <> g) of each run

        # Print run summary
        print('Run {:4d}: # iterations: {:3d}, misclassified: {:3f}'.format(run, numIter, missed/numProb))

        # Training plot (careful with large num_runs)
        reg.plotPoints(xOrg, f_slope, f_intercept, y, w[0], minAxis, maxAxis)

        # Test plot (careful with large num_runs)
        reg.plotPoints(tmpXOrg, f_slope, f_intercept, yActual, w[0], minAxis, maxAxis)

        # Keep track of the # of iterations
        numIterations.append(numIter)

    # Print overall summary
    print('\n\nAverage # iterations after {} runs: {}'.format(num_runs, np.mean(numIterations)))
    print('                Average P(f <> g): {}'.format(np.mean(probabilities)))

# ---------------- End ----------------
