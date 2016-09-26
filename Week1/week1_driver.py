# Week 1, Perceptron, Q7-Q10
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- Functions ----------------
def plotPoints(x, f_slope, f_intercept, y, g):
    '''Plot the random points, target function f and hypothesis g'''
    ax = plt.gca()
    plt.xlim((minAxis, maxAxis))
    plt.ylim((minAxis, maxAxis))

    # Plot the data points, use y to set the color
    plt.scatter(x[:, 0], x[:, 1], c=y)

    # Plot target function f
    x_min, x_max = ax.get_xlim()
    y_min = f_slope * x_min + f_intercept
    y_max = f_slope * x_max + f_intercept
    plt.plot([x_min, x_max], [y_min, y_max], '--', label='target function f', c='g')

    # Plot hypothesis g
    plt.plot([x_min, x_max],
             [(-g[0] - g[1] * x_min) / g[2],
              (-g[0] - g[1] * x_max) / g[2]],
             '--k', label='hypothesis g')

    # Show the plot
    ax.set_aspect('equal')
    ax.grid(True, which='both')
    ax.legend(loc='best')
    plt.tight_layout()
    plt.title('Target function, hypothesis and observations')
    plt.show()


# ---------------- Constants ----------------
minAxis = -1  # grid size
maxAxis = 1
d = 2                       # number of dimensions
N = 10                      # number of training points
num_iter = 250              # maximum number of iterations per run
num_runs = 1000             # number of runs
numProb = 1000              # number of out of sample points used to predict P(f <> g)

#  q7-8: N =  10, num_runs = 1000
# q9-10: N = 100, num_runs = 1000

# ---------------- Start ----------------
numIterations = []
probabilities = []

# Execute Perceptron algorithm
for run in np.arange(num_runs):
    # Initialise run specific variables
    numIter = 1

    xOrg = np.random.uniform(minAxis, maxAxis, size=(N, d))         # create input vector x
    fPts = np.random.uniform(minAxis, maxAxis, size=(2,2))          # find two random points
    f_slope, f_intercept = np.polyfit(fPts[:, 0], fPts[:, 1], 1)    # and use them to create target function f
    y = np.sign(xOrg[:,0] * f_slope - xOrg[:, 1] + f_intercept)     # determine y (= f evaluated on x)
    x = np.hstack((np.ones((N, 1)), xOrg))                          # add x0 to x
    w = np.zeros((1, d + 1))  # reset weights vector to zero

    # Run the Perceptron Learning Algorithm
    for iter in np.arange(num_iter):
        hypothesis = np.sign(np.dot(x, w.T).T)              # calculate the hypothesis
        missed = (y!=hypothesis)                            # determine misclassified points

        if np.sum(missed) != 0:
            # Pick a random missclassified point
            idx = [i for i, ele in enumerate(missed[0]) if ele==True]   # indices of misclassified points
            tmpIndex = np.random.randint(0, len(idx))                   # pick one of the indices
            rndIndex = idx[tmpIndex]                                    # determine the index in x

            w = w + y[rndIndex] * x[rndIndex]                           # update the weights
            numIter += 1
        else:
            # no misclassifications: done
            break

    # Calculate P(f <> g) for a large number of out of sample points
    tmpXOrg = np.random.uniform(minAxis, maxAxis, size=(numProb, d))          # create input vector
    yActual = np.sign(tmpXOrg[:,0] * f_slope - tmpXOrg[:, 1] + f_intercept)   # determine y (= f evaluated on x)
    tmpX = np.hstack((np.ones((numProb, 1)), tmpXOrg))                  # add x0 to x
    yPred = np.sign(np.dot(tmpX, w.T).T)                                # evaluate the hypothesis g on x
    missed = np.sum((yPred != yActual))                                 # number of misclassified points

    # Keep track of the P(f <> g)
    probabilities.append(missed / numProb)

    # Print run summary
    print('Run {:4d}: # iterations: {:3d}, misclassified: {:3f}'.format(run, numIter, missed/numProb))

    # Training plot
    # plotPoints(xOrg, f_slope, f_intercept, y, w[0])

    # Test plot
    # plotPoints(tmpXOrg, f_slope, f_intercept, yActual, w[0])

    # Keep track of the # of iterations
    numIterations.append(numIter)

# Print overall summary
print('\n\nAverage # iterations after {} runs: {}'.format(num_runs, np.mean(numIterations)))
print('                Average P(f <> g): {}'.format(np.mean(probabilities)))

# ---------------- End ----------------

