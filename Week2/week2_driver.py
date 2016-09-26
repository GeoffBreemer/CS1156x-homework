import numpy as np
import matplotlib.pyplot as plt

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


def q1andq2():
    num_coins = 1000
    num_flips = 10
    num_runs = 10000

    # Init variables
    coins = np.zeros((num_coins, num_flips))
    v1 = []
    vrand = []
    vmin = []

    # Perform experiemtn num_runs times
    for run in range(num_runs):
        # Flip num_coins num_flips times
        coins = np.random.randint(0, 2, (num_coins, num_flips))

        # Count the number of heads for each coin
        numHeads = np.array(np.sum(coins, axis=1)).reshape(num_coins)

        # Calculate the proportion of heads for three coins
        v1.append(numHeads[0]/num_flips)
        vrand.append(numHeads[np.random.randint(0, num_coins)]/num_flips)
        vmin.append(numHeads[np.argmin(numHeads)]/num_flips)

    # Print means
    print('   Mean(v1): {}'.format(np.mean(v1)))
    print('Mean(vrand): {}'.format(np.mean(vrand)))
    print(' Mean(vmin): {}'.format(np.mean(vmin)))

    # Plot distributions of v1, vrand and vmin
    fig, ax = plt.subplots(sharex=True, sharey=True)
    plt.subplot(3,1,1)
    plt.hist(v1)
    plt.subplot(3,1,2)
    plt.hist(vrand)
    plt.subplot(3,1,3)
    plt.hist(vmin)
    plt.show()


def runPerceptron(w, x, y):
    '''
    Run without any parameters for Week 1
    Run *with* parameters for Week 2
    '''
    # ---------------- Constants ----------------
    num_iter = 250              # maximum number of iterations per run
    num_runs = 1000             # number of runs

    # ---------------- Start ----------------
    numIterations = []
    probabilities = []

    # Execute Perceptron algorithm
    for run in np.arange(num_runs):
        # Initialise run specific variables
        numIter = 1

        # Run the Perceptron Learning Algorithm
        for iter in np.arange(num_iter):
            hypothesis = np.sign(np.dot(x, w.T).T)              # calculate the hypothesis
            missed = (y!=hypothesis)                            # determine misclassified points

            if np.sum(missed) != 0:
                # Pick a random missclassified point
                idx = [i for i, ele in enumerate(missed[0]) if ele==True]   # indices of misclassified points
                tmpIndex = np.random.randint(0, len(idx))       # pick one of the indices
                rndIndex = idx[tmpIndex]                        # determine the index in x

                w = w + y[rndIndex] * x[rndIndex]               # update the weights
                numIter += 1
            else:
                # no misclassifications: done
                break

        # Keep track of the # of iterations
        numIterations.append(numIter)

    # Print overall summary
    # print('\n\nAverage # iterations after {} runs: {}'.format(num_runs, np.mean(numIterations)))
    # print('                Average P(f <> g): {}'.format(np.mean(probabilities)))

    return np.mean(numIterations)
    # ---------------- End ----------------


def q5q6andq7():
    # ---------------- Constants ----------------
    minAxis = -1  # grid size
    maxAxis = 1
    d = 2  # number of dimensions
    N = 100  # number of training points
    num_runs = 1000  # number of runs
    numProb = 100

    numMissedInSample = []
    numMissedOutOfSample = []
    weights = []
    numPLAIterations = []

    for run in np.arange(num_runs):
        xOrg = np.random.uniform(minAxis, maxAxis, size=(N, d)) # create input vector x
        fPts = np.random.uniform(minAxis, maxAxis, size=(2,2))  # find two random points
        f_slope, f_intercept = np.polyfit(fPts[:, 0], fPts[:, 1], 1)  # and use them to create target function f
        y = np.sign(xOrg[:,0] * f_slope - xOrg[:, 1] + f_intercept)         # determine y (= f evaluated on x)
        x = np.hstack((np.ones((N, 1)), xOrg))                     # add x0 to x

        # Calculate weights
        w = np.dot(np.linalg.inv(np.dot(x.T, x)) , (np.dot(x.T, y)))
        weights.append(w)

        # For q7 only: run Perceptron with these weights
        numPLAIterations.append(runPerceptron(w.reshape((1, d+1)), x, y))

        # Calculate in-sample misclassifications
        yPred = np.sign(np.dot(x, w.T).T)   # evaluate the hypothesis g on x
        missed = np.sum((yPred != y))       # number of misclassified points

        numMissedInSample.append(missed/N)

        # Calculate out-of-sample misclassifications
        tmpXOrg = np.random.uniform(minAxis, maxAxis, size=(numProb, d))  # create input vector
        yActual = np.sign(tmpXOrg[:, 0] * f_slope - tmpXOrg[:, 1] + f_intercept)  # determine y (= f evaluated on x)
        tmpX = np.hstack((np.ones((numProb, 1)), tmpXOrg))  # add x0 to x
        yPred = np.sign(np.dot(tmpX, w.T).T)  # evaluate the hypothesis g on x
        missed = np.sum((yPred != yActual))  # number of misclassified points

        numMissedOutOfSample.append(missed/numProb)

        # print('Run {:4d}, misclassified: {:3f}, PLA iter: {}'.format(run, missed / N, np.mean(numPLAIterations)))

        # Training plot
        # plotPoints(xOrg, f_slope, f_intercept, y, w)

    # Print summary
    print('    Mean # of in-sample misclassifications: {}'.format(np.mean(numMissedInSample)))
    print('Mean # of out-of-sample misclassifications: {}'.format(np.mean(numMissedOutOfSample)))
    print('                  Number of PLA iterations: {}'.format(np.mean(numPLAIterations)))


def q8():
    # ---------------- Constants ----------------
    minAxis = -1  # grid size
    maxAxis = 1
    d = 2  # number of dimensions
    N = 1000  # number of training points
    num_runs = 1000  # number of runs
    numProb = 100

    numMissedInSample = []
    numMissedOutOfSample = []
    weights = []

    for run in np.arange(num_runs):
        xOrg = np.random.uniform(minAxis, maxAxis, size=(N, d))             # create input vector x
        y = np.sign(np.square(xOrg[:,0]) + np.square(xOrg[:,1]) - 0.6 )     # determine y (= f evaluated on x)
        x = np.hstack((np.ones((N, 1)), xOrg))                              # add x0 to x

        # Add noise
        tmpIndex = np.random.randint(0, len(x), 100)
        y[tmpIndex] = -y[tmpIndex]

        # Calculate weights
        w = np.dot(np.linalg.inv(np.dot(x.T, x)) , (np.dot(x.T, y)))
        weights.append(w)

        # Calculate in-sample misclassifications
        yPred = np.sign(np.dot(x, w.T).T)   # evaluate the hypothesis g on x
        missed = np.sum((yPred != y))       # number of misclassified points

        numMissedInSample.append(missed/N)

        # Calculate out-of-sample misclassifications
        tmpXOrg = np.random.uniform(minAxis, maxAxis, size=(numProb, d))  # create input vector
        yActual = np.sign(np.square(tmpXOrg[:, 0]) + np.square(tmpXOrg[:, 1]) - 0.6)  # determine y (= f evaluated on x)
        tmpX = np.hstack((np.ones((numProb, 1)), tmpXOrg))  # add x0 to x
        yPred = np.sign(np.dot(tmpX, w.T).T)  # evaluate the hypothesis g on x
        missed = np.sum((yPred != yActual))  # number of misclassified points

        numMissedOutOfSample.append(missed/numProb)

        # print('Run {:4d}, misclassified: {:3f}, PLA iter: {}'.format(run, missed / N, np.mean(numPLAIterations)))

        # Training plot
        # plotPoints(xOrg, f_slope, f_intercept, y, w)

    # Print summary
    print('    Mean # of in-sample misclassifications: {}'.format(np.mean(numMissedInSample)))
    print('Mean # of out-of-sample misclassifications: {}'.format(np.mean(numMissedOutOfSample)))


def q9q10():
    # ---------------- Constants ----------------
    minAxis = -1  # grid size
    maxAxis = 1
    d = 2  # number of dimensions
    N = 1000  # number of training points
    num_runs = 50  # number of runs
    num_oos_runs = 1000
    numMissedInSample = []
    weights = []

    for run in np.arange(num_runs):
        xOrg = np.random.uniform(minAxis, maxAxis, size=(N, d))             # create input vector x
        y = np.sign(np.square(xOrg[:,0]) + np.square(xOrg[:,1]) - 0.6 )     # determine y (= f evaluated on x)
        x = np.hstack( (np.ones((N, 1)),
                       xOrg[:, 0].reshape(len(xOrg), 1),
                       xOrg[:, 1].reshape(len(xOrg), 1),
                       (xOrg[:, 0] * xOrg[:,1]).reshape(len(xOrg), 1),
                       np.square(xOrg[:,0]).reshape(len(xOrg), 1),
                       np.square(xOrg[:,1]).reshape(len(xOrg), 1)
                       ))

        # Add noise for 100 observations
        tmpIndex = np.random.randint(0, len(x), 100)
        y[tmpIndex] = -y[tmpIndex]

        # Calculate weights
        w = np.dot(np.linalg.inv(np.dot(x.T, x)) , (np.dot(x.T, y)))
        weights.append(w)

        # Calculate in-sample misclassifications
        yPred = np.sign(np.dot(x, w.T).T)   # evaluate the hypothesis g on x
        missed = np.sum((yPred != y))       # number of misclassified points

        numMissedInSample.append(missed/N)

    # Print summary
    print('    Mean # of in-sample misclassifications: {}'.format(np.mean(numMissedInSample)))
    print('                              mean weights:\n{}'.format(np.mean(weights, axis=0)))

    # Determine out of sample error
    w = np.mean(weights, axis=0)
    numMissedOutOfSample = []
    for run in range(num_oos_runs):
        xOrg = np.random.uniform(minAxis, maxAxis, size=(N, d))             # create input vector x
        y = np.sign(np.square(xOrg[:,0]) + np.square(xOrg[:,1]) - 0.6 )     # determine y (= f evaluated on x)
        x = np.hstack( (np.ones((N, 1)),
                       xOrg[:, 0].reshape(len(xOrg), 1),
                       xOrg[:, 1].reshape(len(xOrg), 1),
                       (xOrg[:, 0] * xOrg[:,1]).reshape(len(xOrg), 1),
                       np.square(xOrg[:,0]).reshape(len(xOrg), 1),
                       np.square(xOrg[:,1]).reshape(len(xOrg), 1)
                       ))

        # Add noise for 100 observations
        tmpIndex = np.random.randint(0, len(x), 100)
        y[tmpIndex] = -y[tmpIndex]

        # Make predictions and calculate proportion of misclassifications
        yPred = np.sign(np.dot(x, w.T).T)   # evaluate the hypothesis g on x
        missed = np.sum((yPred != y))       # number of misclassified points

        numMissedOutOfSample.append(missed/num_oos_runs)

    print('Mean # of out-of-sample misclassifications: {}'.format(np.mean(numMissedOutOfSample)))


# ---------------- Start ----------------

# q1andq2()
# q5q6andq7()
# q8()
q9q10()

# ---------------- End ----------------
