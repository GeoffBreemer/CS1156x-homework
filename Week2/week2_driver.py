import matplotlib.pyplot as plt
import numpy as np
from Models import regression as reg

# ---------------- Functions ----------------

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
    for _ in range(num_runs):
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


def q5q6andq7():
    # ---------------- Constants ----------------
    minAxis = -1            # grid size
    maxAxis = 1
    d = 2                   # number of dimensions
    N = 10                  # number of training points   q5/6: N = 100, q7: N = 10
    num_runs = 1000         # number of runs
    numProb = 1000          # number of out-of-sample observations to generate

    numMissedInSample = []
    numMissedOutOfSample = []
    # For q7 only:
    numPLAIterations = []

    for _ in np.arange(num_runs):
        # Create random observations
        xOrg = np.random.uniform(minAxis, maxAxis, size=(N, d)) # create input vector x
        fPts = np.random.uniform(minAxis, maxAxis, size=(2,2))  # find two random points
        f_slope, f_intercept = np.polyfit(fPts[:, 0], fPts[:, 1], 1)  # and use them to create target function f
        y = np.sign(xOrg[:,0] * f_slope - xOrg[:, 1] + f_intercept)         # determine y (= f evaluated on x)
        x = np.hstack((np.ones((N, 1)), xOrg))                     # add x0 to x

        # Calculate Linear Regression weights
        linreg = reg.LinearRegression()
        w = linreg.fit(x, y)

        # For q7 only: run Perceptron with these weights as initial weights
        # per = reg.Perceptron(250, w.reshape((1, d+1)))
        # numPLAIterations.append(per.fit(x, y))
        # w = per.w

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
        reg.plotPoints(xOrg, f_slope, f_intercept, y, w, minAxis, maxAxis)

    # Print summary
    print('    Mean # of in-sample misclassifications: {}'.format(np.mean(numMissedInSample)))
    print('Mean # of out-of-sample misclassifications: {}'.format(np.mean(numMissedOutOfSample)))
    # For q7 only:
    # print('                  Number of PLA iterations: {}'.format(np.mean(numPLAIterations)))


def q8():
    # ---------------- Constants ----------------
    minAxis = -1        # grid size
    maxAxis = 1
    d = 2               # number of dimensions
    N = 1000            # number of training points
    num_runs = 1000     # number of runs
    numProb = 100       # number of out-of-sample observations to generate

    numMissedInSample = []
    numMissedOutOfSample = []

    for _ in np.arange(num_runs):
        # Create random observations
        xOrg = np.random.uniform(minAxis, maxAxis, size=(N, d))             # create input vector x
        y = np.sign(np.square(xOrg[:,0]) + np.square(xOrg[:,1]) - 0.6 )     # determine y (= f evaluated on x)
        x = np.hstack((np.ones((N, 1)), xOrg))                              # add x0 to x

        # Add noise
        tmpIndex = np.random.randint(0, len(x), 100)
        y[tmpIndex] = -y[tmpIndex]

        # Calculate weights
        w = np.dot(np.linalg.inv(np.dot(x.T, x)) , (np.dot(x.T, y)))

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

    # Print summary
    print('    Mean # of in-sample misclassifications: {}'.format(np.mean(numMissedInSample)))
    print('Mean # of out-of-sample misclassifications: {}'.format(np.mean(numMissedOutOfSample)))


def q9q10():
    # ---------------- Constants ----------------
    minAxis = -1            # grid size
    maxAxis = 1
    d = 2                   # number of dimensions
    N = 1000                # number of training points
    num_runs = 50           # number of runs
    num_oos_runs = 1000     # number of out-of-sample observations

    numMissedInSample = []
    weights = []

    for _ in np.arange(num_runs):
        # Create random observations
        xOrg = np.random.uniform(minAxis, maxAxis, size=(N, d))             # create input vector x
        y = np.sign(np.square(xOrg[:,0]) + np.square(xOrg[:,1]) - 0.6 )     # determine y (= f evaluated on x)
        x = np.hstack( (np.ones((N, 1)),                                    # feature vector with transformations
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

    # Determine out of sample error using the *average* weights found above
    w = np.mean(weights, axis=0)
    numMissedOutOfSample = []
    for _ in range(num_oos_runs):
        # Create new random out-of-sample observations
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

if __name__ == "__main__":
    # q1andq2()
    # q5q6andq7()
    q8()
    # q9q10()

# ---------------- End ----------------
