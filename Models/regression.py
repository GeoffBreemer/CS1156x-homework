'''Regression models'''
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- Functions ----------------
def plotPoints(x, f_slope, f_intercept, y, g, minAxis, maxAxis):
    '''Plot the random points, target function f and hypothesis g'''
    ax = plt.gca()
    plt.xlim((minAxis, maxAxis))
    plt.ylim((minAxis, maxAxis))

    # Plot the data points, use y (labels) to set the color
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

# ---------------- Classes ----------------

class LinearRegression:
    '''Linear Regression'''

    def __init__(self):
        pass

    def fit(self, X, y, reg=0):
        dotProd = np.dot(X.T, X)
        self.w = np.dot( np.linalg.inv( dotProd + reg * np.ones(dotProd.shape)) , np.dot(X.T, y))
        return self.w

    def predict(self, X):
        return np.sign(np.dot(X, self.w.T).T)


class Perceptron:
    '''Perceptron'''
    def __init__(self, num_iter, init_weights=None):
        self.num_iter = num_iter                                            # maximum # of iterations
        self.init_weights = init_weights                                    # initial weights

    def fit(self, X, y):
        # Run the Perceptron Learning Algorithm
        iterCount = 1
        if self.init_weights is None:
            self.w = np.zeros((1, X.shape[1]))
        else:
            self.w = self.init_weights

        for iter in np.arange(self.num_iter):
            hypothesis = np.sign(np.dot(X, self.w.T).T)                     # calculate the hypothesis: X * w.T
            missed = (y!=hypothesis)                                        # determine misclassified points

            if np.sum(missed) != 0:
                # Pick a random missclassified point
                idx = [i for i, ele in enumerate(missed[0]) if ele==True]   # indices of misclassified points
                tmpIndex = np.random.randint(0, len(idx))                   # pick one of the indices
                rndIndex = idx[tmpIndex]                                    # determine the index in X

                self.w += y[rndIndex] * X[rndIndex]                         # update the weights
                iterCount += 1
            else:
                # no misclassifications: done with this run
                break

        return iterCount

    def predict(self, X):
        return np.sign(np.dot(X, self.w.T).T)                                # evaluate the hypothesis g on x


class LogisticRegression:
    '''LogisticRegression with SGD'''
    def __init__(self, learning_rate, threshold):
        self.learning_rate = learning_rate                                  # maximum # of iterations
        self.threshold = threshold                                          # stopping condition threshold

    def shuffle(self, X, y):
        '''Permute X and y in the same way'''
        p = np.random.permutation(len(X))
        return X[p], y[p]

    def fit(self, X, y):
        # Run Stochastic Gradient Descent
        epochCount = 1
        self.w = np.zeros((1, X.shape[1]))

        # Stochastic Gradient Descent:
        while True:
            # Permute data for each epoch
            X, y = self.shuffle(X, y)
            w_prev = self.w

            # Update weights for each data point
            for i in range(len(X)):
                # Calculate the gradient
                grad = -((y[i] * (X[i].T))/(1.0 + np.exp((y[i] * self.w).dot(X[i]))))

                # Update weights
                self.w = self.w - self.learning_rate * grad

            # Stop if the Euclidean distance between weight updates is less than the threshold
            if (np.linalg.norm(self.w - w_prev) < self.threshold):
                break

            epochCount += 1

        return epochCount

    def predict(self, X):
        '''Return the predicted class: +1 or -1'''
        return np.sign(np.dot(X, self.w.T))

    def predict_score(self, X):
        '''Return the class score (a real number, does NOT use np.sign)'''
        return np.dot(X, self.w.T)

    def cross_entropy(self, X, yActual):
        '''Calculate the cross-entropy score'''
        return (1 / len(X)) * np.sum(np.log(1 + np.exp(-yActual*(self.predict_score(X).T))))
