import numpy as np

class Perceptron:
    '''Perceptron implementation'''
    def __init__(self, num_iter, init_weights=None):
        self.num_iter = num_iter                                            # maximum # of iterations
        self.init_weights = init_weights                                    # initial weights

    def fit(self, x, y):
        # Run the Perceptron Learning Algorithm
        iterCount = 1
        if self.init_weights is None:
            self.w = np.zeros((1, x.shape[1]))
        else:
            self.w = self.init_weights

        for iter in np.arange(self.num_iter):
            hypothesis = np.sign(np.dot(x, self.w.T).T)                     # calculate the hypothesis: x * w.T
            missed = (y!=hypothesis)                                        # determine misclassified points

            if np.sum(missed) != 0:
                # Pick a random missclassified point
                idx = [i for i, ele in enumerate(missed[0]) if ele==True]   # indices of misclassified points
                tmpIndex = np.random.randint(0, len(idx))                   # pick one of the indices
                rndIndex = idx[tmpIndex]                                    # determine the index in x

                self.w += y[rndIndex] * x[rndIndex]                         # update the weights
                iterCount += 1
            else:
                # no misclassifications: done with this run
                break

        return iterCount

    def predict(self, x):
        return np.sign(np.dot(x, self.w.T).T)                                # evaluate the hypothesis g on x
