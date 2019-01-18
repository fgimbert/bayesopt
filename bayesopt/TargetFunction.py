import numpy as np


class TargetFunction:

    def __init__(self, dimension=2, test=True):

        self.dim = dimension
        self.batch = True
        self.test = test

        if test:
            self.target_f = self.test_function
        else:
            self.target_f = self.custom_function

    def test_function(self, x=[0,0], batch_i=0):

        if self.dim == 1:
            target = (np.sin(10 * np.pi * x[0]) / (2 * x[0])) + (np.power((x[0] - 1),4))
        else:
            target = 1
            for i in range(self.dim):
                target = target * np.sqrt(x[i]) * np.sin(x[i])

        if self.batch:
            return batch_i, -1 * target
        else:
            return -1 * target

    def custom_function(self, x=[0,0], batch_i=0):

        if self.dim == 1:
            target = (np.sin(10 * np.pi * x[0]) / (2 * x[0])) + (np.power((x[0] - 1), 4))
        else:
            target = 1
            for i in range(self.dim):
                target = target * np.sqrt(x[i]) * np.sin(x[i])

        if self.batch:
            return batch_i, -1 * target
        else:
            return -1 * target

    def constraints(self, x=[0,0]):
        if self.test and self.dim == 2:
            return np.abs(x[0] - x[1])
        else:
            return np.sum(x) - 1
