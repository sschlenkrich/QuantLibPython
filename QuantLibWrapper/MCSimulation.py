#!/usr/bin/python

import numpy as np

class MCSimulation:

    # Python constructor
    def __init__(self, model, times, nPaths, seed=123):
        self.model  = model   # an object implementing stochastic process interface
        self.times  = times   # simulation times [0, ..., T], np.array
        self.nPaths = nPaths  # number of paths, long
        # random number generator
        self.dW = np.random.RandomState(seed).standard_normal([self.nPaths,len(self.times)-1,model.factors()])
        # simulate states
        self.X = np.zeros([self.nPaths,len(self.times),model.size()])
        for i in range(self.nPaths):
            self.X[i][0] = self.model.initialValues()
            for j in range(len(self.times)-1):
                self.X[i][j+1] = model.evolve(self.times[j],self.X[i][j],times[j+1]-times[j],self.dW[i][j])


