#!/usr/bin/python

import numpy as np

class MCSimulation:

    # Python constructor
    def __init__(self, model, times, nPaths, seed=123):
        print('Start MC Simulation:', end='', flush=True)
        self.model  = model   # an object implementing stochastic process interface
        self.times  = times   # simulation times [0, ..., T], np.array
        self.nPaths = nPaths  # number of paths, long
        # random number generator
        print(' |dW\'s', end='', flush=True)
        self.dW = np.random.RandomState(seed).standard_normal([self.nPaths,len(self.times)-1,model.factors()])
        print('|', end='', flush=True)
        # simulate states
        self.X = np.zeros([self.nPaths,len(self.times),model.size()])
        for i in range(self.nPaths):
            if i % int(self.nPaths/10) == 0 : print('s', end='', flush=True)
            self.X[i][0] = self.model.initialValues()
            for j in range(len(self.times)-1):
                self.X[i][j+1] = model.evolve(self.times[j],self.X[i][j],times[j+1]-times[j],self.dW[i][j])
        print('| Finished.', end='\n', flush=True)

    def npv(self, payoff):
        print('Calculate payoff...', end='', flush=True)
        obsIdx = np.where(self.times==payoff.observationTime)[0][0]  # assume we simulated these dates
        payIdx = np.where(self.times==payoff.payTime)[0][0]          # otherwise we get an exception
        V0 = np.zeros([self.nPaths])   # simulated discounted payoffs
        VT = np.zeros([self.nPaths])   # simulated payoff at observation time
        N0 = np.zeros([self.nPaths])   # numeraire at 0; should be 1
        NT = np.zeros([self.nPaths])   # simulated numeraire at pay time
        for i in range(self.nPaths):
            VT[i] = payoff.at(self.X[i][obsIdx])
            N0[i] = payoff.model.numeraire(self.X[i][0])
            NT[i] = payoff.model.numeraire(self.X[i][payIdx])
            V0[i] = N0[i] * VT[i] / NT[i]
        print(' Done.', end='\n', flush=True)
        return np.mean(V0)
