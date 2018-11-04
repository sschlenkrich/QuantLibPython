#!/usr/bin/python

import numpy as np

from Regression import Regression

class AMCSolver:

    # Python constructor
    def __init__(self, hwMcSimulation, finalMaturity, maxPolynomialDegree=2, splitRatio=0.25):
        self.hwMcSimulation      = hwMcSimulation
        self.finalMaturity       = finalMaturity
        self.maxPolynomialDegree = maxPolynomialDegree
        self.minSampleIdx        = int(splitRatio*self.hwMcSimulation.nPaths)  # we split training data and simulation data

    def getIndexWithTolerance(self,t):
        return np.where(abs(self.hwMcSimulation.times-t)<1.0e-8)[0][0]

    def xSet(self,expiryTime):
        idx = self.getIndexWithTolerance(expiryTime)
        return np.array([ self.hwMcSimulation.X[i][idx][0] for i in range(self.hwMcSimulation.X.shape[0]) ])

    def rollBack(self, T0, T1, x1, U1, H1):        
        T0idx = self.getIndexWithTolerance(T0)  # assume we simulated these dates
        T1idx = self.getIndexWithTolerance(T1)  # assume we simulated these dates
        if self.minSampleIdx>0:
            # we try state variable approach
            C = np.array([ [c] for c in x1[:self.minSampleIdx] ])
            R = Regression(C,U1[:self.minSampleIdx]-H1[:self.minSampleIdx],self.maxPolynomialDegree)
        else:
            R = None
        V0 = np.zeros(x1.shape[0])
        for i in range(x1.shape[0]):
            N0 =  self.hwMcSimulation.model.numeraire(self.hwMcSimulation.X[i][T0idx])
            N1 =  self.hwMcSimulation.model.numeraire(self.hwMcSimulation.X[i][T1idx])
            I  =  (U1[i] - H1[i])  # as a fall-back we look into the future
            if R!=None: I  =  R.value(np.array([ x1[i] ]))  # use regression for decision only
            V  =  U1[i] if I>0 else H1[i]
            V0[i] = N0 * V / N1
        if T0==0: 
            sampleIdx = self.minSampleIdx if self.minSampleIdx<self.hwMcSimulation.nPaths else 0
            return [ np.array([0.0]), np.array([np.sum(V0[sampleIdx:])/V0[sampleIdx:].shape[0] ]) ]
        return [self.xSet(T0), V0]
