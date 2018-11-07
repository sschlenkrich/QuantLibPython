#!/usr/bin/python

import numpy as np

from Regression import Regression

from Payoffs import SwapRate

class AMCSolver:

    # Python constructor
    def __init__(self, hwMcSimulation, maxPolynomialDegree=2, splitRatio=0.25):
        self.hwMcSimulation      = hwMcSimulation
        self.maxPolynomialDegree = maxPolynomialDegree
        self.minSampleIdx        = int(splitRatio*self.hwMcSimulation.nPaths)  # we split training data and simulation data

    def getIndexWithTolerance(self,t):
        return np.where(abs(self.hwMcSimulation.times-t)<1.0e-8)[0][0]

    def xSet(self,expiryTime):
        idx = self.getIndexWithTolerance(expiryTime)
        return np.array([ self.hwMcSimulation.X[i][idx] for i in range(self.hwMcSimulation.X.shape[0]) ])

    def rollBack(self, T0, T1, x1, U1, H1):        
        x0 = self.xSet(T0)
        N0 =  np.array([ self.hwMcSimulation.model.numeraire(x0[i]) for i in range(x1.shape[0]) ])
        N1 =  np.array([ self.hwMcSimulation.model.numeraire(x1[i]) for i in range(x1.shape[0]) ])
        if self.minSampleIdx>0 and T0>0:   # do not use regression for the last roll-back
            # we try state variable approach
            C = np.array([ [ x0[i][0] ] for i in range(self.minSampleIdx) ])
            O = np.array([ N0[i]/N1[i]*max(U1[i],H1[i]) for i in range(self.minSampleIdx) ])
            R = Regression(C,O,self.maxPolynomialDegree)
        else:
            R = None
        V0 = np.zeros(x1.shape[0])
        for i in range(x1.shape[0]):
            V0[i] = N0[i]/N1[i]*max(U1[i],H1[i])
            if R!=None: V0[i] = R.value(np.array([ x0[i][0] ]))
        if T0==0: 
            sampleIdx = self.minSampleIdx if self.minSampleIdx<self.hwMcSimulation.nPaths else 0
            return [ np.array([0.0]), np.array([np.sum(V0[sampleIdx:])/V0[sampleIdx:].shape[0] ]) ]
        return [x0, V0]


class AMCSolverOnlyExerciseRegression(AMCSolver):

    # Python constructor
    def __init__(self, hwMcSimulation, maxPolynomialDegree=2, splitRatio=0.25):
        AMCSolver.__init__(self,hwMcSimulation,maxPolynomialDegree,splitRatio)

    def rollBack(self, T0, T1, x1, U1, H1):        
        x0 = self.xSet(T0)
        N0 =  np.array([ self.hwMcSimulation.model.numeraire(x0[i]) for i in range(x1.shape[0]) ])
        N1 =  np.array([ self.hwMcSimulation.model.numeraire(x1[i]) for i in range(x1.shape[0]) ])
        if self.minSampleIdx>0 and T0>0:   # do not use regression for the last roll-back
            # we try state variable approach
            C = np.array([ [ x0[i][0] ]  for i in range(self.minSampleIdx) ])
            O = np.array([ U1[i] - H1[i] for i in range(self.minSampleIdx) ])
            R = Regression(C,O,self.maxPolynomialDegree)
        else:
            R = None
        V0 = np.zeros(x1.shape[0])
        for i in range(x1.shape[0]):
            I = U1[i] - H1[i]
            if R!=None: I = R.value(np.array([ x0[i][0] ]))
            V = U1[i] if I>0 else H1[i]
            V0[i] = N0[i] / N1[i] * V
        if T0==0: 
            sampleIdx = self.minSampleIdx if self.minSampleIdx<self.hwMcSimulation.nPaths else 0
            return [ np.array([0.0]), np.array([np.sum(V0[sampleIdx:])/V0[sampleIdx:].shape[0] ]) ]
        return [x0, V0]
