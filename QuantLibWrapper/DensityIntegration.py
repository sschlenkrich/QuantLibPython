#!/usr/bin/python

import numpy as np
from scipy.stats import norm
from scipy import integrate

class DensityIntegration:

    # Python constructor
    def __init__(self, hwModel, nGridPoints=101, stdDevs=5):
        self.hwModel     = hwModel
        self.nGridPoints = nGridPoints
        self.stdDevs     = stdDevs
    
    def xSet(self,expityTime):
        sigma = np.sqrt(self.hwModel.varianceX(0.0,expityTime))
        if sigma==0:
            return np.array([0.0])
        return np.linspace(-self.stdDevs*sigma,self.stdDevs*sigma,self.nGridPoints)

    def rollBack(self, T0, T1, x1, V1):
        x0 = self.xSet(T0)
        V0 = np.zeros(x0.shape[0])
        sigma = np.sqrt(self.hwModel.varianceX(T0,T1))
        for i in range(x0.shape[0]):
            nu = self.hwModel.expectationX(T0, x0[i], T1)
            fx = np.array([ V1[k] * norm.pdf((x1[k]-nu)/sigma)/sigma for k in range(x1.shape[0])])
            I = integrate.simps(fx, x1)
            V0[i] = self.hwModel.zeroBond(T0,x0[i],T1) * I
        return [x0, V0]

