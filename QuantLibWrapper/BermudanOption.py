#!/usr/bin/python

import numpy as np

class BermudanOption:

    # Python constructor
    def __init__(self, expiryTimes, underlyings, method):
        print('Bermudan option pricing: |',end='', flush=True)
        self.expiryTimes = expiryTimes # T_E^1, ..., T_E^k
        self.underlyings = underlyings # U_k(x)
        self.method      = method      # the numerical method used for roll-back
        for k in range(expiryTimes.shape[0],0,-1):
            print(".",end='', flush=True)
            if k==expiryTimes.shape[0]:
                x = method.xSet(expiryTimes[k-1])
                H = np.zeros(x.shape[0])
            else:
                [x, H] = method.rollBack(expiryTimes[k-1],expiryTimes[k],x,U,H)
            if len(x.shape)==1:  # PDE and density integration
                U = np.array([ underlyings[k-1].at([state,0.0]) for state in x ])
            else:   # MC simulation
                U = np.array([ underlyings[k-1].at(state) for state in x ])
        [x, H] = method.rollBack(0.0,expiryTimes[0],x,U,H)
        self.x = x
        self.H = H
        print('| Done.', flush=True)

    def npv(self):
        if self.H.shape[0]==1:
            return self.H[0]
        return np.interp(0.0, self.x, self.H)


# We specify a European payoff (pricer) without [.]^+ operator
class EuropeanPayoff(BermudanOption):

    # Python constructor
    def __init__(self, expiryTime, underlying, method):
        x = method.xSet(expiryTime)
        if len(x.shape)==1:  # PDE and density integration
            U = np.array([ underlying.at([state,0.0]) for state in x ])
        else:   # MC simulation
            U = np.array([ underlying.at(state) for state in x ])
        [x, H] = method.rollBack(0.0,expiryTime,x,U,U)
        self.x = x
        self.H = H