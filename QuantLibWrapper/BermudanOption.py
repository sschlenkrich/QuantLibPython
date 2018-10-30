#!/usr/bin/python

import numpy as np

class BermudanOption:

    # Python constructor
    def __init__(self, expiryTimes, underlyings, method):
        print('Bermudan option pricing: |',end='', flush=True)
        self.expiryTimes = expiryTimes # T_E^1, ..., T_E^k
        self.underlyings = underlyings # U_k(x)
        self.method      = method      # the numerical method used for roll-back
        #
        x = np.zeros(0)
        V = np.zeros(0)
        for k in range(expiryTimes.shape[0],0,-1):
            print(".",end='', flush=True)
            if k==expiryTimes.shape[0]:
                x = method.xSet(expiryTimes[k-1])
                H = np.zeros(x.shape[0])
            else:
                [x, H] = method.rollBack(expiryTimes[k-1],expiryTimes[k],x,V)
            U = np.array([ underlyings[k-1].at([state,0.0]) for state in x ])
            V = np.array([ max(U[j],H[j]) for j in range(U.shape[0]) ])
        [x, H] = method.rollBack(0.0,expiryTimes[0],x,V)
        self.x = x
        self.H = H
        print('| Done.', flush=True)

    def npv(self):
        if self.H.shape[0]==1:
            return self.H[0]
        return np.interp(0.0, self.x, self.H)


