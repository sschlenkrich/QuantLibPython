#!/usr/bin/python

import numpy as np

class Pay:
    # Python constructor
    def __init__(self, payoff, payTime):
        self.observationTime = payoff.observationTime
        self.model   = payoff.model
        #
        self.payoff  = payoff
        self.payTime = payTime
    def at(self,x):
        return self.payoff.at(x)

class Zero:
    def at(self,x):
        return 0.0

class One:
    def at(self,x):
        return 1.0

class Max:
    # Python constructor
    def __init__(self, first, second):
        self.first  = first
        self.second = second

    def at(self,x):
        return max(self.first.at(x),self.second.at(x),)
    

class VanillaOption:
    # Python constructor
    def __init__(self, underlying, strike, callOrPut):
        self.observationTime = underlying.observationTime
        self.model      = underlying.model
        #
        self.underlying = underlying
        self.strike     = strike
        self.callOrPut  = callOrPut
    def at(self, x):
        return max(self.callOrPut*(self.underlying.at(x)-self.strike),0.0)

class CouponBond:
    # Python constructor
    def __init__(self, model, observationTime, payTimes, cashFlows):
        self.observationTime = observationTime
        self.model     = model
        # 
        self.payTimes  = payTimes
        self.cashFlows = cashFlows
    # function    
    def at(self, x):
        bond = 0
        for i in range(len(self.payTimes)):
            bond += self.cashFlows[i] * self.model.zeroBondPayoff(x,self.observationTime,self.payTimes[i])
        return bond

class SwapRate:
    # Python constructor
    def __init__(self, model, observationTime, startTime, endTime):
        self.observationTime = observationTime
        self.model     = model
        # 
        self.startTime = startTime
        self.endTime   = endTime
        #
        tmp = [startTime+k for k in range(int(endTime-startTime)+1)]
        if tmp[-1]<endTime : tmp = tmp + [endTime]
        self.annuityTimes = np.array(tmp)

    # function    
    def at(self, x):
        annuity = 0
        for i in range(1,self.annuityTimes.shape[0]):
            annuity += (self.annuityTimes[i]-self.annuityTimes[i-1]) * self.model.zeroBondPayoff(x,self.observationTime,self.annuityTimes[i])
        floatLeg = self.model.zeroBondPayoff(x,self.observationTime,self.startTime) -  \
                   self.model.zeroBondPayoff(x,self.observationTime,self.endTime)
        return floatLeg / annuity


