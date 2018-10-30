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

