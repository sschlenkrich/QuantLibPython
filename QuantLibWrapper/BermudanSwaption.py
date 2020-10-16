#!/usr/bin/python

import numpy as np
from scipy.optimize import brentq

from QuantLibWrapper.DensityIntegrations import CubicSplineExactIntegration, DensityIntegrationWithBreakEven, SimpsonIntegration
from QuantLibWrapper.PDESolver import PDESolver
from QuantLibWrapper.Helpers import BachelierImpliedVol
from QuantLibWrapper.HullWhiteModel import HullWhiteModel
from QuantLibWrapper.Payoffs import CouponBond
from QuantLibWrapper.BermudanOption import BermudanOption

class BermudanSwaption:

    # Python constructor
    def __init__(self, europeanSwaptions, meanReversion=None, model=None, method=None):
        self.europeanSwaptions = europeanSwaptions   # a list of Swaption objects
        self.method = method                         # a pricing method to bo passed to BermudanOption
        self.meanReversion = 0.03 if meanReversion==None else meanReversion
        if model!=None:
            self.model = model
        else:   # calibrate. this should not take long
            print('Calibrate HW Model: ',end='',flush=True)
            volatilityTimes  = np.array( [swaption.bondOptionDetails()['expiryTime'] for swaption in self.europeanSwaptions] )
            volatilityValues = np.array( [self.europeanSwaptions[0].normalVolatility for i in range(volatilityTimes.shape[0]) ])  # initial guess            
            for k in range(volatilityValues.shape[0]):
                print(str(k)+' ', end='', flush=True)
                def objective(sigma):
                    volatilityValues[k] = sigma
                    model = HullWhiteModel(self.europeanSwaptions[0].underlyingSwap.discYieldCurve,self.meanReversion,volatilityTimes,volatilityValues)
                    return self.europeanSwaptions[k].npvHullWhite(model,'p') - self.europeanSwaptions[k].npv()
                sigma = brentq(objective,0.1*volatilityValues[k],5.0*volatilityValues[k],xtol=1.0e-8)
                volatilityValues[k] = sigma
            self.model = HullWhiteModel(self.europeanSwaptions[0].underlyingSwap.discYieldCurve,self.meanReversion,volatilityTimes,volatilityValues)
            print('.')
        if method!=None:
            self.method = method
        else:
            #self.method = PDESolver(self.model,101,3.0,0.5,1.0/12.0)
            self.method = DensityIntegrationWithBreakEven(CubicSplineExactIntegration(self.model,101,5))
        # Done. We postpone pricing to npv calculation

    def swaptionsNPV(self):
        return np.array([ swaption.npv() for swaption in self.europeanSwaptions ])

    def bondOptionsNPV(self):
        return np.array([ swaption.npvHullWhite(self.model) for swaption in self.europeanSwaptions ])

    def npv(self):
        underlyings = []
        for swaption in self.europeanSwaptions:
            details = swaption.bondOptionDetails()
            underlying = CouponBond(self.model,details['expiryTime'],details['payTimes'],details['cashFlows']*details['callOrPut'])
            underlyings.append(underlying)
        expiryTimes = np.array([ underlying.observationTime for underlying in underlyings ])
        bondOption = BermudanOption(expiryTimes,underlyings,self.method)
        return bondOption.npv()
    