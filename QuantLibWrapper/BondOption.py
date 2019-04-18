#!/usr/bin/python

import QuantLib as ql

import numpy as np
from scipy.optimize import brentq
from scipy import integrate

from YieldCurve import YieldCurve
from SwapRate import SwapRate, SwapRateFromSwaption
from Swaption import createSwaption
from HullWhiteModel import HullWhiteModel
from DensityIntegrations import CubicSplineExactIntegration, DensityIntegrationWithBreakEven
from Payoffs import CouponBond
from BermudanOption import BermudanOption


# implement a European call option on a deterministic cash flow structure
class BondOption:

    # Python constructor
    def __init__(self, expiryTime, notionalTimes, notionalWeights, couponTimes, couponWeights, discYieldCurve, projYieldCurve, fundYieldCurve=None):
        self.expiryTime      = expiryTime
        self.notionalTimes   = notionalTimes
        self.notionalWeights = notionalWeights
        self.couponTimes     = couponTimes
        self.couponWeights   = couponWeights
        self.discYieldCurve  = discYieldCurve
        self.projYieldCurve  = projYieldCurve
        self.fundYieldCurve  = fundYieldCurve
        #
        floatweights = [ -cf for cf in notionalWeights ] # we want a payoff as Vanilla option
        # we need to adjust the bond cash flows for funding spreads
        floatweightsAdjusted  = [ weight*self.spreadDF(self.expiryTime,T) for weight,T in zip(floatweights,self.notionalTimes) ]
        couponWeightsAdjusted = [ weight*self.spreadDF(self.expiryTime,T) for weight,T in zip(self.couponWeights,self.couponTimes) ]
        self.swapRate = SwapRate(notionalTimes,floatweightsAdjusted,couponTimes,couponWeightsAdjusted,discYieldCurve)
        # we need to specify our reference swap rates
        swapTerm1 = round(max(notionalTimes) - expiryTime)  # we use full year swaptions as reference instrumnts
        assert swapTerm1 > 1.0,"Short maturity option not implemented."
        swapTerm2 = 1 # we use 1y swap rate as short term reference swap rate
        # first we set up swaptions
        swaption1 = createSwaption(str(round(expiryTime))+'y',str(swapTerm1)+'y',discYieldCurve,projYieldCurve)
        swaption2 = createSwaption(str(round(expiryTime))+'y',str(swapTerm2)+'y',discYieldCurve,projYieldCurve)
        # now we can also set up the SwapRates
        self.swapRate1 = SwapRateFromSwaption(swaption1)
        self.swapRate2 = SwapRateFromSwaption(swaption2)
        # we need to calibrate our zero bond model and use a nested Brent solver for this task
        S1 = self.swapRate1.forwardRate()
        S2 = self.swapRate2.forwardRate()
        def objectiveAlpha(alpha):
            def objectiveBeta(beta):
                return self.swapRate2.swapRate(0.0,alpha,beta) - S2
            beta = brentq(objectiveBeta,-10.0, 10.0, xtol=1.0e-8) # maybe we can use tighter bounds
            return self.swapRate1.swapRate(0.0,alpha,beta) - S1
        alpha = brentq(objectiveAlpha,-0.1, 0.1, xtol=1.0e-8)
        def objectiveBeta(beta): # we need to calculate beta again for fixed alpha because we don't get it from the earlier nested calibration
            return self.swapRate2.swapRate(0.0,alpha,beta) - S2
        beta = brentq(objectiveBeta,-1.0, 1.0, xtol=1.0e-8)
        # in the next step we can calculate the weights a and b using Cramer's rule
        y = self.swapRate.swapRateGradient(expiryTime,alpha,beta)
        A1 = self.swapRate1.swapRateGradient(expiryTime,alpha,beta) # first column
        A2 = self.swapRate2.swapRateGradient(expiryTime,alpha,beta) # second column
        detA = A1[0]*A2[1] - A1[1]*A2[0]
        self.a = (y[0]*A2[1] - y[1]*A2[0]) / detA
        self.b = (A1[0]*y[1] - A1[1]*y[0]) / detA

    # incorporate deterministic funding spreads into the payoff
    def spreadDF(self,t,T):
        if self.fundYieldCurve!=None:
            return (self.fundYieldCurve.discount(T)/self.discYieldCurve.discount(T)) / \
                   (self.fundYieldCurve.discount(t)/self.discYieldCurve.discount(t))
        return 1.0  # fall back

    def npv(self,volTS,rho):
        # now we can also set up the terminal distributions
        self.swapRate1.setUpTerminalDistribution(self.expiryTime,volTS)
        self.swapRate2.setUpTerminalDistribution(self.expiryTime,volTS)
        S0 = self.swapRate.forwardRate()
        # we also need the PDF of the 2-dim Normal distribution
        def pdf(n1,n2):
            return np.exp(-(n1*n1 + n2*n2 - 2.0*rho*n1*n2)/2.0/(1.0-rho*rho)) /  \
                   2.0 / np.pi / np.sqrt(1.0 - rho*rho)
        # we also need the payoff
        def payoff(n1,n2):
            return max( 1.0 - (S0 + self.a * self.swapRate1.swapRateFromNormal(n1) + \
                        self.b * self.swapRate2.swapRateFromNormal(n2)), 0.0 )
        # the integrand function is composed of payoff and pdf
        def integrand(n1,n2):
            return payoff(n1,n2) * pdf(n1,n2)
        # we calculate the double integral
        expectation = integrate.dblquad(integrand, -5, 5, lambda x: -5, lambda x: 5, epsabs=1.49e-05, epsrel=1.49e-05,) 
        #expectation = integrate.dblquad(integrand, -5, 5, lambda x: -5, lambda x: 5, epsabs=1.49e-03, epsrel=1.49e-03,) 
        # this should be improved by adjusting the inner integration points
        # funding spreads are incorporated separately
        return self.spreadDF(0,self.expiryTime) * self.swapRate.annuity() * expectation[0]

    def npvHullWhite(self,model):
        # we assume a model for the OIS rate and incorporate spreads manually
        payTimes  = self.notionalTimes + self.couponTimes
        cashFlows = self.notionalWeights + self.couponWeights
        cashFlowsAdjusted = [ weight*self.spreadDF(self.expiryTime,T) for weight,T in zip(cashFlows,payTimes) ]
        return self.spreadDF(0,self.expiryTime) * model.couponBondOption(self.expiryTime,payTimes,cashFlowsAdjusted,0.0,1.0)


class BermudanBondOption:

    # Python constructor
    def __init__(self, bondOptions, bondOptionNpvs, meanReversion=None, model=None):
        self.bondOptions = bondOptions  # a list of BondOptions
        self.bondOptionNpvs = bondOptionNpvs  # we provide npv's since we want to avoid exensive recomputation in calibration
        self.meanReversion = 0.03 if meanReversion==None else meanReversion
        if model!=None:
            self.model = model
        else:   # calibrate. this should not take long
            print('Calibrate HW Model: ',end='',flush=True)
            volatilityTimes  = np.array([ bondOption.expiryTime for bondOption in self.bondOptions ])
            volatilityValues = np.array([ 0.0050                for bondOption in self.bondOptions ])  # initial guess            
            for k in range(volatilityValues.shape[0]):
                print(str(k)+' ', end='', flush=True)
                def objective(sigma):
                    volatilityValues[k] = sigma
                    model = HullWhiteModel(self.bondOptions[0].discYieldCurve,self.meanReversion,volatilityTimes,volatilityValues)
                    return self.bondOptions[k].npvHullWhite(model) - self.bondOptionNpvs[k]
                sigma = brentq(objective,0.01*volatilityValues[k],5.0*volatilityValues[k],xtol=1.0e-8)
                volatilityValues[k] = sigma
            self.model = HullWhiteModel(self.bondOptions[0].discYieldCurve,self.meanReversion,volatilityTimes,volatilityValues)
            print('.')

    def npv(self):
        if self.bondOptions[0].fundYieldCurve!=None:  # we need to replace curves in model and setup a new method
            self.model.yieldCurve = self.bondOptions[0].fundYieldCurve
        self.method = DensityIntegrationWithBreakEven(CubicSplineExactIntegration(self.model,101,5))
        #self.method = DensityIntegrationWithBreakEven(CubicSplineExactIntegration(self.model,31,5))
        underlyings = [ CouponBond(self.model,bondOption.expiryTime,bondOption.notionalTimes+bondOption.couponTimes,bondOption.notionalWeights+bondOption.couponWeights)
                        for bondOption in self.bondOptions ]
        # we don't adjust cash flows in the CouponBond payoff because we incorporate funding spreads via discounting curve
        expiryTimes = np.array([ underlying.observationTime for underlying in underlyings ])
        bondOption = BermudanOption(expiryTimes,underlyings,self.method)
        return bondOption.npv()
    
