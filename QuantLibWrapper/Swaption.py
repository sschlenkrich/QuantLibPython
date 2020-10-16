#!/usr/bin/python

import pandas
import QuantLib as ql

import numpy as np
from scipy.optimize import brentq

from QuantLibWrapper.Helpers import Bachelier, BachelierImpliedVol, BachelierVega
from QuantLibWrapper.Swap import Swap
from QuantLibWrapper.Payoffs import CouponBond
from QuantLibWrapper.HullWhiteModel import HullWhiteModel

class Swaption:

    # Python constructor
    def __init__(self, underlyingSwap, expiryDate, normalVolatility):
        self.underlyingSwap = underlyingSwap
        self.exercise = ql.EuropeanExercise(expiryDate)
        self.swaption = ql.Swaption(self.underlyingSwap.swap,self.exercise,ql.Settlement.Physical)
        self.normalVolatility = normalVolatility
        volQuote = ql.SimpleQuote(normalVolatility)
        volHandle = ql.QuoteHandle(volQuote)
        initialEngine = ql.BachelierSwaptionEngine(self.underlyingSwap.discHandle,volHandle,ql.Actual365Fixed())
        self.swaption.setPricingEngine(initialEngine)

    def npv(self):
        return self.swaption.NPV()

    def fairRate(self):
        return self.underlyingSwap.fairRate()

    def annuity(self):
        return self.underlyingSwap.annuity()
    
    def npvRaw(self):
        # calculate npv manually using Bachelier formula
        # use this to cross-check npv calculation via QuantLib engine
        refDate  = self.underlyingSwap.discHandle.referenceDate()
        T = ql.Actual365Fixed().yearFraction(refDate,self.exercise.dates()[0])
        CallOrPutOnS = 1.0 if self.underlyingSwap.payerOrReceiver==ql.VanillaSwap.Payer else -1.0
        return self.annuity() * Bachelier(self.underlyingSwap.fixedRate,self.fairRate(),self.normalVolatility,T,CallOrPutOnS)

    def vega(self):
        refDate  = self.underlyingSwap.discHandle.referenceDate()
        T = ql.Actual365Fixed().yearFraction(refDate,self.exercise.dates()[0])
        return self.annuity() * BachelierVega(self.underlyingSwap.fixedRate,self.fairRate(),self.normalVolatility,T) * 1.0e-4  # 1bp scaling

    def bondOptionDetails(self):
        # calculate expiryTime, (coupon) startTims, payTimes, cashFlows, strike and
        # c/p flag as inputs to Hull White analytic formula
        details = {}
        details['callOrPut'] = 1.0 if self.underlyingSwap.payerOrReceiver==ql.VanillaSwap.Receiver else -1.0
        details['strike']    = 0.0
        refDate  = self.underlyingSwap.discHandle.referenceDate()
        details['expiryTime'] = ql.Actual365Fixed().yearFraction(refDate,self.exercise.dates()[0])
        fixedLeg = [ [ ql.Actual365Fixed().yearFraction(refDate,cf.date()), cf.amount() ]
                     for cf in self.underlyingSwap.swap.fixedLeg() ]
        details['fixedLeg'] = np.array(fixedLeg)
        floatLeg = [ [ ql.Actual365Fixed().yearFraction(refDate,ql.as_coupon(cf).accrualStartDate()),
                       ((1 + ql.as_coupon(cf).accrualPeriod()*ql.as_coupon(cf).rate()) *
                        self.underlyingSwap.discHandle.discount(ql.as_coupon(cf).accrualEndDate()) /
                        self.underlyingSwap.discHandle.discount(ql.as_coupon(cf).accrualStartDate()) - 1.0) *
                       ql.as_coupon(cf).nominal() 
                       ] 
                     for cf in self.underlyingSwap.swap.floatingLeg() ]
        details['floatLeg'] = np.array(floatLeg)    
        payTimes = [ floatLeg[0][0]  ]          +       \
                   [ cf[0] for cf in floatLeg ] +       \
                   [ cf[0] for cf in fixedLeg ] +       \
                   [ ql.Actual365Fixed().yearFraction(refDate,ql.as_coupon(
                     self.underlyingSwap.swap.floatingLeg()[-1]).accrualEndDate()) ]
        caschflows = [ -ql.as_coupon(self.underlyingSwap.swap.floatingLeg()[0]).nominal() ] +  \
                     [ -cf[1] for cf in floatLeg ] +    \
                     [  cf[1] for cf in fixedLeg ] +    \
                     [ ql.as_coupon(self.underlyingSwap.swap.floatingLeg()[0]).nominal() ]
        details['payTimes'  ] = np.array(payTimes)
        details['cashFlows'] = np.array(caschflows)
        return details

    def swaptionDetails(self):
        # calculate times and cash flows as input to (cash-settled) swaption valuation
        details = {}
        details['callOrPut']  = 1.0 if self.underlyingSwap.payerOrReceiver==ql.VanillaSwap.Receiver else -1.0
        details['strikeRate'] = self.underlyingSwap.fixedRate
        details['notional']   = self.underlyingSwap.notional
        refDate  = self.underlyingSwap.discHandle.referenceDate()
        details['expiryTime'] = ql.Actual365Fixed().yearFraction(refDate,self.exercise.dates()[0])
        annuityLeg = [ [ ql.Actual365Fixed().yearFraction(refDate,cf.date()), ql.as_coupon(cf).accrualPeriod() ]
                       for cf in self.underlyingSwap.swap.fixedLeg() ]
        details['annuityLeg'] = np.array(annuityLeg)
        floatLeg = [ [ ql.Actual365Fixed().yearFraction(refDate,ql.as_coupon(cf).accrualStartDate()),
                       ((1 + ql.as_coupon(cf).accrualPeriod()*ql.as_coupon(cf).rate()) *
                        self.underlyingSwap.discHandle.discount(ql.as_coupon(cf).accrualEndDate()) /
                        self.underlyingSwap.discHandle.discount(ql.as_coupon(cf).accrualStartDate()) - 1.0) ]
                     for cf in self.underlyingSwap.swap.floatingLeg() ]
        floatLeg = floatLeg + [[ floatLeg[0][0], 1.0 ]] + \
                   [[ ql.Actual365Fixed().yearFraction(refDate,ql.as_coupon(
                     self.underlyingSwap.swap.floatingLeg()[-1]).accrualEndDate()), -1.0 ]]
        details['floatLeg'] = np.array(floatLeg)
        return details

    def npvHullWhite(self, hwModel, outFlag='p'):   # outFlag = [p]rice, [v]olatility or both [pv]
        details = self.bondOptionDetails()
        npv = hwModel.couponBondOption(details['expiryTime'], details['payTimes'], 
            details['cashFlows'], details['strike'], details['callOrPut'])
        if outFlag=='p': return npv
        vol = BachelierImpliedVol(npv/self.annuity(),self.underlyingSwap.fixedRate,
            self.underlyingSwap.fairRate(), details['expiryTime'], -details['callOrPut'])
        if outFlag=='v': return vol
        return [ npv, vol ]

    def payoff(self, hwModel):   # create a CouponBond payoff
        details = self.bondOptionDetails()
        cashFlowsCoP = [ details['callOrPut']*cf for cf in details['cashFlows'] ]
        return CouponBond(hwModel,details['expiryTime'],details['payTimes'],cashFlowsCoP)

# we provide an easy contructor function for convenience

def createSwaption(expiryTerm, swapTerm, discCurve, projCurve, strike='ATM', payerOrReceiver=ql.VanillaSwap.Payer, normalVolatility=0.01):
    today      = discCurve.yts.referenceDate()
    expiryDate = ql.TARGET().advance(today,ql.Period(expiryTerm),ql.ModifiedFollowing)
    startDate  = ql.TARGET().advance(expiryDate,ql.Period('2d'),ql.Following)
    endDate    = ql.TARGET().advance(startDate,ql.Period(swapTerm),ql.Unadjusted)
    if str(strike).upper()=='ATM':
        swap = Swap(startDate,endDate,0.0,discCurve,projCurve)
        strike = swap.fairRate()
    swap = Swap(startDate,endDate,strike,discCurve,projCurve,payerOrReceiver)
    swaption = swaption = Swaption(swap,expiryDate,normalVolatility)
    return swaption


def HullWhiteModelFromSwaption(swaption, meanReversion=0.01):
    volatilityTimes  = np.array( [ swaption.bondOptionDetails()['expiryTime'] ] )
    volatilityValues = np.array( [ swaption.normalVolatility ] )  # initial guess            
    def objective(sigma):
        volatilityValues[0] = sigma
        model = HullWhiteModel(swaption.underlyingSwap.discYieldCurve,meanReversion,volatilityTimes,volatilityValues)
        return swaption.npvHullWhite(model,'p') - swaption.npv()
    sigma = brentq(objective,0.01*volatilityValues[0],10.0*volatilityValues[0],xtol=1.0e-8)
    volatilityValues[0] = sigma
    return HullWhiteModel(swaption.underlyingSwap.discYieldCurve,meanReversion,volatilityTimes,volatilityValues)

class CashSettledSwaptionPayoff:
    # A Swaption is a priori assumed physically settled. However, we also want
    # to price cash-settled swaptions via Hull White and numerical methods
    
    # Python constructor
    def __init__(self, swaption, hwModel):
        self.swaption = swaption
        self.model = hwModel
        # we pre-calculate some quantities
        self.details = swaption.swaptionDetails()
        # we calculate a uniform year fraction for compounding
        # this is what we find in most papers
        # this should work for annual to quarterly compounding
        self.tau = round(4.0*self.details['annuityLeg'][-1][1])/4.0
        print(self.tau)

    def at(self, x):
        annuity = 0.0
        for cf in self.details['annuityLeg']:
            annuity += cf[1] * self.model.zeroBondPayoff(x,self.details['expiryTime'],cf[0]) 
        floatLeg = 0.0        
        for cf in self.details['floatLeg']: # unfortunately, this only contains spread coupons
            floatLeg += cf[1] * self.model.zeroBondPayoff(x,self.details['expiryTime'],cf[0])
        swapRate = floatLeg / annuity
        cashAnnuity = 0.0  #
        for k in range(self.details['annuityLeg'].shape[0]):
            cashAnnuity += self.tau / np.power(1.0 + self.tau*swapRate, k+1)
        return self.details['notional'] * cashAnnuity * self.details['callOrPut'] * \
               (swapRate - self.details['strikeRate'])


class CashPhysicalSwitchPayoff:    
    # Python constructor
    def __init__(self, swaption, hwModel):
        self.swaption = swaption
        self.model = hwModel
        # we pre-calculate some quantities
        self.details = swaption.swaptionDetails()
        # we calculate a uniform year fraction for compounding
        # this is what we find in most papers
        # this should work for annual to quarterly compounding
        self.tau = round(4.0*self.details['annuityLeg'][-1][1])/4.0

    def at(self, x):
        annuity = 0.0
        for cf in self.details['annuityLeg']:
            annuity += cf[1] * self.model.zeroBondPayoff(x,self.details['expiryTime'],cf[0]) 
        floatLeg = 0.0        
        for cf in self.details['floatLeg']: # unfortunately, this only contains spread coupons
            floatLeg += cf[1] * self.model.zeroBondPayoff(x,self.details['expiryTime'],cf[0])
        swapRate = floatLeg / annuity
        cashAnnuity = 0.0  #
        for k in range(self.details['annuityLeg'].shape[0]):
            cashAnnuity += self.tau / np.power(1.0 + self.tau*swapRate, k+1)
        return self.details['notional'] * (annuity-cashAnnuity) * \
               np.abs(swapRate - self.details['strikeRate'])

    
