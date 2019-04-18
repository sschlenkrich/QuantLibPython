#!/usr/bin/python

import QuantLib as ql

import numpy as np
from scipy.stats import norm
from scipy.interpolate import CubicSpline

# implement a model for a generic swap rate
class SwapRate:

    # Python constructor
    def __init__(self, floatTimes, floatWeights, annuityTimes, annuityWeights, discYieldCurve):
        self.floatTimes     = floatTimes
        self.floatWeights   = floatWeights
        self.annuityTimes   = annuityTimes
        self.annuityWeights = annuityWeights
        self.discYieldCurve = discYieldCurve
        self.invDistributionFunction = None  # interpolated function for CDF^{-1}( Phi(n) )
        self.gridSize   = 5   # +/-stdDev's for distribution function
        self.gridPoints = 201  # number of gridpoints for distribution function interpolation
        self.gapSize    = 1.0e-6  # finite difference step size for cdf calculation

    # calculate forward swap rate from curves provided
    def forwardRate(self):
        num = sum([ self.discYieldCurve.discount(T) * weight for T, weight in zip(self.floatTimes,self.floatWeights) ])
        den = sum([ self.discYieldCurve.discount(T) * weight for T, weight in zip(self.annuityTimes,self.annuityWeights) ])
        return num / den

    # calculate annuity
    def annuity(self):
        return sum([ self.discYieldCurve.discount(T) * weight for T, weight in zip(self.annuityTimes,self.annuityWeights) ])

    # calculate (future) swap rate based on linear zero rate model
    def swapRate(self,t,alpha,beta):
        def zeroBond(t,T):
            return np.exp(-(alpha*(T-t)+beta)*(T-t))  # basic zero bond model
        num = sum([ zeroBond(t,T) * weight for T, weight in zip(self.floatTimes,self.floatWeights) ])
        den = sum([ zeroBond(t,T) * weight for T, weight in zip(self.annuityTimes,self.annuityWeights) ])
        return num / den

    # calculate [dS/dalpha, dS/dbeta]
    def swapRateGradient(self,t,alpha,beta):
        # we use central finite differences for convenience
        h = 1.0e-6  # shift size
        dS_dalpha = (self.swapRate(t,alpha+h,beta) - self.swapRate(t,alpha-h,beta)) / 2.0 / h
        dS_dbeta  = (self.swapRate(t,alpha,beta+h) - self.swapRate(t,alpha,beta-h)) / 2.0 / h
        return [dS_dalpha, dS_dbeta]

    # set up terminal distribution
    def setUpTerminalDistribution(self,T,volTS):
        swapLength = round(max(self.floatTimes)-T)
        assert swapLength>0, 'Short swap rate not implemented'
        smile = ql.SmileSectionFromSwaptionVTS(volTS,T,swapLength)
        fwdRate = smile.atmLevel()
        stdDev  = smile.volatility(fwdRate,ql.Normal) * np.sqrt(smile.exerciseTime())
        uGrid   = np.linspace(norm.cdf(-self.gridSize),norm.cdf(self.gridSize),self.gridPoints)
        sGrid   = np.array([ norm.ppf(u)*stdDev for u in uGrid ])
        cdfGrid = np.array([ smile.digitalOptionPrice(fwdRate+s,ql.Option.Put,1.0,self.gapSize)
                             for s in sGrid ])
        # we need to deal with cases where rate might not be attainable
        minIdx = min(np.where(cdfGrid>0.0)[0])
        nGrid = np.array([ norm.ppf(u) for u in cdfGrid[minIdx:] ])
        self.invDistributionFunction = CubicSpline(nGrid,sGrid[minIdx:],bc_type='natural',extrapolate=True)

    def swapRateFromNormal(self,stdNormalVariable):
        assert self.invDistributionFunction != None,"Inverse CDF not initialised."
        return self.invDistributionFunction(stdNormalVariable)

# a constructor based on a swaption
def SwapRateFromSwaption(swaption):
    details = swaption.swaptionDetails()
    floatTimes     = [ cf[0] for cf in details['floatLeg'] ]
    floatWeights   = [ cf[1] for cf in details['floatLeg'] ]
    annuityTimes   = [ cf[0] for cf in details['annuityLeg'] ]
    annuityWeights = [ cf[1] for cf in details['annuityLeg'] ]
    discYieldCurve = swaption.underlyingSwap.discYieldCurve
    return SwapRate(floatTimes,floatWeights,annuityTimes,annuityWeights,discYieldCurve)


    