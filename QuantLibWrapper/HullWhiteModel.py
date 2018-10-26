#!/usr/bin/python

import numpy as np
from scipy.optimize import brentq
from Helpers import Black, Bachelier, BachelierImpliedVol

class HullWhiteModel:

    # Python constructor
    def __init__(self, yieldCurve, meanReversion, volatilityTimes, volatilityValues):
        self.yieldCurve       = yieldCurve
        self.meanReversion    = meanReversion
        self.volatilityTimes  = volatilityTimes    # assume positive and ascending
        self.volatilityValues = volatilityValues
        # pre-calculate y(t) on the time grid
        # y(t) = G'(s,t)^2 y(s) + sigma^2 [1 - exp{-2a(t-s)}] / (2a)
        t0 = 0.0
        y0 = 0.0
        self.y_ = np.zeros(len(self.volatilityTimes))
        for i in range(len(self.y_)):
            self.y_[i] = (self.GPrime(t0,self.volatilityTimes[i])**2) * y0 +                   \
                         (self.volatilityValues[i]**2) *                                       \
                         (1.0 - np.exp(-2*self.meanReversion*(self.volatilityTimes[i]-t0))) /  \
                         (2.0 * self.meanReversion)
            t0 = self.volatilityTimes[i]
            y0 = self.y_[i]
        
    # auxilliary methods

    def G(self, t, T):
        return (1.0 - np.exp(-self.meanReversion*(T-t))) / self.meanReversion
    
    def GPrime(self, t, T):
        return np.exp(-self.meanReversion*(T-t))
        
    def y(self,t):
        # find idx s.t. t[idx-1] <= t < t[idx]
        idx = len(self.volatilityTimes)-1
        for i in range(len(self.volatilityTimes),0,-1):   # better use bisection here
            idx = (i-1) if (t<self.volatilityTimes[i-1]) else idx
        t0 = 0.0 if (idx<0) else self.volatilityTimes[idx]
        y0 = 0.0 if (idx<0) else self.y_[idx]
        s1 = self.volatilityValues[min(idx+1,len(self.volatilityValues)-1)]
        y1 = (self.GPrime(t0,t)**2) * y0 +                      \
                s1**2 * (1.0 - np.exp(-2*self.meanReversion*(t-t0))) /  \
                (2.0 * self.meanReversion)
        return y1

    def riskNeutralExpectationX(self, t, xt, T):
        # E[x] = G'(t,T)x + \int_t^T G'(u,T)y(u)du
        def f(u):
            return self.GPrime(u,T)*self.y(u)
        # use Simpson's rule to approximate integral, this should better be solved analytically
        integral = (T-t) / 6 * (f(t) + 4*f((t+T)/2) + f(T)) 
        return self.GPrime(t,T)*xt + integral
    
    # model methods

    # conditional expectation in T-forward measure
    def expectationX(self, t, xt, T):
        return self.GPrime(t,T)*(xt + self.G(t,T)*self.y(t))

    def varianceX(self, t, T):
        return self.y(T) - self.GPrime(t,T)**2 * self.y(t)

    def zeroBond(self, t, xt, T):
        G = self.G(t,T)
        return self.yieldCurve.discount(T) / self.yieldCurve.discount(t) * \
            np.exp(-G*xt - 0.5 * G**2 * self.y(t) )

    def zeroBondOption(self, expiryTime, maturityTime, strikePrice, callOrPut):
        nu2 = self.G(expiryTime,maturityTime)**2 * self.y(expiryTime)
        P0  = self.yieldCurve.discount(expiryTime)
        P1  = self.yieldCurve.discount(maturityTime)
        return P0 * Black(strikePrice,P1/P0,np.sqrt(nu2),1.0, callOrPut)

    def couponBondOption(self, expiryTime, payTimes, cashFlows, strikePrice, callOrPut):
        def objective(x):
            bond = 0
            for i in range(len(payTimes)):
                bond += cashFlows[i] * self.zeroBond(expiryTime,x,payTimes[i])
            return bond - strikePrice
        xStar = brentq(objective,-0.3, 0.3, xtol=1.0e-8)   # +/-30% might be too narrow in some situations
        bondOption = 0.0
        for i in range(len(payTimes)):
            strike = self.zeroBond(expiryTime,xStar,payTimes[i])
            bondOption += cashFlows[i] * self.zeroBondOption(expiryTime,payTimes[i],strike,callOrPut)
        return bondOption

    # future yield curve in terms of forward rates
    def forwardRate(self, t, xt, T):
        return self.yieldCurve.forwardRate(T) + self.GPrime(t,T)*(xt + self.G(t,T)*self.y(t))

    # stochastic process interface
    
    def size(self):   # dimension of X(t)
        return 2      # [x, s], we also need for numeraire s = \int_0^t r dt

    def factors(self):   # dimension of W(t)
        return 1

    def initialValues(self):
        return np.array([ 0.0, 0.0 ])
    
    def zeroBondPayoff(self, X, t, T):
        return self.zeroBond(t, X[0], T)

    def numeraire(self, X):
        return np.exp(X[1])

    # evolve X(t0) -> X(t0+dt) using independent Brownian increments dW
    # t0, dt are assumed float, X0, X1, dW are np.array
    def evolve(self, t0, X0, dt, dW):
        x1 = self.riskNeutralExpectationX(t0,X0[0],t0+dt)
        # x1 = X0[0] + (self.y(t0) - self.meanReversion*X0[0])*dt
        nu = np.sqrt(self.varianceX(t0,t0+dt))
        x1 = x1 + nu*dW[0]
        # s1 = s0 + \int_t0^t0+dt r dt via Trapezoidal rule
        r0 = self.yieldCurve.forwardRate(t0)    + X0[0]
        r1 = self.yieldCurve.forwardRate(t0+dt) + x1
        s1 = X0[1] + (r0 + r1) * dt / 2
        # gather results
        return np.array([x1, s1])          
        
