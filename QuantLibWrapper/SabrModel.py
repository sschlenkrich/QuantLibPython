#!/usr/bin/python

import numpy as np
from scipy.optimize import brentq
from Helpers import Bachelier, BachelierImpliedVol

class SabrModel:

    # Python constructor
    def __init__(self, forward, timeToExpiry, alpha, beta, nu, rho):
        self.forward      = forward
        self.timeToExpiry = timeToExpiry
        self.alpha        = alpha
        self.beta         = beta
        self.nu           = nu
        self.rho          = rho
        
    # helpers
    def localVolC(self, rate):
        return np.power(rate,self.beta) if rate>0.0 else 0.0
        
    def localVolCPrime(self, rate):  # for Milstein method
        return self.beta * np.power(rate,self.beta-1) if rate>0.0 else 0.0
        
    def sAverage(self, strike, forward):
        return (strike + forward) / 2.0
        # return np.power(strike*forward,0.5) # check consistency to QL
    
    def zeta(self, strike, forward):
        return self.nu / self.alpha * (np.power(forward,1-self.beta)-np.power(strike,1-self.beta)) / (1-self.beta)
        
    def chi(self, zeta):
        return np.log((np.sqrt(1-2*self.rho*zeta+zeta*zeta)-self.rho+zeta)/(1-self.rho))
     
    # approximate implied normal volatility formula 
    def normalVolatility(self,strike):
        Sav     = self.sAverage(strike,self.forward)
        CSav    = self.localVolC(Sav)
        gamma1  = self.beta / Sav
        gamma2  = self.beta * (self.beta-1) / Sav / Sav
        I1      = (2*gamma2 - gamma1*gamma1) / 24 * self.alpha * self.alpha * CSav * CSav
        I1      = I1 + self.rho * self.nu * self.alpha * gamma1 / 4 * CSav
        I1      = I1 + (2 - 3*self.rho*self.rho) / 24 * self.nu * self.nu
        sigmaN  = self.alpha * CSav  # default, if close to ATM
        if np.fabs(strike-self.forward)>1.0e-8:  # actual calculation for I0
            sigmaN = self.nu * (self.forward - strike) / self.chi(self.zeta(strike,self.forward))
        sigmaN  = sigmaN * (1 + I1*self.timeToExpiry)  # higher order adjustment
        return sigmaN

    def calibrateATM(self, sigmaATM):
        def objective(alpha):
            self.alpha = alpha
            return self.normalVolatility(self.forward) - sigmaATM
        alpha0 = sigmaATM / self.localVolC(self.forward)
        self.alpha = brentq(objective,0.5*alpha0, 2.0*alpha0, xtol=1.0e-8)
        return self.alpha


    def vanillaPrice(self, strike, callOrPut):
        sigmaN = self.normalVolatility(strike)
        return Bachelier(strike,self.forward,sigmaN,self.timeToExpiry,callOrPut)

    def density(self, rate):
        eps = 1.0e-4
        cop = 1.0
        if (rate<self.forward):
            cop = -1.0
        dens = (self.vanillaPrice(rate-eps,cop) - 2*self.vanillaPrice(rate,cop) + self.vanillaPrice(rate+eps,cop))/eps/eps
        return dens

    # stochastic process interface
    
    def size(self):   # dimension of X(t)
        return 2

    def factors(self):   # dimension of W(t)
        return 2

    def initialValues(self):
        return np.array([ self.forward, self.alpha ])
    
    # evolve X(t0) -> X(t0+dt) using independent Brownian increments dW
    # t0, dt are assumed float, X0, X1, dW are np.array
    def evolve(self, t0, X0, dt, dW):
        # first simulate stochastic volatility exact
        dZ = self.rho * dW[0] + np.sqrt(1-self.rho*self.rho)*dW[1]
        alpha0 = X0[1]
        alpha1 = alpha0*np.exp(-self.nu*self.nu/2*dt+self.nu*dZ*np.sqrt(dt))
        alpha01 = np.sqrt(alpha0*alpha1)   # average vol [t0, t0+dt]
        # simulate S via Milstein
        S0 = X0[0]
        S1 = S0 + alpha01*self.localVolC(S0)*dW[0]*np.sqrt(dt) \
                + 0.5*alpha01*self.localVolC(S0)*alpha01*self.localVolCPrime(S0)*(dW[0]*dW[0]-1)*dt 
        # gather results
        return np.array([S1, alpha1])          
        
# calculate normal volatility smile from a MC simulation
    def monteCarloImpliedNormalVol(self, mcSimulation, strikes, fullOutput=False):
        if mcSimulation.times[-1]!=self.timeToExpiry: print('WARNING: times do not match.')
        # forward adjuster
        Sav = 0.0
        for j in range(mcSimulation.nPaths): Sav += mcSimulation.X[j][-1][0]
        Sav /= mcSimulation.nPaths
        S = np.array([(mcSimulation.X[j][-1][0]+self.forward-Sav) for j in range(mcSimulation.nPaths) ])
        options = np.zeros(len(strikes))
        vols    = np.zeros(len(strikes))
        for i in range(len(strikes)):
            cop = 1.0 if strikes[i]>self.forward else -1.0
            for j in range(mcSimulation.nPaths):
                options[i] += np.max([cop*(S[j]-strikes[i]), 0.0])
            options[i] /= mcSimulation.nPaths
            try:     vols[i] = BachelierImpliedVol(options[i],strikes[i],self.forward,mcSimulation.times[-1],cop)
            except:  vols[i] = 0.0
        return vols if not fullOutput else np.array(strikes, options, vols)
