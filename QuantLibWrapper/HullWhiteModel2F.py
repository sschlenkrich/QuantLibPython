#!/usr/bin/python

import numpy as np
from math import ceil
from scipy.optimize import brentq
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from Helpers import Black, Bachelier, BachelierImpliedVol

class HullWhiteModel2F:

    # Python constructor
    def __init__(self, yieldCurve, chi, sigma, rho):
        self.yieldCurve = yieldCurve
        self.chi        = chi      # chi[0:1]
        self.sigma      = sigma    # sigma[0:1]
        self.rho        = rho      # scalar

    # auxilliary methods

    def G(self, t, T):
        return np.array([
            (1.0 - np.exp(-self.chi[0]*(T-t))) / self.chi[0],
            (1.0 - np.exp(-self.chi[1]*(T-t))) / self.chi[1] ])

    def GPrime(self, t, T):
        return np.array([ np.exp(-self.chi[0]*(T-t)), np.exp(-self.chi[1]*(T-t)) ])

    def y(self,t):        
        return np.array([
            [ self.sigma[0]**2/(2*self.chi[0])*(1-np.exp(-2*self.chi[0]*t)),
              self.rho*self.sigma[0]*self.sigma[1]/(self.chi[0]+self.chi[1])*(1-np.exp(-(self.chi[0]+self.chi[1])*t)) ],
            [ self.rho*self.sigma[0]*self.sigma[1]/(self.chi[0]+self.chi[1])*(1-np.exp(-(self.chi[0]+self.chi[1])*t)),
              self.sigma[1]**2/(2*self.chi[1])*(1-np.exp(-2*self.chi[1]*t))                                           ] ])

    def zeroBond(self, t, xt, T):
        G = self.G(t,T)
        y = self.y(t)
        return self.yieldCurve.discount(T) / self.yieldCurve.discount(t) * \
            np.exp(-np.dot(G,xt) - 0.5 * np.dot(G,np.matmul(y,G)))

    def fwdRate(self, t, xt, dT):
        floatLeg = 1.0 - self.zeroBond(t,xt,t+dT)
        times = list(range(ceil(dT))) + [dT]
        annuity = sum([ (times[k]-times[k-1])*self.zeroBond(t,xt,t+times[k]) \
                        for k in range(1,len(times)) ])
        return floatLeg / annuity

    def densityT(self, t, xt):
        return multivariate_normal.pdf(xt, mean=None, cov=self.y(t))

    def fwdYieldVolatility(self, T, dT):
        G = self.G(T,T+dT)
        y = self.y(T)
        return np.sqrt( np.dot(G,np.matmul(y,G)) / dT**2 ) / np.sqrt(T)
        

# We use bi-linear annuity mapping function as base class implementation
class TerminalSwapRateModel:

    # Python constructor
    def __init__(self, yieldCurve, T, dT):
        self.yieldCurve = yieldCurve
        self.T          = T   # fixing time
        self.dT         = dT  # T_M - T
        self.times      = list(range(ceil(dT))) + [dT]   # [0, 1, ..., M, T+dT]

    def annuity(self):
        return sum([ (self.times[k]-self.times[k-1])*self.yieldCurve.discount(self.T+self.times[k]) \
                        for k in range(1,len(self.times)) ])

    def fwdRate(self):
        floatLeg = 1.0 - self.yieldCurve.discount(self.T+self.dT)
        return floatLeg / self.annuity()

    # linear slope function
    def slope(self,Tp):
        u = 1.0 / self.dT
        v = - sum([ (self.times[k]-self.times[k-1])*(self.dT-self.times[k]) \
                  for k in range(1,len(self.times)) ]) / (self.dT**2)
        return u * (self.T + self.dT - Tp) + v

    # linear TSR model function
    def alpha(self,s,Tp):
        return self.slope(Tp)*(s-self.fwdRate()) + self.yieldCurve.discount(Tp) / self.annuity()

    # a(Tp)*An(0)/P(0,Tp)
    def convexityAdjustmentFactor(self,Tp):
        return self.slope(Tp)*self.annuity()/self.yieldCurve.discount(Tp)

class Hw2fTsrModel(TerminalSwapRateModel):
    
    # Python constructor
    def __init__(self, model, T, dT):
        TerminalSwapRateModel.__init__(self,model.yieldCurve,T,dT)
        self.model = model

    def g(self):
        if self.dT>1.0:  # maybe better throw an exception...
            print('Warning! Methodology not implemented for dT>1')
        return self.model.G(self.T,self.T+self.dT)      

    def c(self,s):
        if self.dT>1.0:  # maybe better throw an exception...
            print('Warning! Methodology not implemented for dT>1')
        G = self.model.G(self.T,self.T+self.dT)
        Y = self.model.y(self.T)
        S0 = self.fwdRate()
        return np.log((1+self.dT*s)/(1+self.dT*S0)) - 0.5 * np.dot(G,np.matmul(Y,G))

    def pi(self,s,Tp):
        g = self.g()
        c = self.c(s)
        Y = self.model.y(self.T)
        m = g[1]/g[0]
        barY = np.array([
            [ Y[0][0],             Y[0][1] + Y[0][0]*m                 ],
            [ Y[0][1] + Y[0][0]*m, Y[1][1] + 2*Y[0][1]*m + Y[0][0]*m*m ] ])
        barMu = c/g[1] * np.array([ barY[0][1]/barY[1][1], 1.0 ])
        barV = np.array([
            [ barY[0][0] - barY[0][1]*barY[0][1]/barY[1][1], 0 ],
            [ 0                                            , 0 ] ])
        Minv = np.array([
            [ 1.0, 0.0 ],
            [  -m, 1.0 ] ])
        tilY = Y - np.matmul(Minv,np.matmul(barV,np.transpose(Minv)))
        # now we add the Tp-dependent G-parameter
        G = self.model.G(self.T,Tp)
        tmp = -np.dot(G,np.matmul(Minv,barMu)) - 0.5 * np.dot(G,np.matmul(tilY,G))
        return self.yieldCurve.discount(Tp) / self.yieldCurve.discount(self.T) * np.exp(tmp)

    def alpha(self,s,Tp):
        num = self.pi(s,Tp)
        den = sum([ (self.times[k]-self.times[k-1])*self.pi(s,self.T+self.times[k])  \
                  for k in range(1,len(self.times)) ])
        return num / den

    def slope(self,Tp):
        if self.dT>1.0:  # maybe better throw an exception...
            print('Warning! Methodology not implemented for dT>1')
        S0 = self.fwdRate()        
        # we need to copy the code from pi(...)
        g = self.g()
        Y = self.model.y(self.T)
        m = g[1]/g[0]
        barY = np.array([
            [ Y[0][0],             Y[0][1] + Y[0][0]*m                 ],
            [ Y[0][1] + Y[0][0]*m, Y[1][1] + 2*Y[0][1]*m + Y[0][0]*m*m ] ])
        barMuPrime = self.dT/(1+self.dT*S0)/g[1] * np.array([ barY[0][1]/barY[1][1], 1.0 ])
        Minv = np.array([
            [ 1.0, 0.0 ],
            [  -m, 1.0 ] ])
        deltaG = self.model.G(self.T,Tp) - self.model.G(self.T,self.T+self.dT)
        return -self.alpha(S0,Tp) * np.dot(deltaG,np.matmul(Minv,barMuPrime))

    def slope2(self,Tp):
        ds = 1.0e-4
        S0 = self.fwdRate()
        return (self.alpha(S0+ds,Tp)-self.alpha(S0-ds,Tp))/(2.0*ds)


    # we incorporate some routines for illustration

    def plot(self, s=None):
        X0 = np.linspace(-0.1,0.1,201)
        X1 = np.linspace(-0.1,0.1,201)
        dens = np.array([ [ self.model.densityT(self.T, np.array([x0,x1])) for x1 in X1 ] for x0 in X0 ])
        dLogMin = np.log(np.min(dens))
        dLogMax = np.log(np.max(dens))
        contLevels = np.exp(np.linspace(0.5*(dLogMin+dLogMax),dLogMax,11))
        rate = np.array([ [ self.model.fwdRate(self.T, np.array([x0,x1]),self.dT) for x1 in X1 ] for x0 in X0 ])
        X, Y = np.meshgrid(X0, X1, indexing='ij')
        fig = plt.figure(figsize=(6, 6))
        ax = fig.gca()
        ax.contour(X, Y, dens, contLevels, colors='red', linestyles='dashed')
        #ax.contour(X, Y, rate, np.linspace(-0.1,0.1,11), colors='blue', linestyles='solid')
        # draw single iso-line
        S0 = self.fwdRate() if s==None else s
        c = self.c(S0)
        g = self.g()
        X1 = np.array([ (c - g[0]*x)/g[1] for x in X0 ])
        ax.plot(X0,X1, 'g-')
        ax.set_xlabel(r'$x_1$')
        ax.set_ylabel(r'$x_2$')
        plt.tight_layout()
        return fig

    def plotTransformed(self, s=None):
        # we need to calculate the transformation
        S0 = self.fwdRate() if s==None else s
        g = self.g()
        c = self.c(S0)
        Y = self.model.y(self.T)
        m = g[1]/g[0]
        barY = np.array([
            [ Y[0][0],             Y[0][1] + Y[0][0]*m                 ],
            [ Y[0][1] + Y[0][0]*m, Y[1][1] + 2*Y[0][1]*m + Y[0][0]*m*m ] ])
        # now we can repeat plotting
        X0 = np.linspace(-0.1,0.1,201)
        X1 = np.linspace(-0.1,0.1,201)
        dens = np.array([ [ multivariate_normal.pdf(np.array([x0,x1]), mean=None, cov=barY) for x1 in X1 ] for x0 in X0 ])
        dLogMin = np.log(np.min(dens))
        dLogMax = np.log(np.max(dens))
        contLevels = np.exp(np.linspace(0.5*(dLogMin+dLogMax),dLogMax,11))
        X, Y = np.meshgrid(X0, X1, indexing='ij')
        fig = plt.figure(figsize=(6, 6))
        ax = fig.gca()
        ax.contour(X, Y, dens, contLevels, colors='red', linestyles='dashed')
        # draw single iso-line
        S0 = self.fwdRate() if s==None else s
        c = self.c(S0)
        g = self.g()
        X1 = np.array([ c/g[1] for x in X0 ])
        ax.plot(X0,X1, 'g-')
        ax.set_xlabel(r'$\bar{x}_1$')
        ax.set_ylabel(r'$\bar{x}_2$')
        plt.tight_layout()
        return fig



        