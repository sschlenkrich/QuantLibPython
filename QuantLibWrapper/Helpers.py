#!/usr/bin/python

from scipy.stats import norm
from scipy.optimize import brentq
import numpy as np


def BlackOverK(moneyness, stdDev, callOrPut):
    d1 = np.log(moneyness) / stdDev + stdDev / 2.0
    d2 = d1 - stdDev
    return callOrPut * (moneyness*norm.cdf(callOrPut*d1)-norm.cdf(callOrPut*d2))

def Black(strike, forward, sigma, T, callOrPut):
    nu = sigma*np.sqrt(T)
    if nu<1.0e-12:   # assume zero
        return max(callOrPut*(forward-strike),0.0)  # intrinsic value
    return strike * BlackOverK(forward/strike,nu,callOrPut)

def BlackImpliedVol(price, strike, forward, T, callOrPut):
    def objective(sigma):
        return Black(strike, forward, sigma, T, callOrPut) - price
    return brentq(objective,0.01, 1.00, xtol=1.0e-8)

def BachelierRaw(moneyness, stdDev, callOrPut):
    h = callOrPut * moneyness / stdDev
    return stdDev * (h*norm.cdf(h) + norm.pdf(h))

def BachelierVegaRaw(moneyness, stdDev):
    return norm.pdf(moneyness / stdDev)

def Bachelier(strike, forward, sigma, T, callOrPut):
    return BachelierRaw(forward-strike,sigma*np.sqrt(T),callOrPut)

def BachelierVega(strike, forward, sigma, T):
    return BachelierVegaRaw(forward-strike,sigma*np.sqrt(T))*np.sqrt(T)

def BachelierImpliedVol(price, strike, forward, T, callOrPut):
    def objective(sigma):
        return Bachelier(strike, forward, sigma, T, callOrPut) - price
    return brentq(objective,1e-4, 1e-1, xtol=1.0e-8)

