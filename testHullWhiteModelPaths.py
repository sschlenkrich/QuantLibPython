import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__),'QuantLibWrapper'))

import numpy as np
from scipy.optimize import brentq

import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import pandas

import QuantLib as ql
import QuantLibWrapper.YieldCurve as yc

from QuantLibWrapper import HullWhiteModel, MCSimulation, Payoffs

# yield curves

flatCurve = yc.YieldCurve(['30y'],[0.03])

# We calibrate a Hull-White model to 100bp 'volatility' of x at 10y and 11y

meanReversion    = 0.2

def obj1(sigma1):
    model = HullWhiteModel(flatCurve,meanReversion,[5.0, 10.0],[sigma1, sigma1])
    return model.varianceX(0,5.0) - (1.0e-2)**2 * 5
sigma1 = brentq(obj1,1.0e-4,1.0e-1)

def obj2(sigma2):
    model = HullWhiteModel(flatCurve,meanReversion,[5.0, 10.0],[sigma1, sigma2])
    return model.varianceX(0,10.0) - (1.0e-2)**2 * 10
sigma2 = brentq(obj2,1.0e-4,1.0e-1)

print([sigma1, sigma2])
model = HullWhiteModel(flatCurve,meanReversion,[1.0, 10.0],[sigma1, sigma2])


times  = np.array([k*0.1 for k in range(101)])
nPaths = 50
mcSim  = MCSimulation(model,times,nPaths)

fig = plt.figure(figsize=(4, 6))
for path in mcSim.X:
    plt.plot(times, [ X[1] for X in path ])
plt.xlabel('Simulation time t')
plt.ylabel('State variable x(t)')
plt.title('a = %4.2f' % meanReversion)
plt.show()


