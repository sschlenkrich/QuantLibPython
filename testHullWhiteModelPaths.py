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
    model = HullWhiteModel(flatCurve,meanReversion,np.array([5.0, 10.0]),np.array([sigma1, sigma1]))
    return model.varianceX(0,5.0) - (1.0e-2)**2 * 5
sigma1 = brentq(obj1,1.0e-4,1.0e-1)

def obj2(sigma2):
    model = HullWhiteModel(flatCurve,meanReversion,np.array([5.0, 10.0]),np.array([sigma1, sigma2]))
    return model.varianceX(0,10.0) - (1.0e-2)**2 * 10
sigma2 = brentq(obj2,1.0e-4,1.0e-1)

print([sigma1, sigma2])
model = HullWhiteModel(flatCurve,meanReversion,[1.0, 10.0],[sigma1, sigma2])


times  = np.array([k*0.1 for k in range(101)])
nPaths = 10
mcSim  = MCSimulation(model,times,nPaths)

fig = plt.figure(figsize=(4, 6))
for path in mcSim.X:
    plt.plot(times, [ X[0] for X in path ])
plt.xlabel('Simulation time t')
plt.ylabel('State variable x(t)')
plt.title('a = %4.2f' % meanReversion)

fig = plt.figure(figsize=(4, 6))
for path in mcSim.X:
    plt.plot(times, [ np.exp(X[1]) for X in path ])
plt.xlabel('Simulation time t')
plt.ylabel('Numeraire N(t)')
plt.title('a = %4.2f' % meanReversion)
plt.show()



#exit()

##sigma = np.array([ np.std([ mcSim.X[i][j][0] for i in range(mcSim.X.shape[0])])/np.sqrt(times[j]) for j in range(mcSim.X.shape[1]) ])
#table = pandas.DataFrame([ times, sigma ]).T
#print(table)
#table.to_excel('table.xls')

def sigmaFwd(a):
    T0, T1, sigma = 5.0, 10.0, 1.0e-2
    y0 = sigma**2 * T0
    y1 = sigma**2 * T1
    return np.sqrt((y1 - np.exp(-a*(T1-T0))*y0)/(T1-T0))

fig = plt.figure(figsize=(8, 4))
meanRevs = np.linspace(-0.1,0.1,101)
plt.plot(meanRevs, np.array([ sigmaFwd(a)*1.0e+4 for a in meanRevs]))
plt.xlabel('Mean reversion t')
plt.ylabel('Forward volatility (bp)')
plt.title('T0 = 5y, T1 = 10y, spot sigma = 100bp')
plt.show()

