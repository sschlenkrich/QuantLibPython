import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__),'QuantLibWrapper'))

from scipy.optimize import brentq
import numpy as np
import pandas

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3d plotting
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import QuantLib as ql

from QuantLibWrapper import HullWhiteModel, HullWhiteModel2F, TerminalSwapRateModel, Hw2fTsrModel

# we implement a model calibration routines to simplify model scenario generation
def scaleModel(model, vol=0.01, T=5.0, dT=1.0/365.0):
    lambda_ = vol / model.fwdYieldVolatility(T,dT)
    return HullWhiteModel2F(model.yieldCurve,model.chi,lambda_*model.sigma,model.rho)

def printModel(model):
    print('chi   = [ ' + str(model.chi[0])   + ', ' + str(model.chi[1])   + ' ]')
    print('sigma = [ ' + str(model.sigma[0]) + ', ' + str(model.sigma[1]) + ' ]')
    print('rho   = [ ' + str(model.rho) + ' ]')

# we do tests with a flat curve of 3%; YC should not impact CA anyway
hYts = ql.YieldTermStructureHandle(
           ql.FlatForward(ql.Settings.getEvaluationDate(ql.Settings.instance()),
                          0.03,ql.Actual365Fixed()))

# our base model is 1-factor with 5% mean reversion and 100bp volatility
chi   = np.array([ 0.20, 0.05 ])
sigma = np.array([ 0.00, 0.01 ])
rho   = 0.0 # not used

# we consider a 5y expiry rate on a 6m Libor index as a first step
T  = 5.0
dT = 0.5

# our benchmark model is the bi-linear TSR model
tsrModel = TerminalSwapRateModel(hYts,T,dT)

# we do a first comparison with the base 1-factor model
h1fModel = Hw2fTsrModel(HullWhiteModel2F(hYts,chi,sigma,rho),T,dT)

# for plotting we consider payment offsets from 0 to 5y
Tp = np.linspace(0,5.0,101)

# we calculate convexity adjustment factors a(Tp)*An(0)/P(0,Tp) which are independent of most other factors
y1  = np.array([ tsrModel.convexityAdjustmentFactor(T+t) for t in Tp ])
y2  = np.array([ h1fModel.convexityAdjustmentFactor(T+t) for t in Tp ])

# we plot results...
plt.figure(figsize=(6, 5))
plt.plot(Tp, y1, 'b-', label='Bi-linear TSR')
plt.plot(Tp, y2, 'r-', label='1-factor HJM')
plt.legend()
plt.xlabel(r'$T_p - T_0$')
plt.ylabel(r'$a(T_p) * An(0) / P(0,T_p)$')
plt.title('T = ' + str(T) + ', dT = ' + str(dT))

##################################################################################
# 1-factor volatility test case
##################################################################################

# we analyse the impact of model volatility on the convexity adjustment
fig = plt.figure(figsize=(6, 5))
ax = fig.gca()
sigmas = [ 1, 100, 200, 400]  # bp's
ax.plot(Tp, y1, 'b-', label='Bi-linear TSR')
for s in sigmas:
    model = Hw2fTsrModel(HullWhiteModel2F(hYts,chi,np.array([ 0.00, s*1.0e-4 ]),rho),T,dT)
    y = np.array([ model.convexityAdjustmentFactor(T+t) for t in Tp ])
    ax.plot(Tp, y, label=r'$\sigma = $' + str(s) + 'bp')
ax.legend()
plt.xlabel(r'$T_p - T_0$')
plt.ylabel(r'$a(T_p) * An(0) / P(0,T_p)$')
ax.set_title(r'HJM 1F CA factor per $\sigma$ ($T_0=5$, $\Delta T=0.5$, $\chi = 5\%$)')
# we see that volatility has very limited effect on CA

##################################################################################
# 1-factor mean reversion test case
##################################################################################

# we analyse the impact of mean reversion on the 1-factor model CA
# to avoid (small) impacts from volatility changes we resclale to 100bp vol
meanReversions = [ -30, -10, 10, 30 ]
shortVol = 0.01  # 100bp vol for scaling
fig = plt.figure(figsize=(6, 5))
ax = fig.gca()
ax.plot(Tp, y1, 'b-', label='Bi-linear TSR')
for chi in meanReversions:
    hw2f = scaleModel(HullWhiteModel2F(hYts,np.array([ 1.0, chi*1.0e-2 ]),sigma,rho), vol=shortVol, T=T, dT=1.0/365.0)
    model = Hw2fTsrModel(hw2f,T,dT)
    y = np.array([ model.convexityAdjustmentFactor(T+t) for t in Tp ])
    ax.plot(Tp, y, label=r'$\chi = $' + str(round(chi)) + r'%, $\sigma = $' + str(int(round(hw2f.sigma[1]*1e4))) + 'bp' )
ax.legend()
plt.xlabel(r'$T_p - T_0$')
plt.ylabel(r'$a(T_p) * An(0) / P(0,T_p)$')
ax.set_title(r'HJM 1F CA factor per $\chi$ ($T_0=5$, $\Delta T=0.5$, 5y-Vol 100bp)')

###################################################################################
# 2-factor model transformation tst case
###################################################################################

# we set up a base model for nice contour line plots
chi   = np.array([ 0.20, 0.05 ])
sigma = np.array([ 0.02, 0.01 ])
rho   = -0.50
model2f  = scaleModel(HullWhiteModel2F(hYts,chi,sigma,rho),vol=0.01, T=T, dT=1.0/365.0)
h2fModel = Hw2fTsrModel(model2f,T,dT)

#h2fModel.plot()
#h2fModel.plotTransformed()

###################################################################################
# Base model(s) for all other analysis
###################################################################################

# 1-factor models
h1fModelLow  = Hw2fTsrModel(scaleModel(HullWhiteModel2F(hYts, np.array([ 1.0, -0.1 ]),
                   np.array([ 0.00, 0.01 ]),0.0), vol=shortVol, T=T, dT=1.0/365.0),T,dT)
h1fModelHigh = Hw2fTsrModel(scaleModel(HullWhiteModel2F(hYts, np.array([ 1.0, +0.1 ]),
                   np.array([ 0.00, 0.01 ]),0.0), vol=shortVol, T=T, dT=1.0/365.0),T,dT)
#
yLow  = np.array([ h1fModelLow.convexityAdjustmentFactor(T+t) for t in Tp ])
yHigh = np.array([ h1fModelHigh.convexityAdjustmentFactor(T+t) for t in Tp ])

# 2-factor models
chi   = np.array([ 0.05, 0.20 ])
sigma = np.array([ 0.01, 0.02 ])
rho   = -0.95
model2f  = scaleModel(HullWhiteModel2F(hYts,chi,sigma,rho),vol=0.01, T=T, dT=1.0/365.0)
h2fModel = Hw2fTsrModel(model2f,T,dT)


# 2-factor volatility test case
# we analyse impact of volatility ratio on CA
# for this test we parametrize sigma2 = lambda * sigma1
# to avoid (small) impacts from volatility changes we resclale to 100bp vol
shortVol = 0.01  # 100bp vol for scaling
shift = 15  # in %
lambdas = [ 0.5, 1.0, 2.0, 3.0 ]  # in %
fig = plt.figure(figsize=(6, 5))
ax = fig.gca()
ax.plot(Tp, yLow, 'k--', label='HJM 1F benchmark')
ax.plot(Tp, yHigh, 'k--')
for la in lambdas:
    hw2f = scaleModel(HullWhiteModel2F(hYts,np.array([ 0.05, 0.20 ]),np.array([ 0.01, la*0.01 ]),rho), vol=shortVol, T=T, dT=1.0/365.0)
    model = Hw2fTsrModel(hw2f,T,dT)
    y = np.array([ model.convexityAdjustmentFactor(T+t) for t in Tp ])
    ax.plot(Tp, y, label=r'$\lambda = $' + str(round(la,1)) + r', $\sigma = $[' + str(int(round(hw2f.sigma[0]*1e4))) + ', ' + str(int(round(hw2f.sigma[1]*1e4))) + ']bp' )
ax.legend()
plt.xlabel(r'$T_p - T_0$')
plt.ylabel(r'$a(T_p) * An(0) / P(0,T_p)$')
ax.set_title(r'HJM 2F CA factor per $\lambda$ ($T_0=5$, $\Delta T=0.5$, 5y-Vol 100bp)')


# 2-factor mean reversion level test case
# test dependency on chi1 parameter
shortVol = 0.01  # 100bp vol for scaling
shift = 15  # in %
meanReversions = [ -30, -10, 10, 30 ]  # in %
fig = plt.figure(figsize=(6, 5))
ax = fig.gca()
ax.plot(Tp, yLow, 'k--', label='HJM 1F benchmark')
ax.plot(Tp, yHigh, 'k--')
for chi1 in meanReversions:
    hw2f = scaleModel(HullWhiteModel2F(hYts,np.array([ chi1*1.0e-2, (chi1+shift)*1.0e-2 ]),sigma,rho), vol=shortVol, T=T, dT=1.0/365.0)
    model = Hw2fTsrModel(hw2f,T,dT)
    y = np.array([ model.convexityAdjustmentFactor(T+t) for t in Tp ])
    ax.plot(Tp, y, label=r'$\chi = $[' + str(round(chi1)) + ', ' + str(round(chi1+shift)) + ']%, '
        r'$\sigma = $[' + str(int(round(hw2f.sigma[0]*1e4))) + ', ' + str(int(round(hw2f.sigma[1]*1e4))) + ']bp' )
ax.legend()
plt.xlabel(r'$T_p - T_0$')
plt.ylabel(r'$a(T_p) * An(0) / P(0,T_p)$')
ax.set_title(r'HJM 2F CA factor per $\chi_1$ ($T_0=5$, $\Delta T=0.5$, 5y-Vol 100bp)')

# 2-factor mean revrsion distance test case
# test dependency on shift parameter
shortVol = 0.01  # 100bp vol for scaling
chi1 = 5  # in %
shifts = [ 1, 10, 25, 50 ]  # in %
fig = plt.figure(figsize=(6, 5))
ax = fig.gca()
ax.plot(Tp, yLow, 'k--', label='HJM 1F benchmark')
ax.plot(Tp, yHigh, 'k--')
for shift in shifts:
    hw2f = scaleModel(HullWhiteModel2F(hYts,np.array([ chi1*1.0e-2, (chi1+shift)*1.0e-2]),sigma,rho), vol=shortVol, T=T, dT=1.0/365.0)
    model = Hw2fTsrModel(hw2f,T,dT)
    y = np.array([ model.convexityAdjustmentFactor(T+t) for t in Tp ])
    ax.plot(Tp, y, label=r'$\chi = $[' + str(round(chi1)) + ', ' + str(round(chi1+shift)) + ']%, '
        r'$\sigma = $[' + str(int(round(hw2f.sigma[0]*1e4))) + ', ' + str(int(round(hw2f.sigma[1]*1e4))) + ']bp' )
ax.legend()
plt.xlabel(r'$T_p - T_0$')
plt.ylabel(r'$a(T_p) * An(0) / P(0,T_p)$')
ax.set_title(r'HJM 2F CA factor per $\delta$ ($T_0=5$, $\Delta T=0.5$, 5y-Vol 100bp)')

# 2-factor correlation test case
# we analyse the impact of correlation on the 2-factor model CA
# to avoid (small) impacts from volatility changes we resclale to 100bp vol

shortVol = 0.01  # 100bp vol for scaling
corrs = [ -99, -95, -80, -50 ]
fig = plt.figure(figsize=(6, 5))
ax = fig.gca()
ax.plot(Tp, yLow, 'k--', label='HJM 1F benchmark')
ax.plot(Tp, yHigh, 'k--')
for corr in corrs:
    hw2f = scaleModel(HullWhiteModel2F(hYts,chi,sigma,corr*1.0e-2), vol=shortVol, T=T, dT=1.0/365.0)
    model = Hw2fTsrModel(hw2f,T,dT)
    y = np.array([ model.convexityAdjustmentFactor(T+t) for t in Tp ])
    ax.plot(Tp, y, label=r'$\rho = $' + str(round(corr)) + '%, '
        r'$\sigma = $[' + str(int(round(hw2f.sigma[0]*1e4))) + ', ' + str(int(round(hw2f.sigma[1]*1e4))) + ']bp' )
ax.legend()
plt.xlabel(r'$T_p - T_0$')
plt.ylabel(r'$a(T_p) * An(0) / P(0,T_p)$')
ax.set_title(r'HJM 2F CA factor per $\rho$ ($T_0=5$, $\Delta T=0.5$, 5y-Vol 100bp)')


###################################################################################
# 2-factor model 10y swaption test case
###################################################################################

# we set up a base model for nice contour line plots
chi   = np.array([ 0.20, 0.05 ])
sigma = np.array([ 0.02, 0.01 ])
rho   = -0.50
model2f  = scaleModel(HullWhiteModel2F(hYts,chi,sigma,rho),vol=0.01, T=T, dT=1.0/365.0)
h2fModel = Hw2fTsrModel(model2f,5,10)

h2fModel.plot()

plt.show()

