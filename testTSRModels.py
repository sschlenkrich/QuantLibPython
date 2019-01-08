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

# we implement a couple of model calibration routines to
# simplify model scenario generation

def scaleModel(model, vol=0.01, T=5.0, dT=1.0/365.0):
    #def objective(lambda_):
    #    return HullWhiteModel2F(model.yieldCurve,model.chi,lambda_*model.sigma,model.rho).fwdYieldVolatility(T,dT) - vol
    #lambda_ = brentq(objective,0.1*lambda_, 10.0*lambda_, xtol=1.0e-8)
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
chi   = np.array([ 0.15, 0.05 ])
sigma = np.array([ 0.00, 0.01 ])
rho   = 0.0 # not used

# we consider a 5y expiry rate on a 6m Libor index as a first step
T  = 5.0
dT = 0.5

# our benchmark model is the bi-linear TSR model
tsrModel = TerminalSwapRateModel(hYts,T,dT)

# we do a first comparison with the base 1-factor model
h1fModel = Hw2fTsrModel(HullWhiteModel2F(hYts,chi,sigma,rho),T,dT)

# for plotting we consider payment offsets from 0 to 2y
Tp = np.linspace(0,5.0,101)

# we calculate convexity adjustment factors a(Tp)*An(0)/P(0,Tp) which are independent of most other factors
y1  = np.array([ tsrModel.convexityAdjustmentFactor(T+t) for t in Tp ])
y2  = np.array([ h1fModel.convexityAdjustmentFactor(T+t) for t in Tp ])

# we plot results...
plt.figure(figsize=(6, 5))
plt.plot(Tp, y1, 'b-', label='Bi-linear TSR')
plt.plot(Tp, y2, 'r-', label='1-factor HJM')
plt.legend()
plt.xlabel('T_p - T')
plt.ylabel('a(Tp) * An(0) / P(0,Tp)')
plt.title('T = ' + str(T) + ', dT = ' + str(dT))

# we create a single figure and add subplots to get an overview
fig = plt.figure(figsize=(12, 16))
fig.suptitle('T = ' + str(T) + ', dT = ' + str(dT))

# 1-factor volatility test case
# we analyse the impact of model volatility on the convexity adjustment

sigmas = [ 1, 200, 400]  # bp's
ax = fig.add_subplot(4,2,1)
ax.plot(Tp, y1, 'b-', label='Bi-linear TSR')
ax.plot(Tp, y2, 'r-', label='1-factor HJM')
for s in sigmas:
    model = Hw2fTsrModel(HullWhiteModel2F(hYts,chi,np.array([ 0.00, s*1.0e-4 ]),rho),T,dT)
    y = np.array([ model.convexityAdjustmentFactor(T+t) for t in Tp ])
    ax.plot(Tp, y, label='sigma = ' + str(s) + 'bp')
ax.legend()
ax.set_xlabel('T_p - T')
ax.set_ylabel('a(Tp) * An(0) / P(0,Tp)')
ax.set_title('chi = ' + str(round(chi[1]*100)) + '%')
# we see that volatility has very limited effect on CA

# 1-factor mean reversion test case
# we analyse the impact of mean reversion on the 1-factor model CA
# to avoid (small) impacts from volatility changes we resclale to 100bp vol

meanReversions = [ -30, -10, 10, 30 ]
shortVol = 0.01  # 100bp vol for scaling
ax = fig.add_subplot(4,2,2)
ax.plot(Tp, y1, 'b-', label='Bi-linear TSR')
for chi in meanReversions:
    hw2f = scaleModel(HullWhiteModel2F(hYts,np.array([ 1.0, chi*1.0e-2 ]),sigma,rho), vol=shortVol, T=T, dT=1.0/365.0)
    model = Hw2fTsrModel(hw2f,T,dT)
    y = np.array([ model.convexityAdjustmentFactor(T+t) for t in Tp ])
    ax.plot(Tp, y, label='chi = ' + str(round(chi)) + '%, si = ' + str(int(round(hw2f.sigma[1]*1e4))) + 'bp' )
ax.legend()
ax.set_xlabel('T_p - T')
ax.set_ylabel('a(Tp) * An(0) / P(0,Tp)')
ax.set_title('shortVol = ' + str(round(shortVol*1e4)) + 'bp')

# now we start looking at the 2-factor model...

# we set up a base model; initially with equal volatility on both factors
chi   = np.array([ 0.20, 0.05 ])
sigma = np.array([ 0.02, 0.01 ])
rho   = -0.95
model2f  = scaleModel(HullWhiteModel2F(hYts,chi,sigma,rho),vol=0.01, T=T, dT=1.0/365.0)
h2fModel = Hw2fTsrModel(model2f,T,dT)

# we scale the 1-factor model to be consistent to 2-factor model
model1f  = scaleModel(h1fModel.model,vol=0.01, T=T, dT=1.0/365.0)
h1fModel = Hw2fTsrModel(model1f,T,dT)

# we output the model parameters for documentation
print('1-factor HJM:')
printModel(model1f)
print('2-factor HJM:')
printModel(model2f)

# let's have a look at the model-implied volatilities
terms  = np.array([ 1.0/365, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 ])
vols1f = np.array([ model1f.fwdYieldVolatility(T,deltaT)*1.0e+4 for deltaT in terms ])
vols2f = np.array([ model2f.fwdYieldVolatility(T,deltaT)*1.0e+4 for deltaT in terms ])
ax = fig.add_subplot(4,2,3)
ax.plot(terms, vols1f, label='1-factor HJM')
ax.plot(terms, vols2f, label='2-factor HJM')
ax.legend()
ax.set_xlabel('dT')
ax.set_ylabel('vol[F(T,T+dT)] (bp)')
ax.set_title('1F: chi = ' + str(int(round(model1f.chi[1]*1e2)))   + '%, '  + \
               'si = ' + str(int(round(model1f.sigma[1]*1e4))) + 'bp; ' )


# we have a first comparison of models...
y2  = np.array([ h1fModel.convexityAdjustmentFactor(T+t) for t in Tp ])
y3  = np.array([ h2fModel.convexityAdjustmentFactor(T+t) for t in Tp ])
# we plot results...
ax = fig.add_subplot(4,2,4)
ax.plot(Tp, y1, 'b-', label='Bi-linear TSR')
ax.plot(Tp, y2, 'r-', label='1-factor HJM')
ax.plot(Tp, y3, 'g-', label='2-factor HJM')
ax.legend()
ax.set_xlabel('T_p - T')
ax.set_ylabel('a(Tp) * An(0) / P(0,Tp)')
ax.set_title( '2F: chi = [' + str(int(round(model2f.chi[0]*1e2)))  + ','    + \
                          str(int(round(model2f.chi[1]*1e2)))  + ']%, ' + \
               'si = [' + str(int(round(model2f.sigma[0]*1e4)))  + ','    + \
                          str(int(round(model2f.sigma[1]*1e4)))  + ']bp, ' + \
              'rho = ' + str(int(round(model2f.rho*1e2))) + '%')

h2fModel.plot()
h2fModel.plotTransformed()

# we check whether slope function is correctly set up
#for t in Tp:
#    print('t: '        + str(round(t,2)) + \
#          ', slope: '  + str(round(h2fModel.slope(T+t),2)) + \
#          ', slope2: ' + str(round(h2fModel.slope2(T+t),2)) + \
#          ', alpha: '  + str(round(h2fModel.alpha(h2fModel.fwdRate(),T+t),2)) )

# 2-factor volatility test case
# we analyse the impact of model volatility on the convexity adjustment

sigmas = [ 1, 200, 400]  # bp's
ax = fig.add_subplot(4,2,5)
ax.plot(Tp, y1, 'b-', label='Bi-linear TSR')
ax.plot(Tp, y3, 'g-', label='2-factor HJM')
for s in sigmas:
    lambda_ = sigma[0] / sigma[1]
    model = Hw2fTsrModel(HullWhiteModel2F(hYts,chi,np.array([ lambda_*s*1.0e-4, s*1.0e-4 ]),rho),T,dT)
    y = np.array([ model.convexityAdjustmentFactor(T+t) for t in Tp ])
    ax.plot(Tp, y, label='sigma2 = ' + str(s) + 'bp')
ax.legend()
ax.set_xlabel('T_p - T')
ax.set_ylabel('a(Tp) * An(0) / P(0,Tp)')
ax.set_title('chi = [' + str(int(round(chi[0]*1e2)))  + ','    + \
                          str(int(round(chi[1]*1e2)))  + ']%, ' + \
              'rho = ' + str(int(round(rho*1e2))) + '%')    
# again, no significant impact from volatility


# 2-factor mean reversion test case
# we analyse the impact of mean reversion on the 2-factor model CA
# we parametrise chi1 = chi2 + shift and look at chi2 and shift individually
# to avoid (small) impacts from volatility changes we resclale to 100bp vol

# test dependency on chi2 parameter
shortVol = 0.01  # 100bp vol for scaling
shift = 15  # in %
meanReversions = [ -30, -10, 10, 30 ]  # in %
ax = fig.add_subplot(4,2,6)
ax.plot(Tp, y1, 'b-', label='Bi-linear TSR')
for chi2 in meanReversions:
    hw2f = scaleModel(HullWhiteModel2F(hYts,np.array([ (chi2+shift)*1.0e-2, chi2*1.0e-2 ]),sigma,rho), vol=shortVol, T=T, dT=1.0/365.0)
    model = Hw2fTsrModel(hw2f,T,dT)
    y = np.array([ model.convexityAdjustmentFactor(T+t) for t in Tp ])
    ax.plot(Tp, y, label='chi2 = ' + str(round(chi2)) + '%, si2 = ' + str(int(round(hw2f.sigma[1]*1e4))) + 'bp' )
ax.legend()
ax.set_xlabel('T_p - T')
ax.set_ylabel('a(Tp) * An(0) / P(0,Tp)')
ax.set_title('Shift = ' + str(round(shift)) + ', shortVol = ' + str(int(round(shortVol*1e4))) + 'bp')

# test dependency on shift parameter
shortVol = 0.01  # 100bp vol for scaling
chi2 = 5  # in %
shifts = [ 1, 10, 25, 50 ]  # in %
ax = fig.add_subplot(4,2,7)
ax.plot(Tp, y1, 'b-', label='Bi-linear TSR')
for shift in shifts:
    hw2f = scaleModel(HullWhiteModel2F(hYts,np.array([ (chi2+shift)*1.0e-2, chi2*1.0e-2 ]),sigma,rho), vol=shortVol, T=T, dT=1.0/365.0)
    model = Hw2fTsrModel(hw2f,T,dT)
    y = np.array([ model.convexityAdjustmentFactor(T+t) for t in Tp ])
    ax.plot(Tp, y, label='shift = ' + str(round(shift)) + '%, si2 = ' + str(int(round(hw2f.sigma[1]*1e4))) + 'bp' )
ax.legend()
ax.set_xlabel('T_p - T')
ax.set_ylabel('a(Tp) * An(0) / P(0,Tp)')
ax.set_title('chi2 = ' + str(round(chi2)) + '%, shortVol = ' + str(int(round(shortVol*1e4))) + 'bp')

# 2-factor correlation test case
# we analyse the impact of correlation on the 2-factor model CA
# to avoid (small) impacts from volatility changes we resclale to 100bp vol

shortVol = 0.01  # 100bp vol for scaling
corrs = [ -95, -50, 0, 50, 95 ]
ax = fig.add_subplot(4,2,8)
ax.plot(Tp, y1, 'b-', label='Bi-linear TSR')
for corr in corrs:
    hw2f = scaleModel(HullWhiteModel2F(hYts,chi,sigma,corr*1.0e-2), vol=shortVol, T=T, dT=1.0/365.0)
    model = Hw2fTsrModel(hw2f,T,dT)
    y = np.array([ model.convexityAdjustmentFactor(T+t) for t in Tp ])
    ax.plot(Tp, y, label='rho = ' + str(round(corr)) + '%, si2 = ' + str(int(round(hw2f.sigma[1]*1e4))) + 'bp' )
ax.legend()
ax.set_xlabel('T_p - T')
ax.set_ylabel('a(Tp) * An(0) / P(0,Tp)')
ax.set_title('chi = [' + str(int(round(chi[0]*1e2)))  + ','    + \
                          str(int(round(chi[1]*1e2)))  + ']%, ' )


plt.tight_layout()
plt.savefig('TSR.png')
plt.show()
exit()


