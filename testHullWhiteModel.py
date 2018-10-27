import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__),'QuantLibWrapper'))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import pandas

import QuantLib as ql
import QuantLibWrapper.YieldCurve as yc

from QuantLibWrapper.HullWhiteModel import HullWhiteModel
from QuantLibWrapper import MCSimulation, Payoffs

# yield curves

flatCurve = yc.YieldCurve(['30y'],[0.03])

terms = [    '1y',    '2y',    '3y',    '4y',    '5y',    '6y',    '7y',    '8y',    '9y',   '10y',   '12y',   '15y',   '20y',   '25y',   '30y' ] 
rates = [ 2.70e-2, 2.75e-2, 2.80e-2, 3.00e-2, 3.36e-2, 3.68e-2, 3.97e-2, 4.24e-2, 4.50e-2, 4.75e-2, 4.75e-2, 4.70e-2, 4.50e-2, 4.30e-2, 4.30e-2 ] 
fwdRateYC = yc.YieldCurve(terms,rates)

# Hull-White model

meanReversion    = 0.05
volatilityTimes  = [  1.0 , 2.0 , 5.0 , 10.0  ]
volatilityValues = [  0.01, 0.01, 0.01,  0.01 ]

hwModel          = HullWhiteModel(fwdRateYC,meanReversion,volatilityTimes,volatilityValues)

# first we analyse future yield curves

states = [ -0.10, -0.05, 0.0, 0.05, 0.1 ]
futureTime = 5.0
stepsize   = 1.0/365

times       = [ k*stepsize for k in range(int(round(30.0/stepsize,0))+1) ]
forwardRate = [ hwModel.yieldCurve.forwardRate(T) for  T in times ]
fig = plt.figure(figsize=(4, 6))
ax = fig.gca()
plt.plot(times,forwardRate, label='f(0,T)')
times = [ k*stepsize+futureTime for k in range(int(round(30.0/stepsize,0))+1) ]
for x in states:
    forwardRate = [ hwModel.forwardRate(futureTime,x,T) for T in times ]
    plt.plot(times,forwardRate, label='x='+str(x))

plt.legend()
plt.xlabel('Maturity T')
plt.ylabel('Forward rate f(t,T)')
ax.yaxis.set_major_formatter(FormatStrFormatter('%4.2f'))
plt.title('a = %4.2f' % meanReversion)
plt.show()

# now we analyse coupon bond opition pricing

# coupon bond option details
 
exercise  = 12.0
payTimes  = [ 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 20.0 ]
cashFlows = [ 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03,  1.0 ]

# analytical coupon bond option prices

strikes   = [ 0.70+0.01*k for k in range(41) ]
calls     = [ hwModel.couponBondOption(exercise,payTimes,cashFlows,strike,1.0)  for strike in strikes ]
puts      = [ hwModel.couponBondOption(exercise,payTimes,cashFlows,strike,-1.0) for strike in strikes ]

# cross check with MC prices

# first we simulate all paths

times  = np.array([k*0.1 for k in range(121)])
nPaths = 100
mcSim  = MCSimulation(hwModel,times,nPaths)

# then calculate the payoffs

mcStrikes   = [ 0.70+0.05*k for k in range(9) ]
bondPayoff  = Payoffs.CouponBond(hwModel,exercise,payTimes,cashFlows)
callPayoffs = [ Payoffs.Pay(Payoffs.VanillaOption(bondPayoff,strike,1.0) , exercise) for strike in mcStrikes ] 
putPayoffs  = [ Payoffs.Pay(Payoffs.VanillaOption(bondPayoff,strike,-1.0), exercise) for strike in mcStrikes ] 
callsMC     = [ mcSim.npv(payoff) for payoff in callPayoffs ]
putsMC      = [ mcSim.npv(payoff) for payoff in putPayoffs  ]

# gather results

plt.plot(strikes,calls,          label='analytic call')
plt.plot(strikes,puts,           label='analytic put')
plt.plot(mcStrikes,callsMC, '*', label='MC call')
plt.plot(mcStrikes,putsMC,  '*', label='MC call')
plt.legend()
plt.xlabel('Strike')
plt.ylabel('Bond option price')
plt.show()
