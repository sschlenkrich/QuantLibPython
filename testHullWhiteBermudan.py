import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__),'QuantLibWrapper'))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import pandas

import QuantLib as ql
import QuantLibWrapper.YieldCurve as yc

from QuantLibWrapper import HullWhiteModel, MCSimulation, Payoffs, BermudanOption, \
                            DensityIntegrationWithBreakEven, SimpsonIntegration, \
                            HermiteIntegration, CubicSplineExactIntegration, \
                            PDESolver, AMCSolver 

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
#method           = SimpsonIntegration(hwModel,101,5)
#method           = DensityIntegrationWithBreakEven(SimpsonIntegration(hwModel,101,5))
#method           = HermiteIntegration(hwModel,10,101,5)
#method           = CubicSplineExactIntegration(hwModel,101,5)
#method           = DensityIntegrationWithBreakEven(CubicSplineExactIntegration(hwModel,101,5))
#method           = PDESolver(hwModel,11,2.0,0.5,1.0/12.0,0.0)
#method           = PDESolver(hwModel,11,2.0,0.5,1.0/12.0)

times  = np.linspace(0.0,19.0,20)
nPaths = 10000
mcSim  = MCSimulation(hwModel,times,nPaths)
method = AMCSolver(mcSim,20.0,2,0.1)

# now we test coupon bond opition pricing

# coupon bond option details
 
exercise  = 12.0
payTimes  = [ 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 20.0 ]
cashFlows = [ -1.0, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03,  1.0 ]

print('Bond option with zero strike: ' + str(hwModel.couponBondOption(exercise,payTimes,cashFlows,0.0,1.0)))
print('Bond option with unit strike: ' + str(hwModel.couponBondOption(exercise,payTimes[1:],cashFlows[1:],1.0,1.0)))

expiryTimes = np.array([ 2.0, 6.0, 12.0])
underlyings = [ Payoffs.Zero(), Payoffs.Zero(), Payoffs.CouponBond(hwModel,12.0,payTimes,cashFlows) ]
berm = BermudanOption(expiryTimes, underlyings, method)
print('Pseudo-Bermudan opt. (3 ex.): ' + str(berm.npv()))
payoff = Payoffs.Pay(Payoffs.VanillaOption(Payoffs.CouponBond(hwModel,12.0,payTimes,cashFlows),0.0,1.0),12.0)
print('MC bond option              : ' + str(mcSim.npv(payoff)))  

# test Bermudan bond option pricing

expiryTimes = np.array([ 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0])
underlyings = []
print('European bond option prices 12y - 19y...')
for k in range(8):
    pTimes = [payTimes[k]] + payTimes[(k+1):]
    cFlows = [-1.0 ] + cashFlows[(k+1):]
    print(hwModel.couponBondOption(payTimes[k],pTimes,cFlows,0.0,1.0))
    underlyings.append(Payoffs.CouponBond(hwModel,payTimes[k],pTimes,cFlows))
berm = BermudanOption(expiryTimes, underlyings, method)
print(berm.npv())


