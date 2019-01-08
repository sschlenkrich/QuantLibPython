import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__),'QuantLibWrapper'))

import numpy as np
import matplotlib.pyplot as plt
import pandas

from QuantLibWrapper import SabrModel, MCSimulation

# SabrModel( S(t), T, alpha, beta, nu, rho )
model1 = SabrModel(0.05,1.0,0.0100,0.0001,0.0001,0.0,shift=0.1)
model2 = SabrModel(0.05,1.0,0.0450,0.5000,0.0001,0.0,shift=0.1)
model3 = SabrModel(0.05,1.0,0.0405,0.5000,0.5000,0.0,shift=0.1)
model4 = SabrModel(0.05,1.0,0.0420,0.5000,0.5000,0.7,shift=0.1)
# ATM calibration
print(model1.calibrateATM(0.01), model2.calibrateATM(0.01), model3.calibrateATM(0.01), model4.calibrateATM(0.01))
# Strikes
strikes = [ (i+1)/1000 for i in range(100) ]
# implied volatility
vols1 = [model1.normalVolatility(strike) for strike in strikes]
vols2 = [model2.normalVolatility(strike) for strike in strikes]
vols3 = [model3.normalVolatility(strike) for strike in strikes]
vols4 = [model4.normalVolatility(strike) for strike in strikes]
# implied density
dens1 = [model1.density(strike) for strike in strikes]
dens2 = [model2.density(strike) for strike in strikes]
dens3 = [model3.density(strike) for strike in strikes]
dens4 = [model4.density(strike) for strike in strikes]

plt.figure()
plt.plot(strikes,vols1, 'b-', label='Normal')
plt.plot(strikes,vols2, 'r-', label='CEV')
plt.plot(strikes,vols3, 'g-', label='CEV+SV')
plt.plot(strikes,vols4, 'y-', label='CEV+SV+Corr')
plt.legend()
plt.xlabel('Strike')
plt.ylabel('Normal Volatility')

plt.figure()
plt.plot(strikes,dens1, 'b-', label='Normal')
plt.plot(strikes,dens2, 'r-', label='CEV')
plt.plot(strikes,dens3, 'g-', label='CEV+SV')
plt.plot(strikes,dens4, 'y-', label='CEV+SV+Corr')
plt.legend()
plt.xlabel('Rate')
plt.ylabel('Density')

plt.show()

table = pandas.DataFrame( [ strikes, vols1, vols2, vols3, vols4 ] ).T
table.columns = [ 'Strikes', 'Normal', 'CEV', 'CEV+SV', 'CEV+SV+Corr' ]
# print(table)
table.to_csv('SABRVolsAnalytic.csv')

# MC simulation
print('Start MC simulation...')
times  = np.array([k*0.01 for k in range(501)])
nPaths = 10000
mcSim1 = MCSimulation(model1,times,nPaths)
print('.')
mcSim2 = MCSimulation(model2,times,nPaths)
print('.')
mcSim3 = MCSimulation(model3,times,nPaths)
print('.')
mcSim4 = MCSimulation(model4,times,nPaths)
print('Done.')
# payoffs and normal vols
print('Start MC payoff calculation...')
mcStrikes = np.array([ (i+1)/100 for i in range(10) ])
mcVols1   = model1.monteCarloImpliedNormalVol(mcSim1,mcStrikes)
print('.')
mcVols2   = model2.monteCarloImpliedNormalVol(mcSim2,mcStrikes)
print('.')
mcVols3   = model3.monteCarloImpliedNormalVol(mcSim3,mcStrikes)
print('.')
mcVols4   = model4.monteCarloImpliedNormalVol(mcSim4,mcStrikes)
print('Done.')

# output
plt.figure()
#
plt.plot(strikes,vols1, 'b-', label='Normal')
plt.plot(mcStrikes,mcVols1, 'b*')
#
plt.plot(strikes,vols2, 'r-', label='CEV')
plt.plot(mcStrikes,mcVols2, 'r*')
#
plt.plot(strikes,vols3, 'g-', label='CEV+SV')
plt.plot(mcStrikes,mcVols3, 'g*')
#
plt.plot(strikes,vols4, 'y-', label='CEV+SV+Corr')
plt.plot(mcStrikes,mcVols4, 'y*')
#
plt.legend()
plt.xlabel('Strike')
plt.ylabel('Normal Volatility')
plt.xlim((0.0, 0.101))
plt.ylim((0.005, 0.025))
plt.show()

table = pandas.DataFrame( [ mcStrikes, mcVols1, mcVols2, mcVols3, mcVols4 ] ).T
table.columns = [ 'Strikes', 'Normal', 'CEV', 'CEV+SV', 'CEV+SV+Corr' ]
# print(table)
table.to_csv('SABRVolsMonteCarlo.csv')
