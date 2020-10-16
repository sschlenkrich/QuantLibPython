
import numpy as np
from scipy.optimize import brentq

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3d plotting
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import pandas

import QuantLib as ql

from QuantLibWrapper.YieldCurve import YieldCurve 
from QuantLibWrapper.HullWhiteModel import HullWhiteModel
from QuantLibWrapper.Swap import Swap
from QuantLibWrapper.Swaption import Swaption, createSwaption
from QuantLibWrapper.BermudanSwaption import BermudanSwaption


# curves and vols

atmVols = pandas.read_csv('swaptionATMVols.csv', sep=';', index_col=0 )

terms = [    '1y',    '2y',    '3y',    '4y',    '5y',    '6y',    '7y',    '8y',    '9y',   '10y',   '12y',   '15y',   '20y',   '25y',   '30y', '50y'   ] 
rates = [ 2.70e-2, 2.75e-2, 2.80e-2, 3.00e-2, 3.36e-2, 3.68e-2, 3.97e-2, 4.24e-2, 4.50e-2, 4.75e-2, 4.75e-2, 4.70e-2, 4.50e-2, 4.30e-2, 4.30e-2, 4.30e-2 ] 
rates = [ 0.025 for r in rates ]
rates2 = [ r+0.005 for r in rates ]
discCurve = YieldCurve(terms,rates)
projCurve = YieldCurve(terms,rates2)
meanReversion = 0.05
normalVol = 0.01

# swaption(s)

maturity  = 20  # in years
swaptions = []
terms = []
inputVols = [] 
for k in range(1,maturity):
    expTerm = str(k)+'y'
    swpTerm = str(maturity-k)+'y'
    sigma   = 0.01   # atmVols.loc[expTerm,swpTerm]
    terms.append(expTerm+'-'+swpTerm)
    inputVols.append(sigma)
    swaptions.append( createSwaption(expTerm,swpTerm, discCurve, projCurve, 0.03, ql.VanillaSwap.Receiver, sigma ) )

bermudanSwaption = BermudanSwaption(swaptions,meanReversion)

table = pandas.DataFrame([ np.array(terms),
                           np.array(inputVols),
                           bermudanSwaption.model.volatilityTimes,
                           bermudanSwaption.model.volatilityValues,
                           bermudanSwaption.swaptionsNPV(),
                           bermudanSwaption.bondOptionsNPV()
                        ]).T
table.columns = [ 'Terms', 'InputVols', 'Times', 'Vols', 'SwptNPV', 'CboNpv' ]  

bermT = pandas.DataFrame([ [ 'Berm', bermudanSwaption.npv() ] ])
bermT.columns = [ 'Terms', 'CboNpv' ]  

restT = table.append(bermT,ignore_index=True, sort=False)
print(restT)

x = 4*np.linspace(0,19,20)
fig = plt.figure(figsize=(6, 4))
plt.bar(x,restT['CboNpv'],3.0)
plt.xticks(x, restT['Terms'], rotation='vertical')
plt.ylabel('option price')
plt.ylim(0,4000)

# plot short rate volatility
T = np.linspace(0,20,2001)
s = np.array([ bermudanSwaption.model.sigma(t)*10000 for t in T ])
fig = plt.figure(figsize=(6, 4))
plt.xticks(np.linspace(0,20,11))
plt.xlabel('time t')
plt.ylabel('short rate volatility sigma(t) (bp)')
plt.title('a = %4.2f' % meanReversion)
plt.ylim(0,160)
plt.plot(T,s)
plt.show()


# calculate short rate volatilities per mean reversion
fig = plt.figure(figsize=(7, 4))
T = np.linspace(0,20,2001)
for a in reversed([-0.05, -0.03, -0.01, 0.01, 0.03, 0.05, 0.07, 0.09, 0.11]):
    bermudanSwaption = BermudanSwaption(swaptions,a)  # just set up and calibrate
    s = np.array([ bermudanSwaption.model.sigma(t)*10000 for t in T ])
    plt.plot(T, s, label=str('a = %4.2f' % a))
plt.xticks(np.linspace(0,20,11))
plt.xlabel('time t')
plt.ylabel('short rate volatility sigma(t) (bp)')
plt.legend()
plt.xlim(0,27)
#plt.ylim(0,160)
plt.show()

exit()


# calculate switch value per mean reversion
maxEuropean = max([ swpt.npv() for swpt in swaptions ])
result = []
for a in [-0.05, -0.03, -0.01, 0.01, 0.03, 0.05, 0.07, 0.09, 0.11]:
    result.append([ a, BermudanSwaption(swaptions,a).npv()-maxEuropean ]  )

table = pandas.DataFrame(result, columns=['MeanReversion', 'SwitchOption'])
table.to_excel('SwitchOption-ITM.xls')


### read in aggregated results and plot data

table = pandas.read_excel('SwitchOption.xls')
print(table)

fig = plt.figure(figsize=(6, 4))
plt.plot(table['MeanReversion'],table['ITM'],label='ITM, f=1%')
plt.plot(table['MeanReversion'],table['ATM'],label='ATM, f=3%')
plt.plot(table['MeanReversion'],table['OTM'],label='OTM, f=5%')
plt.xlabel('Mean reversion a')
plt.ylabel('Switch Option value')
plt.ylim(0,600)
plt.xlim(-0.06,0.12)
plt.legend()

plt.show()
