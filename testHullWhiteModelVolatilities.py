import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__),'QuantLibWrapper'))

import numpy as np
from scipy.optimize import brentq

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3d plotting
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import pandas

import QuantLib as ql

from QuantLibWrapper import YieldCurve, HullWhiteModel, Swap, Swaption, createSwaption

# yield curves

flatCurve = YieldCurve(['30y'],[0.03])

terms = [    '1y',    '2y',    '3y',    '4y',    '5y',    '6y',    '7y',    '8y',    '9y',   '10y',   '12y',   '15y',   '20y',   '25y',   '30y', '50y'   ] 
rates = [ 2.70e-2, 2.75e-2, 2.80e-2, 3.00e-2, 3.36e-2, 3.68e-2, 3.97e-2, 4.24e-2, 4.50e-2, 4.75e-2, 4.75e-2, 4.70e-2, 4.50e-2, 4.30e-2, 4.30e-2, 4.30e-2 ] 
rates2 = [ r+0.005 for r in rates ]

discCurve = YieldCurve(terms,rates)
projCurve = YieldCurve(terms,rates2)

# Hull-White model mean reversion
meanReversion    = -0.05

# first we have a look at the model-implied volatility smile

fig = plt.figure(figsize=(4, 6))
atmRate = createSwaption('10y','10y',discCurve,projCurve).fairRate()
relStrikes = [ -0.03+1e-3*k for k in range(61)]
hwVols = [ 0.0050, 0.0075, 0.0100, 0.0125 ]
for hwVol in hwVols:
    hwModel = HullWhiteModel(discCurve,meanReversion,[30.0],[hwVol])
    normalVols = [ createSwaption('10y','10y',discCurve,projCurve,atmRate+strike).npvHullWhite(hwModel,'v')*1e+4
                   for strike in relStrikes ]
    plt.plot(relStrikes, normalVols, label='a='+str(meanReversion)+', sigma_r='+str(hwVol))
plt.ylim(0,250)
plt.legend()
plt.xlabel('Strike (relative to ATM)')
plt.ylabel('Model-implied normal volatility (bp)')
plt.show()

# since Hull White smile is essentially flat we now consider ATM swaptions
# we set up ATM swaptions on a grid of expiry and swap terms
expiries  = np.array([ (1*k+1) for k in range(20) ])   # in years
swapterms = np.array([ (1*k+1) for k in range(20) ])   # in years relative to expiry
vols = np.zeros([len(expiries),len(swapterms)])

# for this test we want to keep the model fixed to a particular vol point
# if we change mean reversion
def objective(sigma):
    tmpModel = HullWhiteModel(discCurve,meanReversion,[30.0],[sigma])
    return createSwaption('10y','10y',discCurve,projCurve).npvHullWhite(tmpModel,'v') - 0.01  # we calibrate to 100bp 10y-10y vols
sigma = brentq(objective, 0.5e-2, 0.5e-1, xtol=1.0e-8)
print('sigma_r: '+str(sigma))
hwModel = HullWhiteModel(discCurve,meanReversion,[30.0],[sigma])

for expiry in expiries:
    for swapterm in swapterms:
        swaption = createSwaption(str(expiry)+'y',str(swapterm)+'y',discCurve,projCurve)
        vols[expiry-1][swapterm-1] = swaption.npvHullWhite(hwModel,'v')*1e+4
    print('.', end='', flush=True)
print('')  # newline

# 3d surface plotting, see https://matplotlib.org/gallery/mplot3d/surface3d.html#sphx-glr-gallery-mplot3d-surface3d-py
fig = plt.figure(figsize=(4, 6))
ax = fig.gca(projection='3d')
X, Y = np.meshgrid(expiries, swapterms)
surf = ax.plot_surface(X, Y, vols, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.xaxis.set_major_formatter(FormatStrFormatter('%2.0f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%2.0f'))
ax.zaxis.set_major_formatter(FormatStrFormatter('%2.0f'))
ax.set_xlim(0, 20)
ax.set_ylim(0, 20)
ax.set_zlim(50, 150)
ax.set_xticks([0, 5, 10, 15, 20])
ax.set_yticks([0, 5, 10, 15, 20])
ax.set_xlabel('Expiries (y)')
ax.set_ylabel('Swap terms (y)')
ax.set_zlabel('Model-implied normal volatility (bp)')
# fig.colorbar(surf, shrink=0.5, aspect=5)
plt.title('a = %4.2f' % meanReversion)
plt.show()
