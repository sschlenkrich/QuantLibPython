import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__),'QuantLibWrapper'))

import pandas
import numpy as np
from scipy.optimize import least_squares

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3d plotting
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from QuantLibWrapper import YieldCurve, HullWhiteModel, createSwaption


flatCurve = YieldCurve(['70y'],[0.03])
discCurve = flatCurve
projCurve = flatCurve

atmVols = pandas.read_csv('swaptionATMVols.csv', sep=';', index_col=0 )

meanReversion    = 0.1
volatilityTimes  = np.array([ 2.0,   5.0,   10.0,   20.0   ])
volatilityValues = np.array([ 0.005, 0.005,  0.005,  0.005 ])
hwModel          = HullWhiteModel(discCurve,meanReversion,volatilityTimes,volatilityValues)

# calibration targets
expiries  = np.array([ 2, 5, 10, 20 ])
swapterms = np.array([ 2, 5, 10, 20 ])
marketVols = np.zeros([len(expiries),len(swapterms)])
swaptions = []
for i in range(expiries.shape[0]):
    swaptionLine = []
    for j in range(swapterms.shape[0]):
        marketVols[i][j] = atmVols.loc[str(expiries[i])+'y',str(swapterms[j])+'y']
        swaptionLine.append(createSwaption(str(expiries[i])+'y',str(swapterms[j])+'y',discCurve,projCurve))
    swaptions.append(swaptionLine)

def objective(sigma):
    hwModel = HullWhiteModel(discCurve,meanReversion,volatilityTimes,sigma)
    modelVols = np.zeros(expiries.shape[0]*swapterms.shape[0])
    for i in range(expiries.shape[0]):
        for j in range(swapterms.shape[0]):
            modelVols[i*swapterms.shape[0]+j] = swaptions[i][j].npvHullWhite(hwModel,'v')
            modelVols[i*swapterms.shape[0]+j] -= marketVols[i][j]
    return modelVols        

#opt = least_squares(objective,volatilityValues,bounds=(0.1*volatilityValues,10*volatilityValues))
opt = least_squares(objective,volatilityValues,method='lm')
print(opt.x)

hwModel = HullWhiteModel(discCurve,meanReversion,volatilityTimes,opt.x)


# derive full model vols
expiries  = np.array([ (1*k+1) for k in range(30) ])   # in years
swapterms = np.array([ (1*k+1) for k in range(30) ])   # in years relative to expiry
vols = np.zeros([len(expiries),len(swapterms)])
for expiry in expiries:
    for swapterm in swapterms:
        swaption = createSwaption(str(expiry)+'y',str(swapterm)+'y',discCurve,projCurve)
        vols[expiry-1][swapterm-1] = swaption.npvHullWhite(hwModel,'v')
    print('.', end='', flush=True)
print('')  # newline

# ATM swaption
fig = plt.figure(figsize=(6, 4))
ax = fig.gca(projection='3d')
X, Y = np.meshgrid(expiries, swapterms)
surf = ax.plot_surface(X, Y, atmVols.values*10000, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.xaxis.set_major_formatter(FormatStrFormatter('%2.0f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%2.0f'))
ax.zaxis.set_major_formatter(FormatStrFormatter('%2.0f'))
ax.set_xlim(0, 30)
ax.set_ylim(0, 30)
#ax.set_zlim(50, 150)
ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
ax.set_yticks([0, 5, 10, 15, 20, 25, 30])
ax.set_xlabel('Expiries (y)')
ax.set_ylabel('Swap terms (y)')
ax.set_zlabel('Market-implied normal volatility (bp)')
# fig.colorbar(surf, shrink=0.5, aspect=5)
plt.title('Market ATM Volatilities')

# vol variance
fig = plt.figure(figsize=(6, 4))
ax = fig.gca(projection='3d')
X, Y = np.meshgrid(expiries, swapterms)
surf = ax.plot_surface(X, Y, (atmVols.values-vols)*10000, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.xaxis.set_major_formatter(FormatStrFormatter('%2.0f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%2.0f'))
ax.zaxis.set_major_formatter(FormatStrFormatter('%2.0f'))
ax.set_xlim(0, 30)
ax.set_ylim(0, 30)
ax.set_zlim(-40, 30)
ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
ax.set_yticks([0, 5, 10, 15, 20, 25, 30])
ax.set_xlabel('Expiries (y)')
ax.set_ylabel('Swap terms (y)')
ax.set_zlabel('[Market-Model] normal volatility (bp)')
# fig.colorbar(surf, shrink=0.5, aspect=5)
plt.title('a = %4.2f' % meanReversion)


plt.show()


