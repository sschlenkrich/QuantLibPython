import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__),'QuantLibWrapper'))

import numpy as np
import pandas

import QuantLib as ql
from QuantLibWrapper import YieldCurve

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3d plotting
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


print(ql.buildinfo()[4])


terms = [    '1y',    '2y',    '3y',    '4y',    '5y',    '6y',    '7y',    '8y',    '9y',   '10y',   '12y',   '15y',   '20y',   '25y',   '30y', '50y'   ] 
rates = [ 2.70e-2, 2.75e-2, 2.80e-2, 3.00e-2, 3.36e-2, 3.68e-2, 3.97e-2, 4.24e-2, 4.50e-2, 4.75e-2, 4.75e-2, 4.70e-2, 4.50e-2, 4.30e-2, 4.30e-2, 4.30e-2 ] 
rates = [ 0.025 for r in rates ]
rates2 = [ r+0.005 for r in rates ]
discCurve = YieldCurve(terms,rates)
projCurve = YieldCurve(terms,rates2)

atmVols = pandas.read_csv('swaptionATMVols2.csv', sep=';', index_col=0 )
swapTerms = [ ql.Period(p) for p in atmVols.columns.values ]
expiTerms = [ ql.Period(p) for p in atmVols.index ] 

vals = ql.Matrix(atmVols.values.shape[0],atmVols.values.shape[1])
for i in range(atmVols.values.shape[0]):
    for j in range(atmVols.values.shape[1]):
        vals[i][j] = atmVols.values[i][j]
sw = ql.SwaptionVolatilityMatrix(ql.TARGET(),ql.Following,expiTerms,swapTerms,vals,ql.Actual365Fixed(),True,ql.Normal)
h = ql.SwaptionVolatilityStructureHandle(sw)


relStrikes = [  -0.0200,  -0.0100,  -0.0050,  -0.0025,   0.0000,   0.0025,   0.0050,   0.0100,   0.0200 ]

smile01x01 = [           0.002357, 0.001985, 0.002038, 0.002616, 0.003324, 0.004017, 0.005518, 0.008431 ]
smile3mx02 = [                     0.001653, 0.001269, 0.002250, 0.003431, 0.004493, 0.006528, 0.010423 ]
smile02x02 = [           0.003641, 0.003766, 0.003987, 0.004330, 0.004747, 0.005177, 0.006096, 0.008203 ]
smile01x05 = [ 0.003925, 0.004376, 0.004284, 0.004364, 0.004680, 0.005118, 0.005598, 0.006645, 0.008764 ]
smile05x05 = [ 0.005899, 0.005975, 0.006202, 0.006338, 0.006431, 0.006639, 0.006793, 0.007135, 0.007907 ]
smile3mx10 = [ 0.006652, 0.005346, 0.004674, 0.004583, 0.004850, 0.005431, 0.006161, 0.007743, 0.010880 ]
smile01x10 = [ 0.005443, 0.005228, 0.005271, 0.005398, 0.005600, 0.005879, 0.006203, 0.006952, 0.008603 ]
smile02x10 = [ 0.005397, 0.005492, 0.005685, 0.005821, 0.005971, 0.006167, 0.006367, 0.006818, 0.007840 ]
smile05x10 = [ 0.006096, 0.006234, 0.006427, 0.006541, 0.006622, 0.006821, 0.006946, 0.007226, 0.007875 ]
smile10x10 = [ 0.006175, 0.006353, 0.006485, 0.006582, 0.006602, 0.006850, 0.006923, 0.007097, 0.007495 ]
smile05x30 = [ 0.005560, 0.005660, 0.005792, 0.005871, 0.005958, 0.006147, 0.006233, 0.006458, 0.007048 ]

today = ql.Settings.getEvaluationDate(ql.Settings.instance())
S0 = 0.0
extrapolationRelativeStrike = relStrikes[-1] + 0.05
extrapolationSlope = 0.0

smile01 = []
smile02 = []
smile05 = []
smile10 = []
smile30 = []

smile01.append( ql.VanillaLocalVolModelSmileSection(ql.TARGET().advance(today,ql.Period('1y')), S0, relStrikes[1:], smile01x01, extrapolationRelativeStrike, extrapolationSlope) )
smile02.append( ql.VanillaLocalVolModelSmileSection(ql.TARGET().advance(today,ql.Period('3m')), S0, relStrikes[2:], smile3mx02, extrapolationRelativeStrike, extrapolationSlope) )
smile02.append( ql.VanillaLocalVolModelSmileSection(ql.TARGET().advance(today,ql.Period('2y')), S0, relStrikes[1:], smile02x02, extrapolationRelativeStrike, extrapolationSlope) )
smile05.append( ql.VanillaLocalVolModelSmileSection(ql.TARGET().advance(today,ql.Period('1y')), S0, relStrikes[0:], smile01x05, extrapolationRelativeStrike, extrapolationSlope) )
smile05.append( ql.VanillaLocalVolModelSmileSection(ql.TARGET().advance(today,ql.Period('5y')), S0, relStrikes[0:], smile05x05, extrapolationRelativeStrike, extrapolationSlope) )
smile10.append( ql.VanillaLocalVolModelSmileSection(ql.TARGET().advance(today,ql.Period('3m')), S0, relStrikes[0:], smile3mx10, extrapolationRelativeStrike, extrapolationSlope) )
smile10.append( ql.VanillaLocalVolModelSmileSection(ql.TARGET().advance(today,ql.Period('1y')), S0, relStrikes[0:], smile01x10, extrapolationRelativeStrike, extrapolationSlope) )
smile10.append( ql.VanillaLocalVolModelSmileSection(ql.TARGET().advance(today,ql.Period('2y')), S0, relStrikes[0:], smile02x10, extrapolationRelativeStrike, extrapolationSlope) )
smile10.append( ql.VanillaLocalVolModelSmileSection(ql.TARGET().advance(today,ql.Period('5y')), S0, relStrikes[0:], smile05x10, extrapolationRelativeStrike, extrapolationSlope) )
smile10.append( ql.VanillaLocalVolModelSmileSection(ql.TARGET().advance(today,ql.Period('10y')), S0, relStrikes[0:], smile10x10, extrapolationRelativeStrike, extrapolationSlope) )
smile30.append( ql.VanillaLocalVolModelSmileSection(ql.TARGET().advance(today,ql.Period('5y')), S0, relStrikes[0:], smile05x30, extrapolationRelativeStrike, extrapolationSlope) )

smileCollection = [ smile01,         
                    smile02,
                    smile05,
                    smile10,
                    smile30 ]
periods = [ ql.Period('1y'), 
            ql.Period('2y'), 
            ql.Period('5y'),
            ql.Period('10y'),
            ql.Period('30y') ]

index = ql.EuriborSwapIsdaFixA( ql.Period('1y'), ql.YieldTermStructureHandle(projCurve.yts), ql.YieldTermStructureHandle(discCurve.yts) )

volTS = ql.VanillaLocalVolSwaptionVTS(h,smileCollection,periods,index)

smile = smile10[4]
date  = smile.exerciseDate()
term  = ql.Period('10y')
S0    = index.clone(term).fixing(date)

strikes = np.linspace(-0.05, 0.05, 101)
vols1   = np.array([ smile.volatility(strike) for strike in strikes ])
vols2   = np.array([ volTS.volatility(date,term,strike+S0,True) for strike in strikes ])

#plt.figure()
#plt.plot(strikes, vols1, label='SmileSection')
#plt.plot(strikes, vols2, label='VolTS')
#plt.show()

#table = pandas.DataFrame([ strikes, vols1, vols2 ]).T
#print(table)

# vol surface for a given CMS rate

term  = ql.Period('2y')
swapIndex = index.clone(term)
expiries = np.linspace(0.25, 12.0, 48)
strikes = np.linspace(-0.03, 0.03, 61)
vols = np.zeros([ expiries.shape[0], strikes.shape[0] ])
for i in range(expiries.shape[0]):
    date = ql.TARGET().advance(today,ql.Period(int(expiries[i]*365),ql.Days),ql.Following)
    S0   = swapIndex.fixing(date)
    for j in range(strikes.shape[0]):
        vols[i][j] = volTS.volatility(date,term,strikes[j]+S0,True)*1e+4

# 3d surface plotting, see https://matplotlib.org/gallery/mplot3d/surface3d.html#sphx-glr-gallery-mplot3d-surface3d-py
fig = plt.figure(figsize=(6, 6))
ax = fig.gca(projection='3d')
X, Y = np.meshgrid(expiries, strikes*1.0e4, indexing='ij')
surf = ax.plot_surface(X, Y, vols, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.xaxis.set_major_formatter(FormatStrFormatter('%2.0f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%3.0f'))
ax.zaxis.set_major_formatter(FormatStrFormatter('%2.0f'))
ax.set_xlim(0, 11)
ax.set_ylim(-300, 300)
ax.set_zlim(50, 150)
#ax.set_xticks([0, 5, 10, 15, 20])
#ax.set_yticks([0, 5, 10, 15, 20])
ax.set_xlabel('Expiries (y)')
ax.set_ylabel('RelStrikes (bp)')
ax.set_zlabel('Model-implied normal volatility (bp)')
# fig.colorbar(surf, shrink=0.5, aspect=5)
plt.title('Term: ' + str(term))
plt.show()

# we calibrate a Quasi-Gaussian model to the volTS

d     = 2

times = [   2.0,   5.0,  10.0,  20.0    ]
sigma = [ [ 0.01,  0.01,  0.01,  0.01 ],
          [ 0.01,  0.01,  0.01,  0.01 ] ]
slope = [ [ 0.10,  0.10,  0.10,  0.10 ],
          [ 0.10,  0.10,  0.10,  0.10 ] ]
curve = [ [ 0.00,  0.00,  0.00,  0.00 ],
          [ 0.00,  0.00,  0.00,  0.00 ] ]
eta   = [   0.10,  0.10,  0.10,  0.10   ]

delta = [   1.0,  10.0  ]
chi   = [   0.01,  0.20 ]
Gamma = [ [ 1.00,  0.80 ],
          [ 0.80,  1.00 ] ]
theta = 0.1

qgModel = ql.QuasiGaussianModel(ql.YieldTermStructureHandle(discCurve.yts),
              d,times,sigma,slope,curve,eta,delta,chi,Gamma,theta)

indices = [ index.clone(ql.Period('2y')), index.clone(ql.Period('10y')) ]

endCrit = ql.EndCriteria(100,10,1.0e-4,1.0e-4,1.0e-4)

qgCalib = ql.QGCalibrator(qgModel,ql.SwaptionVolatilityStructureHandle(volTS),indices,
              0.25,False,0.015,0.3,0.2,1.0,1.0,1.0,0.1,0.1,endCrit)

print(qgCalib.debugLog())

caModel = ql.QuasiGaussianModel(qgCalib.calibratedModel())

print(np.array(caModel.sigma()))
print(np.array(caModel.slope()))
print(np.array(caModel.eta()))


