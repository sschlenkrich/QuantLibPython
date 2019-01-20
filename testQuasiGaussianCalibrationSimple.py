import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__),'QuantLibWrapper'))

from QuantLibWrapper import SwaptionVolatility, BachelierImpliedVol, \
     ModelSmile, McSimSmile, MarketSmile, Smiles

import QuantLib as ql

import matplotlib.pyplot as plt
import numpy as np

hYts = ql.YieldTermStructureHandle(
           ql.FlatForward(ql.Settings.getEvaluationDate(ql.Settings.instance()),
                          0.03,ql.Actual365Fixed()))

d     = 3

times = [   5.0,   10.0,   15.0,   20.0,   ]
sigma = [ [ 0.005,  0.005,  0.005,  0.001, ],
          [ 0.005,  0.005,  0.005,  0.001, ],
          [ 0.005,  0.005,  0.005,  0.001, ] ]
slope = [ [ 0.10,   0.10,   0.10,   0.10,  ],
          [ 0.10,   0.10,   0.10,   0.10,  ], 
          [ 0.10,   0.10,   0.10,   0.10,  ] ]
curve = [ [ 0.00,   0.00,   0.00,   0.00,  ],
          [ 0.00,   0.00,   0.00,   0.00,  ], 
          [ 0.00,   0.00,   0.00,   0.00,  ] ]
eta   = [   0.30,   0.30,   0.30,   0.30,   ]

delta = [   2.0,  7.0, 15.0  ]
chi   = [   0.01, 0.07, 0.15 ]
Gamma = [ [ 1.00,  0.70, 0.50 ],
          [ 0.70,  1.00, 0.70 ],
          [ 0.50,  0.70, 1.00 ] ]
theta = 0.1

qgModel = ql.QuasiGaussianModel(hYts,d,times,sigma,slope,curve,eta,delta,chi,Gamma,theta)

floatTimes    = [ 10.0, 15.0 ] 
floatWeights  = [  1.0, -1.0 ]
fixedTimes    = [ 11.0, 12.0, 13.0, 14.0, 15.0 ]
fixedWeights  = [  1.0,  1.0,  1.0,  1.0,  1.0 ]
modelTimes    = [ 0.1*float(k) for k in range(101) ]

swModel = ql.QGSwaprateModel(qgModel, floatTimes,floatWeights,fixedTimes,fixedWeights,modelTimes,False)

plt.plot(modelTimes,[ swModel.slope(t)*1e2 for t in modelTimes ])
plt.xlabel('time t')
plt.ylabel('slope_S(t) (%)')
plt.show()

avModel = ql.QGAverageSwaprateModel(swModel)
print('sigma: ' + str(avModel.sigma()))
print('slope: ' + str(avModel.slope()))
print('eta:   ' + str(avModel.eta()))

strike         = 0.03
callOrPut      = 1        # call option
accuracy       = 1.0e-6   # for numerical Heston model
maxEvaluations = 1000     # for numerical Heston model

print('E[max(S-K,0)]: ' + str(avModel.vanillaOption(strike,callOrPut,accuracy,maxEvaluations)))

swapTerms = [ ql.Period(str(p)+'y') for p in [2, 5, 10, 20] ]
expiTerms = [ ql.Period(str(p)+'y') for p in [1, 2, 3, 5, 7, 10, 15] ]
valsMatrx = ql.Matrix(len(expiTerms),len(swapTerms))
for i in range(len(expiTerms)):
    for j in range(len(swapTerms)):
        valsMatrx[i][j] = 0.005
atmVTS = ql.SwaptionVolatilityMatrix(ql.TARGET(),ql.Following,expiTerms,swapTerms,valsMatrx,ql.Actual365Fixed(),True,ql.Normal)

indices = [ ql.EuriborSwapIsdaFixA(ql.Period( '2y'),hYts,hYts),
            ql.EuriborSwapIsdaFixA(ql.Period('10y'),hYts,hYts) ]

modelTimesStepSize = 0.25   # time grid size in swap rate models
useExpectedXY      = False  # passed on to swap rate model
endCrit = ql.EndCriteria(100,10,1.0e-4,1.0e-4,1.0e-4)  # for optimizer

sigmaMax     = 0.01  # maximum sigma parameter in optimisation
slopeMax     = 0.3   # maximum slope parameter in optimisation
etaMax       = 0.5   # maximum vol-of-vol parameter in optim.
sigmaWeight  = 1.0   # put emphasis on ATM fit
slopeWeight  = 0.0   # put emphasis on skew fit
etaWeight    = 0.0   # put emphasis on smile fit
penaltySigma = 0.1   # force similar sigma parameters per factor
penaltySlope = 0.01  # force similar slope parameters per factor

qgCalib = ql.QGCalibrator(qgModel,ql.SwaptionVolatilityStructureHandle(atmVTS),
              indices,modelTimesStepSize,useExpectedXY,sigmaMax,slopeMax,etaMax,
              sigmaWeight,slopeWeight,etaWeight,penaltySigma,penaltySlope,
              endCrit)

print(qgCalib.debugLog())
caModel = ql.QuasiGaussianModel(qgCalib.calibratedModel())

print('Sigma:')
print(np.array(caModel.sigma()))
print('Slope:')
print(np.array(caModel.slope()))
print('Eta:')
print(np.array(caModel.eta()))

expiries = [ '5y', '10y', '15y', '20y' ]
swpterms = [ '2y', '5y', '10y']
Smiles(atmVTS,caModel,None,expiries,swpterms,hYts,hYts)
