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

times = [ 5.0+5*k for k in range(4) ]
sigma = [ [ 0.005 for t in times ],
          [ 0.005 for t in times ],
          [ 0.005 for t in times ] ]
slope = [ [ 0.08  for t in times  ],
          [ 0.08  for t in times  ], 
          [ 0.08  for t in times  ] ]
curve = [ [ 0.00  for t in times  ],
          [ 0.00  for t in times  ], 
          [ 0.00  for t in times  ] ]
eta   = [   0.30  for t in times    ]

delta = [   2.0,  7.0, 15.0  ]
chi   = [   0.05, 0.25, 0.75 ]
Gamma = [ [ 1.00,  0.50, 0.00 ],
          [ 0.50,  1.00, 0.50 ],
          [ 0.00,  0.50, 1.00 ] ]
theta = 0.1

qgModel = ql.QuasiGaussianModel(hYts,d,times,sigma,slope,curve,eta,delta,chi,Gamma,theta)

floatTimes    = [ 10.0, 15.0 ] 
floatWeights  = [  1.0, -1.0 ]
fixedTimes    = [ 11.0, 12.0, 13.0, 14.0, 15.0 ]
fixedWeights  = [  1.0,  1.0,  1.0,  1.0,  1.0 ]
modelTimes    = [ 0.1*float(k) for k in range(101) ]

swModelUseExpFalse = ql.QGSwaprateModel(qgModel, floatTimes,floatWeights,fixedTimes,fixedWeights,modelTimes,False)
swModelUseExpTrue  = ql.QGSwaprateModel(qgModel, floatTimes,floatWeights,fixedTimes,fixedWeights,modelTimes,True)

plt.figure()
plt.plot(modelTimes,[ swModelUseExpFalse.sigma(t)*1e4 for t in modelTimes ], label='useExpectedXY=False')
plt.plot(modelTimes,[ swModelUseExpTrue.sigma(t)*1e4 for t in modelTimes ], label='useExpectedXY=True')
plt.xlabel('time t')
plt.ylabel('sigma_S(t) (bp)')
plt.legend()

plt.figure()
plt.plot(modelTimes,[ swModelUseExpFalse.slope(t)*1e2 for t in modelTimes ], label='useExpectedXY=False')
plt.plot(modelTimes,[ swModelUseExpTrue.slope(t)*1e2 for t in modelTimes ], label='useExpectedXY=True')
plt.xlabel('time t')
plt.ylabel('slope_S(t) (%)')
plt.legend()

plt.show()

print('useExpectedXY=False:')
avModel = ql.QGAverageSwaprateModel(swModelUseExpFalse)
print('sigma: ' + str(avModel.sigma()))
print('slope: ' + str(avModel.slope()))
print('eta:   ' + str(avModel.eta()))

print('useExpectedXY=True:')
avModel = ql.QGAverageSwaprateModel(swModelUseExpTrue)
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

indices = [ ql.EuriborSwapIsdaFixA(ql.Period('5y'),hYts,hYts) ]

modelTimesStepSize = 0.25   # time grid size in swap rate models
useExpectedXY      = False  # passed on to swap rate model
endCrit = ql.EndCriteria(1000,10,1.0e-6,1.0e-6,1.0e-6)  # for optimizer

sigmaMax     = 0.01  # maximum sigma parameter in optimisation
slopeMax     = 0.3   # maximum slope parameter in optimisation
etaMax       = 0.5   # maximum vol-of-vol parameter in optim.
sigmaWeight  = 1.0   # put emphasis on ATM fit
slopeWeight  = 0.0   # put emphasis on skew fit
etaWeight    = 0.0   # put emphasis on smile fit
penaltySigma = 0.1   # force similar sigma parameters per factor
penaltySlope = 0.01  # force similar slope parameters per factor
# only for MC calibration
monteCarloStepSize = 0.5
monteCarloPaths    = 1000
curveMax           = 0.5
curveWeight        = 0.0
penaltyCurve       = 0.01

#qgCalib = ql.QGCalibrator(qgModel,ql.SwaptionVolatilityStructureHandle(atmVTS),
#              indices,modelTimesStepSize,useExpectedXY,sigmaMax,slopeMax,etaMax,
#              sigmaWeight,slopeWeight,etaWeight,penaltySigma,penaltySlope,
#              endCrit)
#simul = None

qgCalib = ql.QGMonteCarloCalibrator(qgModel,ql.SwaptionVolatilityStructureHandle(atmVTS),indices,
              monteCarloStepSize,monteCarloPaths,sigmaMax,slopeMax,curveMax,
              sigmaWeight,slopeWeight,curveWeight,penaltySigma,penaltySlope,penaltyCurve,
              endCrit)
simul = qgCalib.mcSimulation()

print(qgCalib.debugLog())
caModel = qgCalib.calibratedModel()

print('Sigma:')
print(np.array(caModel.sigma()))
print('Slope:')
print(np.array(caModel.slope()))
print('Eta:')
print(np.array(caModel.eta()))

expiries = [ '5y', '10y', '15y', '20y' ]
swpterms = [ '2y', '5y', '10y']
Smiles(atmVTS,caModel,simul,expiries,swpterms,hYts,hYts)

times = np.array(caModel.times())
sigma = np.array(caModel.sigma())
plt.figure()
for k in range(sigma.shape[0]):
    plt.step(times,sigma[k]*1e4, label='k='+str(k))
plt.xlabel('time t')
plt.ylabel('sigma(t) (bp)')
plt.legend()
plt.show()