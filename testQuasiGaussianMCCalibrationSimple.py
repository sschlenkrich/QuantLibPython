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

swapTerms = [ ql.Period(str(p)+'y') for p in [2, 5, 10, 20] ]
expiTerms = [ ql.Period(str(p)+'y') for p in [1, 2, 3, 5, 7, 10, 15] ]
valsMatrx = ql.Matrix(len(expiTerms),len(swapTerms))
for i in range(len(expiTerms)):
    for j in range(len(swapTerms)):
        valsMatrx[i][j] = 0.005
atmVTS = ql.SwaptionVolatilityMatrix(ql.TARGET(),ql.Following,expiTerms,swapTerms,valsMatrx,ql.Actual365Fixed(),True,ql.Normal)

indices = [ ql.EuriborSwapIsdaFixA(ql.Period( '2y'),hYts,hYts),
            ql.EuriborSwapIsdaFixA(ql.Period('10y'),hYts,hYts) ]

endCrit = ql.EndCriteria(100,10,1.0e-4,1.0e-4,1.0e-4)  # for optimizer

sigmaMax     = 0.01  # maximum sigma parameter in optimisation
slopeMax     = 0.3   # maximum slope parameter in optimisation
curveMax     = 0.5   # maximum curve parameter in optimisation
sigmaWeight  = 1.0   # put emphasis on ATM fit
slopeWeight  = 0.0   # put emphasis on skew fit
curveWeight  = 0.0   # put emphasis on smile fit
penaltySigma = 0.1   # force similar sigma parameters per factor
penaltySlope = 0.01  # force similar slope parameters per factor
penaltyCurve = 0.01  # force similar curve parameters per factor
monteCarloStepSize = 0.5   # time stepping step size in years
monteCarloPaths    = 1000  # MC paths used in simulation

qgCalib = ql.QGMonteCarloCalibrator(qgModel,ql.SwaptionVolatilityStructureHandle(atmVTS),indices,
              monteCarloStepSize,monteCarloPaths,sigmaMax,slopeMax,curveMax,
              sigmaWeight,slopeWeight,curveWeight,penaltySigma,penaltySlope,penaltyCurve,
              endCrit)
simul   = qgCalib.mcSimulation()
caModel = qgCalib.calibratedModel()
print(qgCalib.debugLog())

print('Sigma:')
print(np.array(caModel.sigma()))
print('Slope:')
print(np.array(caModel.slope()))
print('Curve:')
print(np.array(caModel.curve()))

expiries = [ '5y', '10y', '15y', '20y' ]
swpterms = [ '2y', '5y', '10y']
Smiles(atmVTS,caModel,simul,expiries,swpterms,hYts,hYts)

