import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__),'QuantLibWrapper'))

import numpy as np
import pandas

import QuantLib as ql

from QuantLibWrapper import SwaptionVolatility, BachelierImpliedVol, \
     ModelSmile, McSimSmile, MarketSmile, Smiles

import matplotlib.pyplot as plt

today = ql.Settings.getEvaluationDate(ql.Settings.instance())

hYts = ql.YieldTermStructureHandle(
           ql.FlatForward(ql.Settings.getEvaluationDate(ql.Settings.instance()),
                          0.03,ql.Actual365Fixed()))

d     = 2

times = [   2.0,   5.0,  10.0,  20.0    ]
sigma = [ [ 0.01,  0.01,  0.01,  0.01 ],
          [ 0.01,  0.01,  0.01,  0.01 ] ]
slope = [ [ 0.10,  0.10,  0.10,  0.10 ],
          [ 0.10,  0.10,  0.10,  0.10 ] ]
curve = [ [ 0.00,  0.00,  0.00,  0.00 ],
          [ 0.00,  0.00,  0.00,  0.00 ] ]
eta   = [   0.20,  0.20,  0.20,  0.20   ]


d     = 3

times = [  10.0   ]
sigma = [ [ 0.005 ],
          [ 0.005 ],
          [ 0.005 ] ]
slope = [ [ 0.00  ],
          [ 0.00  ], 
          [ 0.00  ] ]
curve = [ [ 0.00  ],
          [ 0.00  ], 
          [ 0.00  ] ]
eta   = [   0.00   ]

d = 5

times = [   5.0,   10.0,   15.0,   20.0,   ]
sigma = [ [ 0.005,  0.005,  0.005,  0.005, ],
          [ 0.005,  0.005,  0.005,  0.005, ],
          [ 0.005,  0.005,  0.005,  0.005, ],
          [ 0.005,  0.005,  0.005,  0.005, ],
          [ 0.005,  0.005,  0.005,  0.005, ] ]
slope = [ [ 0.01,   0.01,   0.01,   0.01,  ],
          [ 0.01,   0.01,   0.01,   0.01,  ], 
          [ 0.01,   0.01,   0.01,   0.01,  ], 
          [ 0.01,   0.01,   0.01,   0.01,  ], 
          [ 0.01,   0.01,   0.01,   0.01,  ] ]
curve = [ [ 0.00,   0.00,   0.00,   0.00,  ],
          [ 0.00,   0.00,   0.00,   0.00,  ], 
          [ 0.00,   0.00,   0.00,   0.00,  ], 
          [ 0.00,   0.00,   0.00,   0.00,  ], 
          [ 0.00,   0.00,   0.00,   0.00,  ] ]
eta   = [   0.00,   0.00,   0.00,   0.00,   ]

delta = [   0.5,  5.0, 10.0, 15.0, 20.0  ]
chi   = [   0.01, 0.04, 0.07, 0.10, 0.13 ]
Gamma = [ [ 1.00,  0.50, 0.25, 0.00,-0.25 ],
          [ 0.50,  1.00, 0.50, 0.25, 0.00 ],
          [ 0.25,  0.50, 1.00, 0.50, 0.25 ],
          [ 0.00,  0.25, 0.50, 1.00, 0.50 ],
          [-0.25,  0.00, 0.25, 0.50, 1.00 ] ]
theta = 0.1

qgModel = ql.QuasiGaussianModel(hYts,d,times,sigma,slope,curve,eta,delta,chi,Gamma,theta)

sw = SwaptionVolatility('swaptionATMVols.csv',hYts,hYts)

index = ql.EuriborSwapIsdaFixA( ql.Period('1y'),hYts,hYts)
indices = [ index.clone(ql.Period('2y')), index.clone(ql.Period('10y')) ]
#indices = [ index.clone(ql.Period('10y')) ]

endCrit = ql.EndCriteria(100,10,1.0e-4,1.0e-4,1.0e-4)

input("Press Enter to continue...")

qgCalib = ql.QGCalibrator(qgModel,ql.SwaptionVolatilityStructureHandle(sw.volTS),indices,
              0.25,False,0.010,0.12,0.5,1.0,0.0,0.0,1.0,0.01,endCrit)
print(qgCalib.debugLog())
caModel = qgCalib.calibratedModel()
#caModel = qgModel

print('Sigma:')
print(np.array(caModel.sigma()))
print('Slope:')
print(np.array(caModel.slope()))
print('Eta:')
print(np.array(caModel.eta()))

simTimes = np.linspace(0.0,20.0,241).tolist()
obsTimes = simTimes

adjTimes = np.linspace(1.0,20.0,20).tolist()

input("Press Enter to continue...")

mcsim = ql.RealMCSimulation(caModel,simTimes,obsTimes,pow(2,15),1234,False,True,True)

print('Start simulation... ',end='',flush=True)
mcsim.simulate()
print('Done. ',end='\n',flush=True)
print('Calculate numeraire adjuster... ',end='',flush=True)
mcsim.calculateNumeraireAdjuster(adjTimes)
print('Done. ',end='\n',flush=True)
print('Calculate zcb adjuster... ',end='',flush=True)
mcsim.calculateZCBAdjuster(adjTimes,np.linspace(1.0,10.0,10).tolist())
print('Done. ',end='\n',flush=True)

with np.printoptions(precision=1, suppress=True):
    print(np.array(mcsim.numeraireAdjuster())*1e+4)
    print(np.array(mcsim.zcbAdjuster())*1e+4)

expiries = [ '5y', '10y', '15y', '20y' ]
swpterms = [ '2y', '5y', '10y']
Smiles(sw.volTS,caModel,mcsim,expiries,swpterms,hYts,hYts)


print('Done.')