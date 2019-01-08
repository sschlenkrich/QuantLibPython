import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__),'QuantLibWrapper'))

import numpy as np
import pandas

import QuantLib as ql

from QuantLibWrapper import SwaptionVolatility, BachelierImpliedVol, \
     ModelSmile, McSimSmile, MarketSmile, Smiles, LVSmiles

import matplotlib.pyplot as plt

today = ql.Settings.getEvaluationDate(ql.Settings.instance())

hYts = ql.YieldTermStructureHandle(
           ql.FlatForward(ql.Settings.getEvaluationDate(ql.Settings.instance()),
                          0.03,ql.Actual365Fixed()))

sw = SwaptionVolatility('swaptionATMVols2.csv',hYts,hYts)

index = ql.EuriborSwapIsdaFixA( ql.Period('10y'),hYts,hYts)

times = np.linspace(1.0,10.25,41).tolist()

model = ql.QGLocalvolModel('SLV',hYts,ql.SwaptionVolatilityStructureHandle(sw.volTS),
            0.03,0.1,0.0,index,times,[],201,False,0.15,3.0,pow(2,13),1234,1)

model.simulateAndCalibrate()

for s in model.debugLog():
    print(s)

exerStr = [ '2y', '5y', '7y', '10y' ]
LVSmiles(sw.volTS,model,exerStr,'10y',hYts,hYts,times)

print('Done.')

