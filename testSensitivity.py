import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__),'QuantLibWrapper'))

import matplotlib.pyplot as plt

import numpy as np
import pandas

import QuantLib as ql

from QuantLibWrapper import YieldCurve, Swap, Swaption, createSwaption, \
    HullWhiteModel, HullWhiteModelFromSwaption, BermudanSwaption, PDESolver, \
    DensityIntegrationWithBreakEven, CubicSplineExactIntegration

today = ql.Date(3,9,2018)
ql.Settings.setEvaluationDate(ql.Settings.instance(),today)

# market data 

terms = [    '1y',    '2y',    '3y',    '4y',    '5y',    '6y',    '7y',    '8y',    '9y',   '10y',   '12y',   '15y',   '20y',   '25y',   '30y' ] 
rates = [ 2.70e-2, 2.75e-2, 2.80e-2, 3.00e-2, 3.36e-2, 3.68e-2, 3.97e-2, 4.24e-2, 4.50e-2, 4.75e-2, 4.75e-2, 4.70e-2, 4.50e-2, 4.30e-2, 4.30e-2 ] 

terms = [ '30y'   ]  # flat curve
rates = [ 5.00e-2 ] 

rates2 = [ r+0.005 for r in rates ]

discCurve = YieldCurve(terms,rates)
projCurve = YieldCurve(terms,rates)

a = 0.03 # mean reversion
vol = 0.01
h = 1.0e-12

swaptionStrike = createSwaption('10y','10y',discCurve,projCurve).fairRate() \
                 + vol * np.sqrt(10.0) * np.sqrt(3.0)
swaptionStrike = 0.1048
print('swaptionStrike = ' + str(swaptionStrike))


res = []
for h in [ 10.0**(-k) for k in range(3, 18) ]:
    print('h = ' + str(h))
    # swaptions
    s0 = createSwaption('10y','10y',discCurve,projCurve,strike=swaptionStrike,normalVolatility=vol)
    sp = createSwaption('10y','10y',discCurve,projCurve,strike=swaptionStrike,normalVolatility=vol+h)
    sm = createSwaption('10y','10y',discCurve,projCurve,strike=swaptionStrike,normalVolatility=vol-h)
    # models
    h0 = HullWhiteModelFromSwaption(s0,a)
    hp = HullWhiteModelFromSwaption(sp,a)
    hm = HullWhiteModelFromSwaption(sm,a)
    # Berms via PDE
    p0 = BermudanSwaption([s0],a,model=h0,method=PDESolver(h0,101,3.0,0.5,1.0/12.0))
    pp = BermudanSwaption([sp],a,model=hp,method=PDESolver(hp,101,3.0,0.5,1.0/12.0))
    pm = BermudanSwaption([sm],a,model=hm,method=PDESolver(hm,101,3.0,0.5,1.0/12.0))
    # Berms via density integration
    i0 = BermudanSwaption([s0],a,model=h0,method=DensityIntegrationWithBreakEven(CubicSplineExactIntegration(h0,101,5)))
    ip = BermudanSwaption([sp],a,model=hp,method=DensityIntegrationWithBreakEven(CubicSplineExactIntegration(hp,101,5)))
    im = BermudanSwaption([sm],a,model=hm,method=DensityIntegrationWithBreakEven(CubicSplineExactIntegration(hm,101,5)))
    res.append([ h, s0.vega(),
        s0.npv(),            sp.npv(),            sm.npv(),
        s0.npvHullWhite(h0), sp.npvHullWhite(hp), sm.npvHullWhite(hm),
        p0.npv(),            pp.npv(),            pm.npv(),
        i0.npv(),            ip.npv(),            im.npv()
    ])

table = pandas.DataFrame(res)
table.columns = [ 'h', 'Vega', 'B0', 'Bp', 'Bm', 'H0', 'Hp', 'Hm', 
                   'P0', 'Pp', 'Pm', 'I0', 'Ip', 'Im' ]

table.to_excel('Sensitivities.xls')
# table = pandas.read_excel('Sensitivities.xls')

def plotRelativeError(id,title=''):
    fig = plt.figure(figsize=(6, 4))
    # upward derivative
    vega = (table[id+'p']-table[id+'0'])/table['h']*1.0e-4
    relError = abs(vega/table['Vega']-1)
    plt.loglog(table['h'],relError,label='Upward')
    # downward derivative
    vega = (table[id+'0']-table[id+'m'])/table['h']*1.0e-4
    relError = abs(vega/table['Vega']-1)
    plt.loglog(table['h'],relError,label='Downward')
    # two-sided derivative
    vega = (table[id+'p']-table[id+'m'])/table['h']/2.0*1.0e-4
    relError = abs(vega/table['Vega']-1)
    plt.loglog(table['h'],relError,label='Two-sided')
    # plot properties
    plt.xlabel('shift size h')
    plt.ylabel('|RelError|')
    plt.title(title)
    plt.ylim(1.0e-11,10.0)
    plt.legend()
    

# plt.ylim(0,600)
# plt.xlim(-0.06,0.12)

plotRelativeError('B','Bachelier Formula')
plotRelativeError('H','Hull-White Analytical')
plotRelativeError('P','Hull-White PDE Solver')
plotRelativeError('I','Hull-White Density Integration')

plt.show()
