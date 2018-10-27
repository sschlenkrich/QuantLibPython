import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__),'QuantLibWrapper'))

import pandas

import QuantLib as ql

from QuantLibWrapper import YieldCurve, Swap, Swaption

# market data 

terms = [    '1y',    '2y',    '3y',    '4y',    '5y',    '6y',    '7y',    '8y',    '9y',   '10y',   '12y',   '15y',   '20y',   '25y',   '30y' ] 
rates = [ 2.70e-2, 2.75e-2, 2.80e-2, 3.00e-2, 3.36e-2, 3.68e-2, 3.97e-2, 4.24e-2, 4.50e-2, 4.75e-2, 4.75e-2, 4.70e-2, 4.50e-2, 4.30e-2, 4.30e-2 ] 
rates2 = [ r+0.005 for r in rates ]

discCurve = YieldCurve(terms,rates)
projCurve = YieldCurve(terms,rates2)

normalVol = 0.01

# underlying swap

startDate = ql.Date(30, 10, 2028)
endDate = ql.Date(30, 10, 2038)
swap = Swap(startDate,endDate,0.052513,discCurve,projCurve)

# swaption

expiryDate = ql.TARGET().advance(startDate,ql.Period('-2d'),ql.Preceding)
swaption = Swaption(swap,expiryDate,normalVol)

print('Swap NPV: %11.2f' % (swap.npv()))
print('FairRate: %11.6f' % (swap.fairRate()))
print('Swpt NPV: %11.2f' % (swaption.npv()))
print('Annuity:  %11.2f' % (swaption.annuity()))

details = swaption.bondOptionDetails()
print('ExpiryTime: ' + str(details['expiryTime']))
print('Strike:     ' + str(details['strike']))
print('CallOrPut:  ' + str(details['callOrPut']))

table = pandas.DataFrame( [ details['payTimes'], details['cashFlows'],
    [ discCurve.discount(T) for T in details['payTimes'] ] ]).T
table.columns = [ 'payTimes', 'cashFlows', 'DF' ]
print(table)

