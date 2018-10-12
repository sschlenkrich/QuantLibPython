import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__),'QuantLibWrapper'))

import QuantLib as ql
import QuantLibWrapper.YieldCurve as yc
import QuantLibWrapper.Swap as sw

terms = [     \
        '1y', \
        '2y', \
        '3y', \
        '4y', \
        '5y', \
        '6y', \
        '7y', \
        '8y', \
        '9y', \
        '10y',\
        '12y',\
        '15y',\
        '20y',\
        '25y',\
        '30y' ]

rates = [       \
        2.70e-2,\
        2.75e-2,\
        2.80e-2,\
        3.00e-2,\
        3.36e-2,\
        3.68e-2,\
        3.97e-2,\
        4.24e-2,\
        4.50e-2,\
        4.75e-2,\
        4.75e-2,\
        4.70e-2,\
        4.50e-2,\
        4.30e-2,\
        4.30e-2 ]

rates2 = [ r+0.005 for r in rates ]

discCurve = yc.YieldCurve(terms,rates)
projCurve = yc.YieldCurve(terms,rates)

startDate = ql.Date(30, 10, 2018)
endDate = ql.Date(30, 10, 2038)

swap = sw.Swap(startDate,endDate,0.03,discCurve,projCurve)

print('NPV:      '+str(swap.npv()))
print('FairRate: '+str(swap.fairRate()))


