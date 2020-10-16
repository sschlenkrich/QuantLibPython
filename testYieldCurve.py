
import QuantLib as ql
import QuantLibWrapper.YieldCurve as yc

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

fwdRateYC = yc.YieldCurve(terms,rates)

print(fwdRateYC.table())

fwdRateYC.plot(1.0/365)

today = ql.Settings.getEvaluationDate(ql.Settings.instance())
print('Today: '+str(today))
calendar = ql.TARGET()
period   = ql.Period('500d')
bdc      = ql.Following
print(str(calendar)+', '+str(period)+', '+str(bdc))
maturity = calendar.advance(today,period,bdc)
print('Maturity: '+str(maturity))
discountFactor = fwdRateYC.discount(maturity)
print('Discount: '+str(discountFactor))
