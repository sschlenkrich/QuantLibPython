#!/usr/bin/python

import QuantLib as ql

import matplotlib.pyplot as plt
import pandas

class YieldCurve:

    # Python constructor
    def __init__(self, terms, rates):
        today = ql.Settings.getEvaluationDate(ql.Settings.instance())
        self.terms = terms
        self.dates = [ ql.WeekendsOnly().advance(today,ql.Period(term),ql.ModifiedFollowing) for term in terms ]
        self.rates = rates
        # use rates as backward flat interpolated continuous compounded forward rates
        self.yts = ql.ForwardCurve(self.dates,self.rates,ql.Actual365Fixed(),ql.NullCalendar())

    # zero coupon bond
    def discount(self,dateOrTime):
        return self.yts.discount(dateOrTime,True)

    def forwardRate(self,time):
        return self.yts.forwardRate(time,time,ql.Continuous,ql.Annual,True).rate()
  
    # plot zero rates and forward rate
    def plot(self,stepsize=0.1):
        times = [ k*stepsize for k in range(int(round(30.0/stepsize,0))+1) ]
        continuousForwd = [ self.yts.forwardRate(time,time,ql.Continuous,ql.Annual,True).rate() for time in times ]
        continuousZeros = [ self.yts.zeroRate(time,ql.Continuous,ql.Annual,True).rate() for time in times ]
        annualZeros     = [ self.yts.zeroRate(time,ql.Compounded,ql.Annual,True).rate() for time in times ]
        # print(times, continuousForwd, continuousZeros, annualZeros)
        plt.plot(times,continuousForwd, label='Cont. forward rate')
        plt.plot(times,continuousZeros, label='Cont. zero rate')
        plt.plot(times,annualZeros,     label='Annually comp. zero rate')
        plt.legend()
        plt.xlabel('Maturity')
        plt.ylabel('Interest rate')
        plt.show()

    # return a table with curve data
    def table(self):
        table = pandas.DataFrame( [ self.terms, self.dates, self.rates ] ).T
        table.columns = [ 'Terms', 'Dates', 'Rates' ]
        return table

    def referenceDate(self):
        return self.referenceDate()

# end of YieldCurve

