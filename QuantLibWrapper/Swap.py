#!/usr/bin/python

import pandas
import QuantLib as ql

class Swap:

    # Python constructor
    def __init__(self, startDate, endDate, fixedRate, discYieldCurve, projYieldCurve):
        # we need handles of the yield curves...
        discHandle = ql.RelinkableYieldTermStructureHandle()
        projHandle = ql.RelinkableYieldTermStructureHandle()
        discHandle.linkTo(projYieldCurve.yts)
        projHandle.linkTo(discYieldCurve.yts)
        # schedule generation details
        fixedLegTenor = ql.Period('1y')
        floatLegTenor = ql.Period('6m')
        calendar = ql.TARGET()
        fixedLegAdjustment = ql.ModifiedFollowing
        floatLegAdjustment = ql.ModifiedFollowing
        endOfMonthFlag = False
        # schedule creation
        fixedSchedule = ql.Schedule(startDate, endDate,
                          fixedLegTenor, calendar,
                          fixedLegAdjustment, fixedLegAdjustment,
                          ql.DateGeneration.Backward, endOfMonthFlag)
        floatSchedule = ql.Schedule(startDate, endDate,
                          floatLegTenor, calendar,
                          floatLegAdjustment, floatLegAdjustment,
                          ql.DateGeneration.Backward, endOfMonthFlag)
        # interest rate details
        index = ql.Euribor(floatLegTenor,projHandle)
        spread = 0.0   # no floating rate spread applied
        fixedLegDayCounter = ql.Thirty360()
        floatLegDayCounter = index.dayCounter()
        # paymentAdjustment  = ql.Following ... not exposed to user via Python
        # notional and payer/receiver
        notional = 1.0e+8
        payerOrReceiver = ql.VanillaSwap.Payer
        # swap creation
        self.swap = ql.VanillaSwap(payerOrReceiver, notional,
                   fixedSchedule, fixedRate, fixedLegDayCounter,
                   floatSchedule, index, spread,
                   floatLegDayCounter)
        # pricing engine to allow discounting etc.
        swapEngine = ql.DiscountingSwapEngine(discHandle)
        self.swap.setPricingEngine(swapEngine)

    def npv(self):
        return self.swap.NPV()

    def fairRate(self):
        return self.swap.fairRate()
    
    def fixedCashFlows(self):
        table = pandas.DataFrame( [
            [cf.date()   for cf in self.swap.fixedLeg()],
            [cf.amount() for cf in self.swap.fixedLeg()]
            ] ).T
        table.columns = [ 'Date', 'Amount' ]  
        return table

    def floatCashFlows(self):
        table = pandas.DataFrame( [
            [cf.date()   for cf in self.swap.floatingLeg()],
            [cf.amount() for cf in self.swap.floatingLeg()]
            ] ).T
        table.columns = [ 'Date', 'Amount' ]  
        return table
