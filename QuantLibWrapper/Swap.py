#!/usr/bin/python

import pandas
import QuantLib as ql

class Swap:

    # Python constructor
    def __init__(self, startDate, endDate, fixedRate, discYieldCurve, projYieldCurve, payerOrReceiver=ql.VanillaSwap.Payer, notional=1.0e+4):
        # we need some swap details for swaption pricing
        self.payerOrReceiver = payerOrReceiver
        self.notional        = notional
        self.fixedRate       = fixedRate        
        # for Hull-White pricing we need to reference the yield curves
        self.discYieldCurve = discYieldCurve
        self.projYieldCurve = projYieldCurve
        # we need handles of the yield curves...        
        self.discHandle = ql.RelinkableYieldTermStructureHandle()
        self.projHandle = ql.RelinkableYieldTermStructureHandle()
        self.discHandle.linkTo(discYieldCurve.yts)
        self.projHandle.linkTo(projYieldCurve.yts)  
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
        index = ql.Euribor(floatLegTenor,self.projHandle)
        spread = 0.0   # no floating rate spread applied
        fixedLegDayCounter = ql.Thirty360()
        floatLegDayCounter = index.dayCounter()
        # paymentAdjustment  = ql.Following ... not exposed to user via Python
        # swap creation
        self.swap = ql.VanillaSwap(self.payerOrReceiver, self.notional,
                   fixedSchedule, fixedRate, fixedLegDayCounter,
                   floatSchedule, index, spread,
                   floatLegDayCounter)
        # pricing engine to allow discounting etc.
        swapEngine = ql.DiscountingSwapEngine(self.discHandle)
        self.swap.setPricingEngine(swapEngine)

    def npv(self):
        return self.swap.NPV()

    def fairRate(self):
        return self.swap.fairRate()

    def annuity(self):
        return abs(self.swap.fixedLegBPS())/1.0e-4
    
    def fixedCashFlows(self):
        table = pandas.DataFrame( [
            [ql.as_fixed_rate_coupon(cf).accrualStartDate() for cf in self.swap.fixedLeg()],
            [ql.as_fixed_rate_coupon(cf).accrualEndDate()   for cf in self.swap.fixedLeg()],
            [ql.as_fixed_rate_coupon(cf).rate()             for cf in self.swap.fixedLeg()],
            [cf.date()   for cf in self.swap.fixedLeg()],
            [cf.amount() for cf in self.swap.fixedLeg()]
            ] ).T
        table.columns = [ 'AccrualStartDate', 'AccrualEndDate', 'Rate', 'PayDate', 'Amount' ]  
        return table

    def floatCashFlows(self):
        table = pandas.DataFrame( [
            [ql.as_floating_rate_coupon(cf).accrualStartDate() for cf in self.swap.floatingLeg()],
            [ql.as_floating_rate_coupon(cf).accrualEndDate()   for cf in self.swap.floatingLeg()],
            [ql.as_floating_rate_coupon(cf).rate()             for cf in self.swap.floatingLeg()],
            [cf.date()   for cf in self.swap.floatingLeg()],
            [cf.amount() for cf in self.swap.floatingLeg()],
            [ql.as_floating_rate_coupon(cf).fixingDate() for cf in self.swap.floatingLeg()]
            ] ).T
        table.columns = [ 'AccrualStartDate', 'AccrualEndDate', 'Rate', 'PayDate', 'Amount', 'FixingDate' ]  
        return table
