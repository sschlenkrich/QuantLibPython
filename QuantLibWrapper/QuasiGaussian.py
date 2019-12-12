#!/usr/bin/python

import pandas
import numpy as np
import QuantLib as ql
import matplotlib.pyplot as plt

from Helpers import BachelierImpliedVol


def ModelSmile(model, expiryStr, swaptermStr, projYtsH, discYtsH):
    today = ql.Settings.getEvaluationDate(ql.Settings.instance())
    index = ql.EuriborSwapIsdaFixA( ql.Period(swaptermStr), projYtsH, discYtsH )
    cal = index.fixingCalendar()
    date = cal.advance(today,ql.Period(expiryStr))
    time = ql.Actual365Fixed().yearFraction(today,date)
    S0 = index.fixing(date)
    cf = ql.SwapCashFlows(index.underlyingSwap(date),discYtsH)
    saModel = ql.QGAverageSwaprateModel( ql.QGSwaprateModel(model,
              cf.floatTimes(),cf.floatWeights(),cf.fixedTimes(),cf.annuityWeights(),
              np.linspace(0.0,ql.Actual365Fixed().yearFraction(today,date),int(time/0.25)+1).tolist(),False))
    relStrikes = [ -0.02, -0.01, 0.00, 0.01, 0.02 ]
    vols = []
    for strike in relStrikes:
        try:
            cop = 1 if strike>0 else -1
            price = saModel.vanillaOption(S0+strike,cop)
            vol = BachelierImpliedVol(price,S0+strike,S0,time,cop)*1.0e+4
        except:
            vol = 0.0
        vols.append(vol)
    return relStrikes, vols

def McSimSmile(mcsim, expiryStr, swaptermStr, projYtsH, discYtsH):
    today = ql.Settings.getEvaluationDate(ql.Settings.instance())
    index = ql.EuriborSwapIsdaFixA( ql.Period(swaptermStr), projYtsH, discYtsH )
    cal = index.fixingCalendar()
    date = cal.advance(today,ql.Period(expiryStr))
    time = ql.Actual365Fixed().yearFraction(today,date)
    S0 = index.fixing(date)
    cf = ql.SwapCashFlows(index.underlyingSwap(date),discYtsH)
    annuity = 0.0
    for k in range(len(cf.fixedTimes())):
        annuity += cf.annuityWeights()[k]*discYtsH.discount(cf.floatTimes()[k])
    relStrikes = [ -0.02, -0.01, 0.00, 0.01, 0.02 ]
    vols = []
    for strike in relStrikes:
        try:
            cop = 1 if strike>0 else -1
            swaption = ql.RealMCSwaption(time,cf.floatTimes(),cf.floatWeights(),cf.fixedTimes(),cf.annuityWeights(),S0+strike,cop)
            price = ql.RealMCPayoffPricer_NPV([swaption],mcsim) / annuity
            vol = BachelierImpliedVol(price,S0+strike,S0,time,cop)*1.0e+4
        except:
            vol = 0.0
        vols.append(vol)
    return relStrikes, vols

def MarketSmile(volTS, expiryStr, swaptermStr, projYtsH, discYtsH):
    today = ql.Settings.getEvaluationDate(ql.Settings.instance())
    index = ql.EuriborSwapIsdaFixA( ql.Period(swaptermStr), projYtsH, discYtsH )
    cal = index.fixingCalendar()
    date = cal.advance(today,ql.Period(expiryStr))
    S0 = index.fixing(date)
    term = ql.Period(swaptermStr)
    relStrikes = np.linspace(-0.03, 0.03, 61).tolist()
    vols = [ volTS.volatility(date,term,S0+strike,True)*1e+4
                 for strike in relStrikes ]
    return relStrikes, vols


def Smiles(volTS, model, mcsim, expiries, swapterms, projYtsH, discYtsH):
    fig = plt.figure(figsize=(len(swapterms)*4,len(expiries)*2))
    k = 0
    for expStr in expiries:
        for termStr in swapterms:
            k = k + 1
            ax = fig.add_subplot(len(expiries),len(swapterms),k)
            if volTS:
                s1, v1 = MarketSmile(volTS,expStr,termStr,projYtsH,discYtsH)
                ax.plot(s1,v1,'b-')
            if model:
                s2, v2 = ModelSmile(model,expStr,termStr,projYtsH,discYtsH)
                ax.plot(s2,v2,'b*')
            if mcsim:
                s3, v3 = McSimSmile(mcsim,expStr,termStr,projYtsH,discYtsH)
                ax.plot(s3,v3,'r*')
            ax.set_ylim(0,100)    
    plt.show()


def LVSmiles(volTS, model, expiries, swapTermStr, projYtsH, discYtsH,times):
    today = ql.Settings.getEvaluationDate(ql.Settings.instance())
    index = ql.EuriborSwapIsdaFixA( ql.Period(swapTermStr), projYtsH, discYtsH )
    exerDates = [ index.fixingCalendar().advance(today,ql.Period(e)) for e in expiries ]
    exerTimes = [ ql.Actual365Fixed().yearFraction(today,e) for e in exerDates ]
    stdDevStr = [ -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0 ]
    tbl = np.array(model.calibrationTest(exerDates,stdDevStr))
    fig = plt.figure(figsize=(6,len(expiries)*2))
    k = 0
    for expStr in expiries:
        ax = fig.add_subplot(len(expiries),1,k+1)
        if volTS:
            s1, v1 = MarketSmile(volTS,expStr,swapTermStr,projYtsH,discYtsH)
            ax.plot(s1,v1,'b-')
        if model:
            S0 = tbl[k*len(stdDevStr)+3][6]
            s2 = np.array([ tbl[i][ 6] for i in range(k*len(stdDevStr),(k+1)*len(stdDevStr)) ]) - S0
            v2 = np.array([ tbl[i][10] for i in range(k*len(stdDevStr),(k+1)*len(stdDevStr)) ])*1.0e4
            ax.plot(s2,v2,'b*')
            v3 = np.array([ tbl[i][11] for i in range(k*len(stdDevStr),(k+1)*len(stdDevStr)) ])*1.0e4
            ax.plot(s2,v3,'r*')
            v4 = np.array([ tbl[i][12] for i in range(k*len(stdDevStr),(k+1)*len(stdDevStr)) ])*1.0e4
            ax.plot(s2,v4,'g*')
            if times:
                idx = (np.abs(np.asarray(times) - exerTimes[k])).argmin()
                s5 = np.linspace(-0.03,0.03,61)
                v5 = []
                for s in s5:
                    try:
                        v = model.sigmaS(int(idx),S0+s)
                    except:
                        v = 0.0
                    v5 = v5 + [v]
                v5 = np.array(v5)*1.0e4
                ax.plot(s5,v5,'r-')
        k = k + 1
    plt.show()

