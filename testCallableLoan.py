# CallableLoan.py
#
# a module for pricing and cash flow representation of callable loans
#
# (C) Sebastian Schlenkrich, December 2019
#

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__),'QuantLibWrapper'))


import QuantLib as ql
import numpy as np
import pandas as pd

from QuantLibWrapper.QuantLibMonteCarlo import *

class CallableLoan:

    # python constructor
    def __init__(self, startDate, payDates, principalPayments, couponRatePayoff):
        # check dimensions
        assert len(payDates)>1, 'Error! len(payDates)>1 required'
        assert len(payDates)==len(principalPayments), 'Error! len(payDates)==len(principalPayments) required'
        #
        self.startDate         = startDate          # start date of first coupon; other coupons are assumed start/fixing at previous pay date
        self.payDates          = payDates           # expiry and cash flow dates T_1,...,T_N
        self.principalPayments = np.array(principalPayments)  # principal payments [P_1,...,P_N]
        self.couponRatePayoff  = couponRatePayoff   # payoff for R_1; template payoff for R_i
        # some pre-determined quantities
        self.K = np.array([ sum(principalPayments[k:]) for k in range(len(principalPayments)) ] + [0.0] )  # strikes K_0,...,K_N-1, K_N
        #self.tau = np.array( [ ql.Actual360().yearFraction(startDate,payDates[0]) ] + \
        #                     [ ql.Actual360().yearFraction(payDates[i],payDates[i+1]) for i in range(len(payDates)-1) ] )
        self.tau = np.array( [ ql.Actual365Fixed().yearFraction(startDate,payDates[0]) ] + \
                             [ ql.Actual365Fixed().yearFraction(payDates[i],payDates[i+1]) for i in range(len(payDates)-1) ] )
        today = ql.Settings.getEvaluationDate(ql.Settings.instance())
        self.T = np.array([ ql.Actual365Fixed().yearFraction(today,p) for p in payDates ])  # times T_1,...,T_N
        self.fixingTimes = np.array( [ ql.Actual365Fixed().yearFraction(today,startDate) ] + \
                                     [ ql.Actual365Fixed().yearFraction(today,p) for p in payDates[:-1] ] )

    def payoffScript(self, reg=[], sim=None):
        # interest cash flows I_1,...,I_N
        I = [ Cache(Pay((K*tau)*self.couponRatePayoff(Tfix),Tpay)) \
              for K, tau, Tfix, Tpay in zip(self.K[:-1],self.tau,self.fixingTimes,self.T) ]
        # contractual principal cash flows P_1,...,P_N
        P = [ Fixed(p,Tpay) for p, Tpay in zip(self.principalPayments,self.T) ]
        # underlying cash flows X_i = P_i + I_i
        X = [ Cache(Pay(i+p,Tpay)) for i, p, Tpay in zip(I,P,self.T) ]
        # hold values H_N-1,...,H_0
        # obsTimes = [ self.T[0] ] + self.T.tolist()[:-2]  # T_1, T_1,...,T_N-2
        obsTimes = self.T.tolist()[:-1]  # T_1, T_2,...,T_N-1
        Hi = Fixed(0.0,self.T[-2]) # hold value terminal condition, H_N-1 = 0
        H = [ Hi ]
        for i in reversed(range(len(self.T)-1)):  # we start with calculating H_N-2 from H_N-1 and U_N-1
            Kminus = Fixed(-self.K[i+1],self.T[i])
            U = [ Kminus ] + X[i+1:]
            # obsTime = self.T[i-1] if i>0 else self.T[0]  # H0 is observed at T1
            # Hi = Cache( AmcMax(U,Hi,[ r(obsTime) for r in reg ],obsTime,sim,2) )
            Hi = Cache( AmcMax(U,Hi,[ r(obsTimes[i]) for r in reg ],obsTimes[i],sim,2) )
            H.append( Hi )
        # we re-order from H_0,...,H_N-1
        H = list(reversed(H))
        # now we can calculate exercise flags and survivals
        #
        # Delta calculation is the tricky piece here. We cannot directly look into the AmcMax function
        # and observe the trigger. Therefore, we compare the input and result of the AmcMax function.
        # However, H0 and H1 have different observation times. We use AmcOne to discount to the earlier
        # observation time (by looking into the future).
        # Also, the regression in AmcMax might exercise into sub-optimal states, i.e. if U<H. Thus we
        # include this case into exercise determination to ansure consistency between cash flows and
        # Bermudan option. [ToDo] Maybe it is better to exclude this case to improve the optimal
        # exercise strategy. This requires a bit more research.
        delta = [ Cache( 0.0 +                                   # make sure obsTime=0 for debugging
                  AmcOne( H1, H0, [], obsTime, 1.0, None, 0) +   # the AMC trigger may exercise into non-optimal states
                  AmcOne( H0, H1, [], obsTime, 1.0, None, 0) )   # this checks the actual max-condition
                  for H0, H1, obsTime in zip(H[:-1],H[1:],obsTimes ) ]  # delta_1,...,delta_N-1; we look into the future here
        delta = delta + [ Fixed(0.0) ]  # delta_1,...,delta_N-1, delta_N
        Qi = Fixed(1.0)                  
        Q = [ Qi ]  # Q_0
        for d in delta[:-1]:
              Qi = Cache(Qi * (1.0 - d))
              Q.append( Qi )  # Q_1,...,Q_N-1
        # finally we can calculate option cash flows
        IOpt = [ Pay((1-q)*i,T) for q,i,T in zip(Q,I,self.T) ]   # I_1, I_2,...,I_N
        POpt = [ Pay((1-q)*p - q*d*k, T) for q, p, d, k, T in zip(Q, P, delta, self.K[1:], self.T) ]

        # keys
        IKeys = [ 'I' + str(i) for i in range(len(I)) ]
        PKeys = [ 'P' + str(i) for i in range(len(P)) ]
        HKeys = [ 'H' + str(i) for i in range(len(H)) ]
        deltaKeys = [ 'delta' + str(i) for i in range(len(delta)) ]
        QKeys = [ 'Q' + str(i) for i in range(len(Q)) ]
        IOptKeys = [ 'IOpt' + str(i) for i in range(len(IOpt)) ]
        POptKeys = [ 'POpt' + str(i) for i in range(len(POpt)) ]
        # script
        script = ql.RealMCScript(IKeys+PKeys+HKeys+deltaKeys+QKeys+IOptKeys+POptKeys,
                     Raw(I+P+H+delta+Q+IOpt+POpt),['payoff = H0'])
        # we return the script and relevant keys
        result = {}
        result['Script'] = script
        result['IKeys']  = IKeys
        result['PKeys']  = PKeys
        result['HKeys']  = HKeys
        result['deltaKeys']  = deltaKeys
        result['QKeys']  = QKeys
        result['IOptKeys']  = IOptKeys
        result['POptKeys']  = POptKeys
        return result

    # return a default model
    def model(self,rate=0.01,discSpread=0.01):
        # discount curve
        today = ql.Settings.getEvaluationDate(ql.Settings.instance())
        ratesYtsH = ql.YieldTermStructureHandle(ql.FlatForward(today,rate,ql.Actual365Fixed()))
        # we specify a single factor Gaussian model for rates simulation
        d     = 1
        times = [  20.0      ]
        sigma = [ [ 0.0050 ] ]
        slope = [ [ 0.0    ] ]
        curve = [ [ 0.0    ] ]
        eta   = [   0.0      ]
        delta = [   0.001    ]
        chi   = [   0.01   ]
        Gamma = [ [ 1.00  ]  ]
        theta = 0.1
        ratesModel = ql.QuasiGaussianModel(ratesYtsH,d,times,sigma,slope,curve,eta,delta,chi,Gamma,theta)
        # spread model
        spreadYtsH = ql.YieldTermStructureHandle(ql.FlatForward(today,discSpread,ql.Actual365Fixed()))
        spreadSigma = [ [ 0.0050 ] ]
        spreadChi   = [   0.0001   ]
        corr        = 0.0
        spreadModel = ql.QuasiGaussianModel(spreadYtsH,d,times,spreadSigma,slope,curve,eta,delta,spreadChi,Gamma,theta)
        corrMatrix  = np.identity(4)
        corrMatrix[0,2] = corr
        corrMatrix[2,0] = corr
        # credit hybrid model
        hybridModel = ql.SpreadModel(ratesModel,spreadModel,corrMatrix)
        hybridYtsH = ql.YieldTermStructureHandle(ql.FlatForward(today,rate+discSpread,ql.Actual365Fixed()))
        # return hybridModel, hybridYtsH
        return ratesModel, ratesYtsH

    # return a list of regression variables
    def regressionVariables(self):
        libor = Libor(1.0,'1y','CCY')
        disc  = Libor(1.0,'1y')
        # for hybrid modelling and floaters we want more accurate regression variables
        # return [ libor, lPlus ]
        # return [ self.couponRatePayoff, libor, disc ]
        return [ disc ]

    # return a default Monte-Carlo simulation
    def simulation(self, model=None, nPaths=pow(2,15), seed=4321):
        model = model if model else self.model()
        script = self.payoffScript()['Script']  # only used to get simTimes
        simTimes = script.observationTimes(script.payoffsKeys())
        simTimes = np.linspace(0.0, simTimes[-1],round(2*simTimes[-1])+1)
        mcSim = ql.RealMCSimulation(model,simTimes,simTimes,nPaths,seed)
        return mcSim

    # return present value of loan and option components
    # this method incorporates quite some logic of the valuation methodology
    def npvs(self, mcSim=None):
        model, termStructure = self.model()
        # we need a Monte-Carlo simulation
        mcSim = mcSim if mcSim else self.simulation(model)
        # we also need a separate simulation for AMC regression
        regSim = self.simulation(model,1024,1234)
        # further for regression we need regression variable(s)
        regVars = self.regressionVariables()
        # now we can set up the actual payoff
        payoff = self.payoffScript(regVars,regSim)
        # payoff = self.payoffScript()  # swith off regression
        keys   = payoff['IKeys'] + payoff['PKeys'] + payoff['IOptKeys'] + payoff['POptKeys'] + \
                 payoff['HKeys'] + payoff['deltaKeys'] + payoff['QKeys'] + ['H0']
        script = payoff['Script']
        # we need to simulate the model...
        print('Simulate model...', end='',flush=True)
        mcSim.simulate()
        print('Done',flush=True)
        print('Calculate adjuster...', end='',flush=True)
        mcSim.calculateNumeraireAdjuster(mcSim.simTimes()[1:])
        print('Done',flush=True)
        print('Simulate regression...', end='',flush=True)
        regSim.simulate()
        print('Done',flush=True)
        print('Calculate regression adjuster...', end='',flush=True)
        regSim.calculateNumeraireAdjuster(regSim.simTimes()[1:])
        print('Done',flush=True)
        # now we can do pricing
        print('Simulate payoffs...', end='',flush=True)
        cfs = script.NPV(mcSim,keys)
        print('Done',flush=True)
        result = {}
        N = len(payoff['IKeys'])  #  assume equal length
        result['T']    = self.T
        result['I']    = cfs[0*N:1*N]
        result['P']    = cfs[1*N:2*N]
        result['IOpt'] = cfs[2*N:3*N]
        result['POpt'] = cfs[3*N:4*N]
        result['H']    = cfs[4*N:5*N]
        result['delta'] = cfs[5*N:6*N]
        result['Q']    = cfs[6*N:7*N]
        result['H0']   = cfs[-1]
        # forward cash flows
        df = lambda t : termStructure.discount(t)
        result['I_Fwd'] = [ x/df(t) for x,t in zip(result['I'],self.T) ]
        result['P_Fwd'] = [ x/df(t) for x,t in zip(result['P'],self.T) ]
        result['IOpt_Fwd'] = [ x/df(t) for x,t in zip(result['IOpt'],self.T) ]
        result['POpt_Fwd'] = [ x/df(t) for x,t in zip(result['POpt'],self.T) ]

        return result


# we test the model

today = ql.Settings.getEvaluationDate(ql.Settings.instance())
startDate         = today + 0*365
payDates          = [ startDate + k*365 for k in range(1,11) ]
principalPayments = [ 0.0 for k in range(9) ] + [ 1.0 ]
couponRatePayoff  = Fixed(0.01)
# couponRatePayoff  = Libor(1.0,'1y','CCY') + 0.01
# couponRatePayoff  = Max(Libor(1.0,'1y','CCY'),0.0) + 0.01

instrument        = CallableLoan(startDate,payDates,principalPayments,couponRatePayoff)

res = instrument.npvs()

#table = pd.DataFrame([ res['T'], res['P'], res['I'], res['POpt'], res['IOpt'] ]).T
table = pd.DataFrame([ res['T'], res['P_Fwd'], res['I_Fwd'], res['POpt_Fwd'], res['IOpt_Fwd'], \
                       res['H'], res['delta'], res['Q'] ]).T
table.columns = ['T', 'P', 'I', 'POpt', 'IOpt', 'H', 'delta', 'Q']

print(table)
print('Loan:   %10.8f' % (sum(res['P'])+sum(res['I'])))
print('Option: %10.8f' % (sum(res['POpt'])+sum(res['IOpt'])))
print('H0:     %10.8f' % res['H0'])
