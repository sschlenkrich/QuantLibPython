# QuantLibMonteCarlo
#
# A module for payoff scripting and Monte-Carlo Pricing with QuantLib
#
# (C) Sebastian Schlenkrich, December 2019
#

import QuantLib as ql

# an object adapter for QuantLib payoffs
class Payoff:

    # python constructor
    def __init__(self, qlPayoff):
        self.qlPayoff = qlPayoff

    # normal operators

    def __add__(self, other):
        return Payoff(ql.RealMCAxpy(1.0,self.qlPayoff,makeQlPayoff(other)))

    def __sub__(self, other):
        return Payoff(ql.RealMCAxpy(-1.0,makeQlPayoff(other),self.qlPayoff))
    
    def __mul__(self, other):
        if isinstance(other,(int,float)):
            return Payoff(ql.RealMCAxpy(float(other),self.qlPayoff,None))
        else:
            return Payoff(ql.RealMCMult(self.qlPayoff,makeQlPayoff(other)))
    
    def __div__(self, other):
        return Payoff(ql.RealMCDivision(self.qlPayoff,makeQlPayoff(other)))

    # reflective operators

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return Payoff(ql.RealMCAxpy(-1.0,self.qlPayoff,makeQlPayoff(other)))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rdiv__(self, other):
        return Payoff(ql.RealMCDivision(makeQlPayoff(other),self.qlPayoff))

    # use payoff as function

    def __call__(self, obsTime):
        return Payoff(self.qlPayoff.at(obsTime))


# wrappers for QuantLib payoffs

# avoid repeated evaluation if requested for a single path
def Cache(p):
    return Payoff(ql.RealMCCache(makeQlPayoff(p)))

# set payment date for a payoff
def Pay(p, T):
    return Payoff(ql.RealMCPay(makeQlPayoff(p),T))

# define a fixed amount as payoff
def Fixed(amount=0.0, payTime=None):
    if payTime==None:
        return Payoff(ql.RealMCFixedAmount(amount))
    return Payoff(ql.RealMCPay(ql.RealMCFixedAmount(amount),payTime))

def Max(x, y):
    return Payoff(ql.RealMCMax(makeQlPayoff(x),makeQlPayoff(y)))

def Min(x, y):
    return Payoff(ql.RealMCMin(makeQlPayoff(x),makeQlPayoff(y)))

# define a Libor rate without basis spreads; more options may allow capturing basis spreads as well
def Libor(t=1.0, term='1y', alias=None):
        today = ql.Settings.getEvaluationDate(ql.Settings.instance())
        ytsH = ql.YieldTermStructureHandle(ql.FlatForward(today,0.0,ql.Actual365Fixed()))
        index = ql.IborIndex('None',ql.Period(term),0,ql.EURCurrency(),ql.NullCalendar(),ql.Unadjusted,False,ql.Actual365Fixed(),ytsH)
        if alias:
            libor = Payoff(ql.RealMCLiborRateCcy(t,index,ytsH,alias))
        else:
            libor = Payoff(ql.RealMCLiborRate(t,index,ytsH))
        return libor

# define an underlying asset (stock or FX spot)
def Asset(t=0.0, alias=None):
    return Payoff(ql.RealMCAsset(t,alias))

# wrappers for American Monte Carlo valuations

def AmcMax(x, y, z=None, obsTime=0.0, sim=None, maxPolyDegree=2):
    x = __makeQlPayoffList(x)
    y = __makeQlPayoffList(y)
    z = __makeQlPayoffList(z)
    return Payoff(ql.RealAMCMinMax(x,y,z,obsTime,1.0,sim,maxPolyDegree))

def AmcMin(x, y, z=[], obsTime=0.0, sim=None, maxPolyDegree=2):
    x = __makeQlPayoffList(x)
    y = __makeQlPayoffList(y)
    z = __makeQlPayoffList(z)
    return Payoff(ql.RealAMCMinMax(x,y,z,obsTime,-1.0,sim,maxPolyDegree))

def AmcOne(x, y, z=None, obsTime=0.0, largerOrLess=1.0, sim=None, maxPolyDegree=2):
    x = __makeQlPayoffList(x)
    y = __makeQlPayoffList(y)
    z = __makeQlPayoffList(z)
    return Payoff(ql.RealAMCOne(x,y,z,obsTime,largerOrLess,sim,maxPolyDegree))

# conversions to raw QuantLib payoff types

def Raw(arg):
    if isinstance(arg,Payoff):
        return arg.qlPayoff
    if isinstance(arg,list):
        return [ p.qlPayoff for p in arg ]
    raise TypeError('Error function Raw: unsupported input type.')

# test input and try to convert to required QuantLib input
def __makeQlPayoffList(arg):
    # we need a list of payoffs
    argList = arg if isinstance(arg,list) else [arg]
    # arguments are assumed to be of type Payoff
    resList = [ makeQlPayoff(p) for p in argList ]
    # maybe some more checks and conversions...
    return resList

# test input and try to convert to required QuantLib input
def makeQlPayoff(arg):
    if isinstance(arg,ql.RealMCPayoff):
        return arg
    if isinstance(arg,Payoff):
        return arg.qlPayoff
    if isinstance(arg,(int,float)):
        return ql.RealMCFixedAmount(float(arg))
    raise TypeError('Error class Payoff: unsupported input type.')



