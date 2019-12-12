import QuantLib as ql
import numpy as np

def QuasiGaussianModel(rate, sigma, mean):
    today = ql.Settings.getEvaluationDate(ql.Settings.instance())
    discontYtsH =  discontYtsH = ql.YieldTermStructureHandle( ql.FlatForward(today,rate,ql.Actual365Fixed()) )
    d      = 1
    times  = [  20.0      ]
    sigmas = [ [ sigma ] ]
    slopes = [ [ 0.0    ] ]
    curves = [ [ 0.0    ] ]
    etas   = [   0.0      ]
    deltas = [   0.50     ]
    chis   = [   mean     ]
    Gammas = [ [ 1.00  ]  ]
    theta  = 0.1
    qgModel = ql.QuasiGaussianModel(discontYtsH,d,times,sigmas,slopes,curves,etas,deltas,chis,Gammas,theta)
    return qgModel

def AssetModel(X0, sigma):
    return ql.AssetModel(X0, sigma)

domAlias      = 'EUR'
domRatesModel = QuasiGaussianModel(0.01, 50e-4, 1e-2)

forAliases = ['USD', 'GBP']
forAssetModels = [
    AssetModel(1.0, 0.30),
    AssetModel(2.0, 0.15) ]
forRatesModels = [
    QuasiGaussianModel(0.02, 60e-4, 2e-2),
    QuasiGaussianModel(0.03, 70e-4, 3e-2)
]


# [ EUR, EUR, USD-EUR, USD, USD, GBP-EUR, GBP, GBP ]
#   0    1    2        3    4    5        6    7

correlations = np.identity(8)
#correlations = np.identity(5)
# USD-EUR - USD
correlations[2,3] = -0.5
correlations[3,2] = -0.5
# EUR - USD
correlations[0,3] = -0.3
correlations[3,0] = -0.3
# GBP-EUR - GBP
correlations[5,6] = 0.5
correlations[6,5] = 0.5
# USD - GBP
correlations[3,6] = 0.3
correlations[6,3] = 0.3

model = ql.HybridModel(domAlias,domRatesModel,forAliases,forAssetModels,forRatesModels,correlations)
#model = ql.HybridModel(domAlias,domRatesModel,forAliases[:1],forAssetModels[:1],forRatesModels[:1],correlations)

# we print some basic model stats
print('Hybid Model details:')
print('Size:     ' + str(model.size()))
print('Factors:  ' + str(model.factors()))
print('StartIdx: ' + str(model.modelsStartIdx()))

input('Start MC simulation. Press enter...')
simTimes = np.linspace(0.0,10.0,11)
obsTimes = simTimes
mcsim = ql.RealMCSimulation(model,simTimes,obsTimes,pow(2,13),1234)
mcsim.simulate()
print('Done.')

print('Calculate numeraire adjuster... ', end='')
mcsim.calculateNumeraireAdjuster(simTimes[1:])
print('Done.')

print('Calculate zcb adjuster... ', end='')
mcsim.calculateZCBAdjuster(simTimes[1:],[0.25, 0.5, 1.0, 10.0])
print('Done.')

# asset adjuster not implemented yet

print('Model adjusters:')
print('Numeraire Adjusters: [ %7.1e, %7.1e ]' % \
    (np.amin(mcsim.numeraireAdjuster()),np.amax(mcsim.numeraireAdjuster())))
print('Zcb Adjusters:       [ %7.1e, %7.1e ]' % \
    (np.amin(mcsim.zcbAdjuster()),np.amax(mcsim.zcbAdjuster())))

# we test the forward asset

print('Model asset test:')
spot = ql.RealMCPayoffPricer_NPV([ql.RealMCAsset(10.0,'USD')],mcsim) / \
       forRatesModels[0].termStructure().discount(10.0)
print('USD spot implied 10y: %6.4f' % spot)
spot = ql.RealMCPayoffPricer_NPV([ql.RealMCAsset(10.0,'GBP')],mcsim) / \
       forRatesModels[1].termStructure().discount(10.0)
print('GBP spot implied 10y: %6.4f' % spot)

print('Foreign zcb test:')
pay  = ql.RealMCPay(ql.RealMCMult(ql.RealMCAsset(10.0,'USD'),  \
                    ql.RealMCZeroBond(10.0,20.0,'USD')),10.0)
spot = ql.RealMCPayoffPricer_NPV([pay],mcsim) / \
       forRatesModels[0].termStructure().discount(20.0)
print('USD spot implied 10y into 10y: %6.4f' % spot)
pay  = ql.RealMCPay(ql.RealMCMult(ql.RealMCAsset(10.0,'GBP'),  \
                    ql.RealMCZeroBond(10.0,20.0,'GBP')),10.0)
spot = ql.RealMCPayoffPricer_NPV([pay],mcsim) / \
       forRatesModels[1].termStructure().discount(20.0)
print('GBP spot implied 10y into 10y: %6.4f' % spot)

# correlation tests

eurIndex = ql.Euribor(ql.Period('3m'),domRatesModel.termStructure())
usdIndex  = ql.Euribor(ql.Period('3m'),forRatesModels[0].termStructure())
gbpIndex  = ql.Euribor(ql.Period('3m'),forRatesModels[1].termStructure())

today = ql.Settings.getEvaluationDate(ql.Settings.instance())
expiry = ql.TARGET().advance(today,ql.Period('10y'),ql.Following)
expTime = ql.Actual365Fixed().yearFraction(today,expiry)

eurPayoff = ql.RealMCLiborRateCcy(expTime,eurIndex,domRatesModel.termStructure(),'EUR')
usdPayoff = ql.RealMCLiborRateCcy(expTime,eurIndex,forRatesModels[0].termStructure(),'USD')
gbpPayoff = ql.RealMCLiborRateCcy(expTime,eurIndex,forRatesModels[1].termStructure(),'GBP')


X = ql.RealMCPay(eurPayoff,0.0)
Y = ql.RealMCPay(usdPayoff,0.0)
Z = ql.RealMCPay(gbpPayoff,0.0)
X2 = ql.RealMCMult(X,X)
Y2 = ql.RealMCMult(Y,Y)
Z2 = ql.RealMCMult(Z,Z)
XY = ql.RealMCMult(X,Y)
XZ = ql.RealMCMult(X,Z)
YZ = ql.RealMCMult(Y,Z)

res = ql.RealMCPayoffPricer_NPVs([ X, Y, Z, X2, Y2, Z2, XY, XZ, YZ ],mcsim)
EX  = res[0]
EY  = res[1]
EZ  = res[2]
EX2 = res[3]
EY2 = res[4]
EZ2 = res[5]
EXY = res[6]
EXZ = res[7]
EYZ = res[8]

varX  = EX2 - EX*EX
varY  = EY2 - EY*EY
varZ  = EZ2 - EZ*EZ
covXY = EXY - EX*EY
covXZ = EXZ - EX*EZ
covYZ = EYZ - EY*EZ

corXY  = covXY / np.sqrt(varX*varY)
corXZ  = covXZ / np.sqrt(varX*varZ)
corYZ  = covYZ / np.sqrt(varY*varZ)
sigX  = np.sqrt(varX/10.0)
sigY  = np.sqrt(varY/10.0)
sigZ  = np.sqrt(varZ/10.0)

print('Volatility and correlation test:')
print('Model volatilities (bp) EUR = %4.1f, USD = %4.1f, GBP = %4.1f' % \
      (sigX*1e4, sigY*1e4, sigZ*1e4))
print('Rates correlation EUR-USD = %4.1f, EUR-GBP = %4.1f, USD-GBP = %4.1f' % \
      (corXY*1e2, corXZ*1e2, corYZ*1e2))


# we test a rates only hybrid model

model = ql.HybridModel(domAlias,domRatesModel,[],[],[],[])

print('Degenerate Model details:')
print('Size:     ' + str(model.size()))
print('Factors:  ' + str(model.factors()))
print('StartIdx: ' + str(model.modelsStartIdx()))

input('Start MC simulation. Press enter...')
mcsim = ql.RealMCSimulation(model,simTimes,obsTimes,pow(2,13),1234)
mcsim.simulate()
print('Done.')

print('Calculate numeraire adjuster... ', end='')
mcsim.calculateNumeraireAdjuster(simTimes[1:])
print('Done.')

print('Calculate zcb adjuster... ', end='')
mcsim.calculateZCBAdjuster(simTimes[1:],[0.25, 0.5, 1.0, 10.0])
print('Done.')

print('Model adjusters:')
print('Numeraire Adjusters: [ %7.1e, %7.1e ]' % \
    (np.amin(mcsim.numeraireAdjuster()),np.amax(mcsim.numeraireAdjuster())))
print('Zcb Adjusters:       [ %7.1e, %7.1e ]' % \
    (np.amin(mcsim.zcbAdjuster()),np.amax(mcsim.zcbAdjuster())))

# we test Libor rate estimation

maturity = ql.TARGET().advance(expiry,ql.Period('2d'))
maturity = ql.TARGET().advance(maturity,ql.Period('3m'))
payTime = ql.Actual365Fixed().yearFraction(today,maturity)
discCurve = domRatesModel.termStructure()
DF = discCurve.discount(payTime)

pay1 = ql.RealMCPay(ql.RealMCLiborRate(expTime,eurIndex,discCurve),payTime)
pay2 = ql.RealMCPay(ql.RealMCLiborRateCcy(expTime,eurIndex,discCurve,'EUR'),payTime)

rate0 = eurIndex.fixing(expiry)
rate1 = ql.RealMCPayoffPricer_NPV([pay1],mcsim) / DF
rate2 = ql.RealMCPayoffPricer_NPV([pay2],mcsim) / DF

print('3m Libor rate in 10y (bp): Index = %4.1f, Rates = %4.1f, Hybrid = %4.1f' % \
      (rate0*1e4,rate1*1e4,rate2*1e4))

