import QuantLib as ql
import numpy as np
import matplotlib.pyplot as plt

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

# model setup
domAlias      = 'EUR'
domRatesModel = QuasiGaussianModel(0.02, 50e-4, 1e-2)
hybModel      = ql.HybridModel(domAlias,domRatesModel,[],[],[],[])
creditModel   = QuasiGaussianModel(0.01, 50e-4, 1e-2)

# correlation [ x, x, c, c ]
correlations = np.identity(4)
# rates - credit corr
correlations[0,2] = -0.5
correlations[2,0] = -0.5

model         = ql.SpreadModel(hybModel,creditModel,correlations)

print('Hybid Model details:')
print('Size:     ' + str(hybModel.size()))
print('Factors:  ' + str(hybModel.factors()))
print('StartIdx: ' + str(hybModel.modelsStartIdx()))

print('CreditHybid Model details:')
print('Size:     ' + str(model.size()))
print('Factors:  ' + str(model.factors()))
print('State details:')
for a in model.stateAliases():
    print(a)
print('Factor details:')
for a in model.factorAliases():
    print(a)


input('Start MC simulation. Press enter...')
simTimes = np.linspace(0.0,10.0,11)
obsTimes = simTimes
mcsim = ql.RealMCSimulation(model,simTimes,obsTimes,pow(2,13),1234)
mcsim.simulate()
print('Done.')

print('Calculate numeraire adjuster... ', end='')
mcsim.calculateNumeraireAdjuster(simTimes[1:])
print('Done.')

times = np.array(simTimes[1:])
nuadj = np.array(mcsim.numeraireAdjuster())
plt.plot(times,nuadj*1e4)
plt.show()

pay = ql.RealMCPay(ql.RealMCFixedAmount(1.0),10.0)
dfMc = ql.RealMCPayoffPricer_NPV([pay],mcsim)
dfCv = domRatesModel.termStructure().discount(10.0) * \
       creditModel.termStructure().discount(10.0)
print(dfMc)
print(dfCv)

