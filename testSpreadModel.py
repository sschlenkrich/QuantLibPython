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
correlations[0,2] = 0.1
correlations[2,0] = 0.1

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

pay = ql.RealMCPay(ql.RealMCFixedAmount(1.0),10.0)
dfMc = ql.RealMCPayoffPricer_NPV([pay],mcsim)
dfCv = domRatesModel.termStructure().discount(10.0) * \
       creditModel.termStructure().discount(10.0)
print(dfMc)
print(dfCv)

print('Test Forward Libors')

today = ql.Settings.getEvaluationDate(ql.Settings.instance())
index = ql.IborIndex('CIbor',ql.Period('3m'),0,
    ql.EURCurrency(),ql.NullCalendar(),ql.Following,False,ql.Actual360(),
    domRatesModel.termStructure())
indexPayoff = ql.RealMCLiborRateCcy(1.0,index,domRatesModel.termStructure(),'EUR')

times = np.linspace(1,10,10)
dates = [ today + int(t*365) for t in times ]
matis = [ index.maturityDate(d) for d in dates ]
time = lambda d : ql.Actual365Fixed().yearFraction(today,d)

payoffs = [ ql.RealMCPay(ql.RealMCLiborRateCcy(
    time(d),index,domRatesModel.termStructure(),'EUR'),time(m))
    for d,m in zip(dates,matis) ]
keys = [ 'Libor'+str(k) for k in range(len(payoffs)) ]
script = ql.RealMCScript(keys,payoffs,[],True)

input('Start MC simulation. Press enter...')
simTimes = script.observationTimes(keys)
obsTimes = simTimes
mcsim = ql.RealMCSimulation(model,simTimes,obsTimes,pow(2,15),1234)
mcsim.simulate()
print('Done.')

print('Price payoffs without numeraire adjuster... ', end='')
npv = script.NPV(mcsim,keys)
fwdLib = [ pv / domRatesModel.termStructure().discount(d) / creditModel.termStructure().discount(d) 
           for pv,d in zip(npv,matis) ]
indLib = [ index.fixing(d) for d in dates ]
print('Done.')

libTimes0 = np.array([ time(d) for d in dates ])
libDiffs0 = np.array([ a-b for a,b in zip(indLib,fwdLib) ])

print('Calculate numeraire adjuster... ', end='')
mcsim.calculateNumeraireAdjuster(simTimes[1:])
print('Done.')

print('Price payoffs with numeraire adjuster... ', end='')
npv = script.NPV(mcsim,keys)
fwdLib = [ pv / domRatesModel.termStructure().discount(d) / creditModel.termStructure().discount(d) 
           for pv,d in zip(npv,matis) ]
indLib = [ index.fixing(d) for d in dates ]
print('Done.')

libTimes1 = np.array([ time(d) for d in dates ])
libDiffs1 = np.array([ a-b for a,b in zip(indLib,fwdLib) ])

plt.figure()
plt.plot(times,nuadj*1e4, label='numeraire adjuster')
plt.plot(libTimes0,libDiffs0*1e4, label='libor difference')
plt.plot(libTimes1,libDiffs1*1e4, label='libor difference adj')

plt.legend()
plt.show()


