import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__),'QuantLibWrapper'))

import pandas
import numpy as np
from scipy.optimize import least_squares
from scipy.optimize import brentq
from scipy.stats import norm

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3d plotting
from matplotlib import cm


import QuantLib as ql

from QuantLibWrapper import YieldCurve, createSwaption, Payoffs, CashSettledSwaptionPayoff, \
     CubicSplineExactIntegration, EuropeanPayoff, HullWhiteModel, HullWhiteModelFromSwaption, \
     CashPhysicalSwitchPayoff

today = ql.DateParser.parseISO('2019-03-20')
ql.Settings.instance().evaluationDate = today

CurveData = [
    # term   date           ois zero      swap zero     swap zero (single curve)
    [ '0d',  '2019-03-20', -0.003680519, -0.002395740, -0.002394957 ],
    [ '1Y',  '2020-03-20', -0.003567616, -0.002211397, -0.002210564 ],
    [ '2Y',  '2021-03-20', -0.003123120, -0.001676478, -0.001675351 ],
    [ '3Y',  '2022-03-20', -0.002394343, -0.000879961, -0.000877958 ],
    [ '4Y',  '2023-03-20', -0.001562972,  0.000018115,  0.000021648 ],
    [ '5Y',  '2024-03-20', -0.000690034,  0.000967101,  0.000972941 ],
    [ '6Y',  '2025-03-20',  0.000260744,  0.001953177,  0.001962094 ],
    [ '7Y',  '2026-03-20',  0.001263721,  0.002961077,  0.002973712 ],
    [ '8Y',  '2027-03-20',  0.002292005,  0.003973743,  0.003990589 ],
    [ '9Y',  '2028-03-20',  0.003313020,  0.004967610,  0.004988944 ],
    [ '10Y', '2029-03-20',  0.004277978,  0.005907654,  0.005933485 ],
    [ '11Y', '2030-03-20',  0.005197139,  0.006802629,  0.006833029 ],
    [ '12Y', '2031-03-20',  0.006033906,  0.007616502,  0.007651195 ],
    [ '13Y', '2032-03-20',  0.006794605,  0.008359102,  0.008397895 ],
    [ '14Y', '2033-03-20',  0.007476810,  0.009009114,  0.009051466 ],
    [ '15Y', '2034-03-20',  0.008080836,  0.009578170,  0.009623506 ],
    [ '16Y', '2035-03-20',  0.008607018,  0.010075744,  0.010123578 ],
    [ '17Y', '2036-03-20',  0.009063400,  0.010496578,  0.010546319 ],
    [ '18Y', '2037-03-20',  0.009454450,  0.010859459,  0.010910640 ],
    [ '19Y', '2038-03-20',  0.009786994,  0.011169176,  0.011221422 ],
    [ '20Y', '2039-03-20',  0.010065419,  0.011412768,  0.011465490 ],
    [ '21Y', '2040-03-20',  0.010294856,  0.011611397,  0.011664186 ],
    [ '22Y', '2041-03-20',  0.010481735,  0.011779131,  0.011831762 ],
    [ '23Y', '2042-03-20',  0.010634034,  0.011901576,  0.011953648 ],
    [ '24Y', '2043-03-20',  0.010757823,  0.012003226,  0.012054593 ],
    [ '25Y', '2044-03-20',  0.010858452,  0.012081271,  0.012131770 ],
    [ '26Y', '2045-03-20',  0.010939556,  0.012149813,  0.012199424 ],
    [ '27Y', '2046-03-20',  0.011004452,  0.012195677,  0.012244229 ],
    [ '28Y', '2047-03-20',  0.011055588,  0.012230537,  0.012277975 ],
    [ '29Y', '2048-03-20',  0.011095169,  0.012253048,  0.012299314 ],
    [ '30Y', '2049-03-20',  0.011124809,  0.012265616,  0.012310646 ],
    [ '31Y', '2050-03-20',  0.011146187,  0.012267353,  0.012311077 ],
    [ '32Y', '2051-03-20',  0.011160505,  0.012260340,  0.012302715 ],
    [ '33Y', '2052-03-20',  0.011168798,  0.012247525,  0.012288537 ],
    [ '34Y', '2053-03-20',  0.011171920,  0.012231622,  0.012271293 ],
    [ '35Y', '2054-03-20',  0.011170699,  0.012214937,  0.012253311 ],
    [ '36Y', '2055-03-20',  0.011165843,  0.012199045,  0.012236180 ],
    [ '37Y', '2056-03-20',  0.011157957,  0.012183298,  0.012219238 ],
    [ '38Y', '2057-03-20',  0.011147651,  0.012166738,  0.012201517 ],
    [ '39Y', '2058-03-20',  0.011135414,  0.012148389,  0.012182023 ],
    [ '40Y', '2059-03-20',  0.011121703,  0.012127410,  0.012159900 ],
    [ '41Y', '2060-03-20',  0.011106859,  0.012103205,  0.012134541 ],
    [ '42Y', '2061-03-20',  0.011091207,  0.012076370,  0.012106551 ],
    [ '43Y', '2062-03-20',  0.011074896,  0.012047496,  0.012076528 ],
    [ '44Y', '2063-03-20',  0.011058102,  0.012017179,  0.012045075 ],
    [ '45Y', '2064-03-20',  0.011040934,  0.011985876,  0.012012652 ],
    [ '46Y', '2065-03-20',  0.011023631,  0.011954253,  0.011979937 ],
    [ '47Y', '2066-03-20',  0.011006280,  0.011922679,  0.011947303 ],
    [ '48Y', '2067-03-20',  0.010989000,  0.011891570,  0.011915168 ],
    [ '49Y', '2068-03-20',  0.010971858,  0.011861225,  0.011883836 ],
    [ '50Y', '2069-03-20',  0.010955048,  0.011832163,  0.011853833 ],
    [ '51Y', '2070-03-20',  0.010938605,  0.011804550,  0.011825327 ],
    [ '52Y', '2071-03-20',  0.010922548,  0.011778338,  0.011798267 ],
    [ '53Y', '2072-03-20',  0.010906841,  0.011753344,  0.011772464 ],
    [ '54Y', '2073-03-20',  0.010891573,  0.011729597,  0.011747947 ],
    [ '55Y', '2074-03-20',  0.010876705,  0.011706927,  0.011724542 ],
    [ '56Y', '2075-03-20',  0.010862240,  0.011685241,  0.011702152 ],
    [ '57Y', '2076-03-20',  0.010848146,  0.011664397,  0.011680632 ],
    [ '58Y', '2077-03-20',  0.010834501,  0.011644430,  0.011660016 ],
    [ '59Y', '2078-03-20',  0.010821271,  0.011625205,  0.011640167 ],
    [ '60Y', '2079-03-20',  0.010808457,  0.011606653,  0.011621013 ],
    [ '70Y', '2089-03-20',  0.010808457,  0.011606653,  0.011621013 ] ]


oisYTS = ql.MonotonicCubicZeroCurve(
    [ ql.DateParser.parseISO(c[1]) for c in CurveData ],
    [ c[2]                         for c in CurveData ],
    ql.Actual365Fixed(), ql.NullCalendar())
oisCurve = YieldCurve(['1y'],[0.0]) # dummy
oisCurve.yts = oisYTS

swapYTS = ql.MonotonicCubicZeroCurve(
    [ ql.DateParser.parseISO(c[1]) for c in CurveData ],
    [ c[3]                         for c in CurveData ],
    ql.Actual365Fixed(), ql.NullCalendar())
swapCurve = YieldCurve(['1y'],[0.0]) # dummy
swapCurve.yts = swapYTS

expiryTerms = [
    '1M' ,
    '2M' ,
    '3M' ,
    '6M' ,
    '9M' ,
    '1Y' ,
    '18M',
    '2Y' ,
    '3Y' ,
    '4Y' ,
    '5Y' ,
    '7Y' ,
    '10Y',
    '15Y',
    '20Y',
    '25Y',
    '30Y' ]

swapTerms = [
    '1Y' ,
    '2Y' ,
    '3Y' ,
    '4Y' ,
    '5Y' ,
    '6Y' ,
    '7Y' ,
    '8Y' ,
    '9Y' ,
    '10Y',
    '15Y',
    '20Y',
    '25Y',
    '30Y' ]

physicalySettledPremium = [
    [ 1.0, 4.0, 11.0, 19.0, 28.5, 37.5, 46.5, 56.5, 66.0, 76.0, 112.0, 148.0, 181.0, 209.0                      ],
    [ 2.5, 6.5, 15.0, 25.5, 37.5, 50.0, 62.0, 75.0, 89.0, 102.0, 152.0, 202.0, 246.0, 283.0                     ],
    [ 3.0, 8.5, 20.5, 34.5, 49.0, 65.0, 80.5, 97.5, 114.0, 132.0, 195.0, 255.0, 310.0, 357.0                    ],
    [ 5.5, 15.0, 32.5, 52.5, 75.5, 98.5, 122.0, 145.0, 170.0, 196.0, 289.0, 379.0, 458.0, 525.0                 ],
    [ 8.5, 22.5, 44.0, 70.5, 98.5, 125.0, 155.0, 183.0, 213.0, 246.0, 360.0, 469.0, 568.0, 658.0                ],
    [ 12.0, 31.0, 56.5, 86.5, 120.0, 150.0, 185.0, 217.0, 251.0, 289.0, 423.0, 554.0, 669.0, 771.0              ],
    [ 20.0, 47.5, 82.5, 121.0, 161.0, 203.0, 245.0, 286.0, 326.0, 371.0, 536.0, 692.0, 834.0, 968.0             ],
    [ 28.5, 65.5, 108.0, 154.0, 199.0, 249.0, 296.0, 346.0, 395.0, 442.0, 635.0, 817.0, 984.0, 1136.0           ],
    [ 46.0, 100.0, 158.0, 218.0, 277.0, 340.0, 399.0, 459.0, 520.0, 580.0, 810.0, 1033.0, 1239.0, 1434.0        ],
    [ 64.5, 133.0, 207.0, 276.0, 351.0, 424.0, 493.0, 568.0, 637.0, 704.0, 967.0, 1222.0, 1463.0, 1697.0        ],
    [ 81.5, 162.0, 246.0, 330.0, 416.0, 498.0, 580.0, 661.0, 740.0, 814.0, 1104.0, 1390.0, 1662.0, 1933.0       ],
    [ 108.0, 213.0, 320.0, 424.0, 525.0, 625.0, 719.0, 815.0, 910.0, 1000.0, 1344.0, 1683.0, 2003.0, 2322.0     ],
    [ 136.5, 268.0, 397.0, 523.0, 649.0, 769.0, 887.0, 999.0, 1108.0, 1223.0, 1646.0, 2058.0, 2437.0, 2823.0    ],
    [ 164.0, 323.0, 479.0, 629.0, 776.0, 921.0, 1058.0, 1194.0, 1330.0, 1467.0, 1984.0, 2471.0, 2928.0, 3355.0  ],
    [ 182.0, 359.0, 533.0, 699.0, 865.0, 1032.0, 1189.0, 1345.0, 1498.0, 1644.0, 2246.0, 2749.0, 3245.0, 3707.0 ],
    [ 194.5, 386.0, 572.0, 749.0, 924.0, 1098.0, 1263.0, 1425.0, 1589.0, 1745.0, 2392.0, 2904.0, 3423.0, 3908.0 ],
    [ 202.5, 400.0, 596.0, 785.0, 971.0, 1148.0, 1317.0, 1474.0, 1633.0, 1795.0, 2469.0, 2996.0, 3533.0, 4046.0 ] ]

cashSettledPremium = [
    [ 1.0, 4.0, 11.0, 19.0, 28.5, 37.0, 46.0, 55.5, 64.5, 74.0, 108.0, 141.0, 172.0, 199.0                      ],
    [ 2.5, 6.5, 15.0, 25.5, 37.0, 49.5, 61.0, 73.5, 87.0, 100.0, 146.0, 192.0, 234.0, 269.0                     ],
    [ 3.0, 8.5, 20.5, 34.5, 48.5, 64.5, 79.5, 96.0, 112.0, 129.0, 187.0, 242.0, 294.0, 340.0                    ],
    [ 5.5, 15.0, 32.5, 52.0, 75.0, 97.5, 120.0, 142.0, 166.0, 191.0, 278.0, 361.0, 436.0, 500.0                 ],
    [ 8.5, 22.5, 43.5, 70.0, 97.5, 124.0, 153.0, 180.0, 208.0, 240.0, 346.0, 448.0, 542.0, 629.0                ],
    [ 12.0, 31.0, 56.0, 86.0, 119.0, 148.0, 182.0, 213.0, 246.0, 282.0, 407.0, 530.0, 640.0, 739.0              ],
    [ 20.0, 47.5, 82.0, 120.0, 160.0, 201.0, 241.0, 281.0, 319.0, 362.0, 518.0, 665.0, 802.0, 933.0             ],
    [ 28.5, 65.5, 107.0, 153.0, 197.0, 246.0, 292.0, 340.0, 387.0, 432.0, 615.0, 788.0, 949.0, 1098.0           ],
    [ 46.0, 99.0, 157.0, 216.0, 274.0, 336.0, 393.0, 451.0, 510.0, 568.0, 788.0, 1003.0, 1203.0, 1394.0         ],
    [ 64.5, 132.0, 206.0, 274.0, 348.0, 419.0, 486.0, 559.0, 626.0, 691.0, 945.0, 1192.0, 1428.0, 1658.0        ],
    [ 81.0, 161.0, 245.0, 328.0, 413.0, 493.0, 573.0, 652.0, 729.0, 801.0, 1085.0, 1363.0, 1631.0, 1897.0       ],
    [ 108.0, 213.0, 318.0, 422.0, 522.0, 620.0, 713.0, 807.0, 900.0, 989.0, 1328.0, 1662.0, 1979.0, 2291.0      ],
    [ 136.5, 267.0, 396.0, 521.0, 646.0, 765.0, 881.0, 992.0, 1100.0, 1214.0, 1633.0, 2040.0, 2412.0, 2786.0    ],
    [ 164.0, 322.0, 478.0, 628.0, 774.0, 919.0, 1055.0, 1189.0, 1325.0, 1461.0, 1973.0, 2450.0, 2890.0, 3296.0  ],
    [ 182.0, 358.0, 532.0, 698.0, 864.0, 1030.0, 1184.0, 1339.0, 1491.0, 1635.0, 2227.0, 2713.0, 3184.0, 3614.0 ],
    [ 194.5, 384.0, 571.0, 748.0, 922.0, 1095.0, 1259.0, 1418.0, 1580.0, 1733.0, 2364.0, 2857.0, 3342.0, 3788.0 ],
    [ 202.5, 400.0, 595.0, 784.0, 969.0, 1144.0, 1311.0, 1465.0, 1621.0, 1780.0, 2436.0, 2938.0, 3437.0, 3904.0 ] ]


class PremiumHelper:
    # Python constructor
    def __init__(self, expiryTerm, swapTerm, physicalStraddle, cashStraddle):
        self.expiryTerm = expiryTerm
        self.swapTerm   = swapTerm
        self.physicalStraddle = physicalStraddle
        self.cashStraddle = cashStraddle
        # first we back out implied normal volatility from physical price
        swaption = createSwaption(self.expiryTerm,self.swapTerm,oisCurve,swapCurve,'ATM',ql.VanillaSwap.Receiver)
        self.annuity = swaption.annuity()
        self.expiryTime = ql.Actual365Fixed().yearFraction(today,swaption.exercise.dates()[0])
        self.discountToExpiry = oisCurve.discount( self.expiryTime )
        fwdStraddle = self.physicalStraddle * self.discountToExpiry / self.annuity
        self.normalVolatility = fwdStraddle / np.sqrt(2.0*self.expiryTime/np.pi)
        # now we can set up reference swaptions
        self.payer = createSwaption(self.expiryTerm,self.swapTerm,oisCurve,swapCurve,'ATM',ql.VanillaSwap.Payer,self.normalVolatility)
        self.receiver = createSwaption(self.expiryTerm,self.swapTerm,oisCurve,swapCurve,'ATM',ql.VanillaSwap.Receiver,self.normalVolatility)
        # we specify a valuation method
        self.method = CubicSplineExactIntegration(None,101,5)

    def HullWhiteModel(self, meanReversion):
        return HullWhiteModelFromSwaption(self.payer,meanReversion)

    def physicalStraddleModel(self, hwModel):
        payerPysicalPayoff     = Payoffs.Max(self.payer.payoff(hwModel), Payoffs.Zero())
        receiverPhysicalPayoff = Payoffs.Max(self.receiver.payoff(hwModel), Payoffs.Zero())
        self.method.hwModel = hwModel
        return 1.0 / self.discountToExpiry * \
                ( EuropeanPayoff(self.expiryTime,payerPysicalPayoff,self.method).npv() + \
                  EuropeanPayoff(self.expiryTime,receiverPhysicalPayoff,self.method).npv() )

    def cashStraddleModel(self, hwModel):
        payerCashPayoff        = Payoffs.Max(CashSettledSwaptionPayoff(self.payer,hwModel), Payoffs.Zero())
        receiverCashPayoff     = Payoffs.Max(CashSettledSwaptionPayoff(self.receiver,hwModel), Payoffs.Zero())
        self.method.hwModel = hwModel
        return 1.0 / self.discountToExpiry * \
                  ( EuropeanPayoff(self.expiryTime,payerCashPayoff,self.method).npv() + \
                    EuropeanPayoff(self.expiryTime,receiverCashPayoff,self.method).npv() )

    def cashPhysicalSwitchModel(self, hwModel):
        cashPhysicalSwitchPayoff = CashPhysicalSwitchPayoff(self.payer,hwModel)
        self.method.hwModel = hwModel
        return 1.0 / self.discountToExpiry * \
                  EuropeanPayoff(self.expiryTime,cashPhysicalSwitchPayoff,self.method).npv() 

    def objective(self, meanReversion):
        if abs(meanReversion) < 1.0-12:
            meanReversion = 1.0e-12
        return self.cashPhysicalSwitchModel(self.HullWhiteModel(meanReversion)) - \
               ( self.physicalStraddle - self.cashStraddle )

    def meanReversion(self):
        # we try a couple of intervalls...
        lowHighs = [ [-0.15,-0.10], [-0.10,-0.05], [-0.05,-0.01], [-0.01,0.01], [0.01,0.05], [0.05,0.1], [0.10,0.15] ]
        res = None
        for lowHigh in lowHighs:
            try:
                res = brentq(self.objective,lowHigh[0],lowHigh[1],xtol=1.0e-8)
            except:
                res = None
            if not res==None:
                return res
        raise Exception('No mean reversion found!')

def meanReversion(swapTime1, swapTime2):
    # assume swapTime1/2 in full years
    expiryTimes = [  2,  3,  4,  5,  7,  10,  15,  20,  25,  30 ]
    meanReversions = []
    meanReversionsSimple = []
    for expiryTime in expiryTimes:
        expiryTerm = str(expiryTime) + 'Y'
        swapTerm1  = str(swapTime1)  + 'Y'
        swapTerm2  = str(swapTime2)  + 'Y'
        helper1 = PremiumHelper(expiryTerm, swapTerm1,
                     physicalySettledPremium[expiryTerms.index(expiryTerm)][swapTerms.index(swapTerm1)],
                     cashSettledPremium[expiryTerms.index(expiryTerm)][swapTerms.index(swapTerm1)])
        helper2 = PremiumHelper(expiryTerm, swapTerm2,
                     physicalySettledPremium[expiryTerms.index(expiryTerm)][swapTerms.index(swapTerm2)],
                     cashSettledPremium[expiryTerms.index(expiryTerm)][swapTerms.index(swapTerm2)])
        lambda_ = helper2.normalVolatility / helper1.normalVolatility
        def objective(meanReversion):
            num = (1.0-np.exp(-meanReversion*swapTime2)) / swapTime2
            den = (1.0-np.exp(-meanReversion*swapTime1)) / swapTime1
            return num/den - lambda_
        meanReversions.append( 100*brentq(objective,-0.05,0.15,xtol=1.0e-8) )
        meanReversionsSimple.append( 100*(-2.0)*np.log(lambda_)/(swapTime2-swapTime1) )
    return expiryTimes, meanReversions, meanReversionsSimple



swapPairs = [ [5,10], [10,15], [15,20], [20,25], [25,30] ]
swapPairs = [ [5,15], [10,20], [15,25], [20,30] ]
plt.figure(figsize=(8, 4))
table  = [ [2.0,3.0,4.0,5.0,7.0,10.0,15.0,20.0,25.0,30.0] ]
labels = [ 'ExpiryTimes' ]
for item in swapPairs:
    expiryTimes, meanReversions, meanReversionsSimple = meanReversion(item[0],item[1])
    plt.figure(figsize=(8, 4))
    plt.plot(expiryTimes,meanReversionsSimple,'b--*',label=str(item[0])+'Y' + ' vs. ' + str(item[1])+'Y')
    plt.plot(expiryTimes,meanReversions,'b-*',label=str(item[0])+'Y' + ' vs. ' + str(item[1])+'Y')
    plt.xlim(0.0,32.0)
    plt.ylim(-10.0,5.0)
    plt.xlabel('time to expiry (y)')
    plt.ylabel('implied mean reversion (%)')
    #plt.legend()
    plt.title(str(item[0])+'Y' + ' vs. ' + str(item[1])+'Y' + ' swap term')
    plt.tight_layout()
    # save for later use
    table  = table  + [ meanReversions, meanReversionsSimple ]
    labels = labels + [ str(item[0])+'Y vs. '+str(item[1])+'Y',
                        str(item[0])+'Y vs. '+str(item[1])+'Y (simple)', ]
frame = pandas.DataFrame(table).T
frame.columns = labels
frame.to_csv('MeanReversionFromVolRatios.csv', index=False)
plt.show()
#exit()

vols = [ ]
for j in range(len(swapTerms)):
    volsPerTerm = []
    for i in range(len(expiryTerms)):
        helper = PremiumHelper(expiryTerms[i], swapTerms[j],
                     physicalySettledPremium[i][j], cashSettledPremium[i][j])
        volsPerTerm.append(helper.normalVolatility)
    vols.append(volsPerTerm)
expiryTimes = [ (float(term[:-1]) if term[-1]=='Y' else float(term[:-1])/12.0) for term in expiryTerms ]
swapTimes   = [ int(term[:-1]) for term in swapTerms   ]
table = [ expiryTimes ] + vols
labels = [ 'ExpiryTimes' ] + swapTerms
frame = pandas.DataFrame(table).T
frame.columns = labels
frame.to_csv('ImpledATMVolatilities.csv', index=False)
fig = plt.figure()
ax = fig.gca(projection='3d')
X, Y = np.meshgrid(expiryTimes,swapTimes,indexing='ij')
surf = ax.plot_surface(X, Y, 1e4*np.transpose(np.array(vols)), cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.set_xlim(0, 30)
ax.set_ylim(0, 30)
#ax.set_zlim(50, 150)
ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
ax.set_yticks([0, 5, 10, 15, 20, 25, 30])
ax.set_xlabel('Expiries (y)')
ax.set_ylabel('Swap terms (y)')
ax.set_zlabel('Market-implied normal volatility (bp)')
plt.show()
#exit()

# we analyse switch prices for a few examples
swapPairs = [ ['10Y', '10Y', 'r'], ['20Y', '20Y', 'b'] ]
plt.figure(figsize=(6, 4))
for item in swapPairs:
    helper = PremiumHelper(item[0], item[1],
                physicalySettledPremium[expiryTerms.index(item[0])][swapTerms.index(item[1])],
                cashSettledPremium[expiryTerms.index(item[0])][swapTerms.index(item[1])])
    meanReversions = np.linspace(-0.05, 0.10, 20)
    cashPhysicalSw = np.array([helper.cashPhysicalSwitchModel(helper.HullWhiteModel(mr)) for mr in meanReversions ])
    marketpriceSw  = np.array([helper.physicalStraddle - helper.cashStraddle for mr in meanReversions ])
    plt.plot(meanReversions*100,cashPhysicalSw, item[2]+'-',label=item[0] + '-' + item[1])
    plt.plot(meanReversions*100,marketpriceSw,  item[2]+'--')
    plt.xlim(-6.0,11.0)
    #plt.ylim(-10.0,5.0)
    plt.xlabel('mean reversion (%)')
    plt.ylabel('cash-physical-switch price')
    plt.legend()
    #plt.title(item[0] + '-' + item[1] + ' swaption')
    plt.tight_layout()
plt.show()

# how does the physical price impact the switch value???
swapPairs = [ ['10Y', '10Y', 'r'], ['20Y', '20Y', 'b'] ]
discount  = 0.8
plt.figure(figsize=(6, 4))
for item in swapPairs:
    meanReversions = np.linspace(-0.05, 0.10, 20)
    physPrem = physicalySettledPremium[expiryTerms.index(item[0])][swapTerms.index(item[1])]
    cashPrem = cashSettledPremium[expiryTerms.index(item[0])][swapTerms.index(item[1])]
    switch   = physPrem - cashPrem
    helper = PremiumHelper(item[0], item[1], physPrem, cashPrem)
    cashPhysicalSw = np.array([helper.cashPhysicalSwitchModel(helper.HullWhiteModel(mr)) for mr in meanReversions ])
    plt.plot(meanReversions*100,cashPhysicalSw, item[2]+'-',label=item[0] + '-' + item[1])
    # now with discount
    helper = PremiumHelper(item[0], item[1], discount*physPrem, discount*physPrem-switch)
    cashPhysicalSw = np.array([helper.cashPhysicalSwitchModel(helper.HullWhiteModel(mr)) for mr in meanReversions ])
    plt.plot(meanReversions*100,cashPhysicalSw, item[2]+'--')
    plt.xlim(-6.0,11.0)
    #plt.ylim(-10.0,5.0)
    plt.xlabel('mean reversion (%)')
    plt.ylabel('cash-physical-switch price')
    plt.legend()
    #plt.title(item[0] + '-' + item[1] + ' swaption')
    plt.tight_layout()
plt.show()
#exit()

# also we want to get an idea of how the payoff looks like depending on mean reversion
meanReversions = [ -0.05, 0.025, 0.10 ]
colors         = [   'r',   'r',  'r' ]
item = ['20Y', '20Y']
timeToExpiry = 20.0   # set this consistent to item
for meanReversion, color in zip(meanReversions,colors):
    plt.figure(figsize=(8, 4))
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    helper = PremiumHelper(item[0], item[1],
                physicalySettledPremium[expiryTerms.index(item[0])][swapTerms.index(item[1])],
                cashSettledPremium[expiryTerms.index(item[0])][swapTerms.index(item[1])])
    hwModel = helper.HullWhiteModel(meanReversion)
    cashPhysicalSwitchPayoff = CashPhysicalSwitchPayoff(helper.payer,hwModel)
    states = np.linspace(-0.10,0.10,201)
    payoff = np.array([ cashPhysicalSwitchPayoff.at([x,0.0]) for x in states ])
    ax1.plot(states*100,payoff,color+'-',label='MR: '+str(meanReversion*100)+'%')
    # we also need to consider the change in distribution
    mu    = hwModel.expectationX(0.0,0.0,timeToExpiry)
    sigma = np.sqrt(hwModel.varianceX(0.0,timeToExpiry))
    pdf = np.array([ norm.pdf((x-mu)/sigma)/sigma for x in states ])
    ax2.plot(states*100,pdf,color+'--')

    ax1.set_xlabel('state variable (%)')
    ax1.set_ylabel('cash-physical-switch payoff')
    ax1.set_ylim(-500,500)
    #ax1.legend()
    ax2.set_ylabel('pdf')
    ax2.set_ylim(0,35)
    plt.title('MR: '+str(meanReversion*100)+'%')
    plt.tight_layout()
plt.show()

#exit()

# we calculate implied mean reversion for a couple of swap terms an expiries
expiries = {}
expiries['10Y'] = [                               '10Y', '15Y', '20Y', '25Y', '30Y' ]
expiries['15Y'] = [                         '7Y', '10Y', '15Y', '20Y', '25Y', '30Y' ]
expiries['20Y'] = [             '4Y', '5Y', '7Y', '10Y', '15Y', '20Y', '25Y', '30Y' ]
expiries['25Y'] = [ '2Y', '3Y', '4Y', '5Y', '7Y', '10Y', '15Y', '20Y', '25Y', '30Y' ]
expiries['30Y'] = [ '2Y', '3Y', '4Y', '5Y', '7Y', '10Y', '15Y', '20Y', '25Y', '30Y' ]

expTimes = {}
expTimes['10Y'] = [                                10.0,  15.0,  20.0,  25.0,  30.0 ]
expTimes['15Y'] = [                          7.0,  10.0,  15.0,  20.0,  25.0,  30.0 ]
expTimes['20Y'] = [              4.0,  5.0,  7.0,  10.0,  15.0,  20.0,  25.0,  30.0 ]
expTimes['25Y'] = [  2.0,  3.0,  4.0,  5.0,  7.0,  10.0,  15.0,  20.0,  25.0,  30.0 ]
expTimes['30Y'] = [  2.0,  3.0,  4.0,  5.0,  7.0,  10.0,  15.0,  20.0,  25.0,  30.0 ]

swaps = ['10Y', '15Y', '20Y', '25Y', '30Y']
#swaps = ['10Y' ]

table = []
labels = []
for swapTerm in swaps:
    print(swapTerm)
    meanReversions = []
    variances = []
    for expiry in expiries[swapTerm]:
        helper = PremiumHelper(expiry, swapTerm,
                     physicalySettledPremium[expiryTerms.index(expiry)][swapTerms.index(swapTerm)],
                     cashSettledPremium[expiryTerms.index(expiry)][swapTerms.index(swapTerm)])
        meanReversion = helper.meanReversion()
        sw1 = helper.cashPhysicalSwitchModel(helper.HullWhiteModel(meanReversion-0.01))
        sw2 = helper.cashPhysicalSwitchModel(helper.HullWhiteModel(meanReversion+0.01))
        variance = 0.02 / (sw2 - sw1)
        print(expiry + '-' + swapTerm + ': MR: ' + str(meanReversion) + ', Var: ' + str(variance), flush=True )
        meanReversions.append(meanReversion*100.0)
        variances.append(variance*100.0)
    # we plot the results
    plt.figure(figsize=(8, 4))
    plt.errorbar(expTimes[swapTerm],meanReversions,variances,fmt='r-*')
    plt.xlim(0.0,32.0)
    plt.ylim(-10.0,5.0)
    plt.xlabel('time to expiry (y)')
    plt.ylabel('implied mean reversion (%)')
    plt.title(swapTerm + ' swap term')
    plt.tight_layout()
    # save data for later use
    table = table + [expTimes[swapTerm], meanReversions, variances ]
    labels = labels + [ swapTerm+'-Times', swapTerm+'-MeanReversion', swapTerm+'-Variance' ]
plt.show()
frame = pandas.DataFrame(table).T
frame.columns = labels
frame.to_csv('MeanReversionFromCPSwitches.csv', index=False)


# we use saved data and create additional plots

frame1 = pandas.read_csv('MeanReversionFromVolRatios.csv')
frame2 = pandas.read_csv('MeanReversionFromCPSwitches.csv')

print(frame1.columns)

for k in range(5):
    plt.figure(figsize=(6, 4))
    plt.errorbar(frame2.iloc[:,3*k+0],frame2.iloc[:,3*k+1],frame2.iloc[:,3*k+2],fmt='r-*',label=frame2.columns[3*k][0:3])
    plt.plot(frame1.iloc[:,0],frame1.iloc[:,1+2*k],'b-*')
    plt.xlim(0.0,32.0)
    plt.ylim(-2.0,3.2)
    plt.xlabel('time to expiry (y)')
    plt.ylabel('implied mean reversion (%)')
    plt.legend()
    #plt.title(swapTerm + ' swap term')
    plt.tight_layout()

plt.show()
print(frame1)

print(frame2)

