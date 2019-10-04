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

today = ql.DateParser.parseISO('2019-06-18')
ql.Settings.instance().evaluationDate = today

CurveData = [
    # term   date           ois zero      swap zero     swap zero (single curve)
    [ '0d',  '2019-06-18', -0.003731245, -0.003438507, -0.003436844 ],
    [ '1Y',  '2020-06-18', -0.004903774, -0.003435947, -0.003434280 ],
    [ '2Y',  '2021-06-18', -0.005269786, -0.003760585, -0.003759319 ],
    [ '3Y',  '2022-06-18', -0.005131634, -0.003577121, -0.003575369 ],
    [ '4Y',  '2023-06-18', -0.004679281, -0.003071878, -0.003069037 ],
    [ '5Y',  '2024-06-18', -0.004003627, -0.002365117, -0.002360416 ],
    [ '6Y',  '2025-06-18', -0.003235259, -0.001588845, -0.001581742 ],
    [ '7Y',  '2026-06-18', -0.002388742, -0.000745669, -0.000735533 ],
    [ '8Y',  '2027-06-18', -0.001502092,  0.000147360,  0.000161220 ],
    [ '9Y',  '2028-06-18', -0.000601395,  0.001064374,  0.001082676 ],
    [ '10Y', '2029-06-18',  0.000260260,  0.001902820,  0.001925390 ],
    [ '11Y', '2030-06-18',  0.001035619,  0.002694505,  0.002721495 ],
    [ '12Y', '2031-06-18',  0.001810503,  0.003419099,  0.003450190 ],
    [ '13Y', '2032-06-18',  0.002429721,  0.004083135,  0.004118351 ],
    [ '14Y', '2033-06-18',  0.003046383,  0.004672182,  0.004711199 ],
    [ '15Y', '2034-06-18',  0.003663044,  0.005190053,  0.005232011 ],
    [ '16Y', '2035-06-18',  0.004029206,  0.005642099,  0.005686655 ],
    [ '17Y', '2036-06-18',  0.004394988,  0.006033651,  0.006080562 ],
    [ '18Y', '2037-06-18',  0.004759770,  0.006366808,  0.006415720 ],
    [ '19Y', '2038-06-18',  0.005124552,  0.006646954,  0.006697421 ],
    [ '20Y', '2039-06-18',  0.005489334,  0.006878319,  0.006929795 ],
    [ '21Y', '2040-06-18',  0.005629305,  0.007065762,  0.007117639 ],
    [ '22Y', '2041-06-18',  0.005767656,  0.007213382,  0.007265120 ],
    [ '23Y', '2042-06-18',  0.005906007,  0.007327039,  0.007378208 ],
    [ '24Y', '2043-06-18',  0.006044359,  0.007412021,  0.007462300 ],
    [ '25Y', '2044-06-18',  0.006183089,  0.007473760,  0.007522931 ],
    [ '26Y', '2045-06-18',  0.006214931,  0.007516816,  0.007564759 ],
    [ '27Y', '2046-06-18',  0.006246186,  0.007544624,  0.007591257 ],
    [ '28Y', '2047-06-18',  0.006277441,  0.007560006,  0.007605267 ],
    [ '29Y', '2048-06-18',  0.006308781,  0.007565787,  0.007609634 ],
    [ '30Y', '2049-06-18',  0.006340036,  0.007564762,  0.007607183 ],
    [ '31Y', '2050-06-18',  0.006335455,  0.007559394,  0.007600394 ],
    [ '32Y', '2051-06-18',  0.006330578,  0.007550539,  0.007590132 ],
    [ '33Y', '2052-06-18',  0.006325687,  0.007538613,  0.007576817 ],
    [ '34Y', '2053-06-18',  0.006320809,  0.007524135,  0.007560979 ],
    [ '35Y', '2054-06-18',  0.006315931,  0.007507529,  0.007543045 ],
    [ '36Y', '2055-06-18',  0.006311053,  0.007489175,  0.007523400 ],
    [ '37Y', '2056-06-18',  0.006306162,  0.007469097,  0.007502060 ],
    [ '38Y', '2057-06-18',  0.006301284,  0.007447411,  0.007479140 ],
    [ '39Y', '2058-06-18',  0.006296407,  0.007424068,  0.007454583 ],
    [ '40Y', '2059-06-18',  0.006291529,  0.007399075,  0.007428387 ],
    [ '41Y', '2060-06-18',  0.006274616,  0.007372425,  0.007400538 ],
    [ '42Y', '2061-06-18',  0.006257685,  0.007344592,  0.007371521 ],
    [ '43Y', '2062-06-18',  0.006240753,  0.007315899,  0.007341661 ],
    [ '44Y', '2063-06-18',  0.006223821,  0.007286739,  0.007311359 ],
    [ '45Y', '2064-06-18',  0.006206842,  0.007257425,  0.007280931 ],
    [ '46Y', '2065-06-18',  0.006189910,  0.007228513,  0.007250944 ],
    [ '47Y', '2066-06-18',  0.006172979,  0.007200315,  0.007221714 ],
    [ '48Y', '2067-06-18',  0.006156047,  0.007173225,  0.007193641 ],
    [ '49Y', '2068-06-18',  0.006139068,  0.007147568,  0.007167054 ],
    [ '50Y', '2069-06-18',  0.006122136,  0.007123879,  0.007142499 ],
    [ '51Y', '2070-06-18',  0.006114477,  0.007102377,  0.007120197 ],
    [ '52Y', '2071-06-18',  0.006106868,  0.007082938,  0.007100020 ],
    [ '53Y', '2072-06-18',  0.006099239,  0.007065288,  0.007081683 ],
    [ '54Y', '2073-06-18',  0.006091631,  0.007049293,  0.007065052 ],
    [ '55Y', '2074-06-18',  0.006084022,  0.007034673,  0.007049836 ],
    [ '56Y', '2075-06-18',  0.006076414,  0.007021200,  0.007035802 ],
    [ '57Y', '2076-06-18',  0.006068784,  0.007008612,  0.007022678 ],
    [ '58Y', '2077-06-18',  0.006061176,  0.006996748,  0.007010300 ],
    [ '59Y', '2078-06-18',  0.006053567,  0.006985342,  0.006998394 ],
    [ '60Y', '2079-06-18',  0.006045959,  0.006974167,  0.006986726 ],
    [ '70Y', '2089-06-18',  0.005980679,  0.006878619,  0.006886955 ] ]


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
    [   5.5,  12.5,  20.5,  29.5,   40.0,   50.0,   61.5,   73.5,   85.5,   97.5,  156.0,  218.0,  276.0,  334.0 ],
    [   7.5,  17.5,  28.5,  42.0,   57.0,   72.0,   87.5,  104.0,  121.0,  137.0,  219.0,  305.0,  382.0,  455.0 ],
    [   9.0,  22.5,  37.0,  52.5,   70.0,   87.5,  106.0,  125.0,  145.0,  165.0,  261.0,  357.0,  445.0,  534.0 ],
    [  13.0,  31.0,  52.5,  74.5,   97.5,  123.0,  149.0,  175.0,  203.0,  230.0,  356.0,  478.0,  592.0,  701.0 ],
    [  16.5,  38.5,  65.0,  92.5,  121.0,  151.0,  183.0,  213.0,  245.0,  279.0,  426.0,  572.0,  706.0,  832.0 ],
    [  19.5,  45.0,  77.5, 109.0,  143.0,  178.0,  215.0,  253.0,  289.0,  327.0,  491.0,  651.0,  800.0,  940.0 ],
    [  27.5,  60.5, 101.0, 141.0,  183.0,  228.0,  275.0,  319.0,  365.0,  412.0,  613.0,  804.0,  984.0, 1155.0 ],
    [  35.5,  76.0, 125.0, 173.0,  222.0,  275.0,  329.0,  383.0,  437.0,  488.0,  719.0,  940.0, 1149.0, 1342.0 ],
    [  54.0, 112.0, 175.0, 238.0,  302.0,  367.0,  432.0,  497.0,  560.0,  623.0,  898.0, 1162.0, 1412.0, 1650.0 ],
    [  71.0, 145.0, 221.0, 297.0,  373.0,  450.0,  527.0,  603.0,  675.0,  743.0, 1050.0, 1349.0, 1624.0, 1892.0 ],
    [  86.5, 174.0, 262.0, 349.0,  437.0,  522.0,  607.0,  692.0,  773.0,  850.0, 1188.0, 1510.0, 1809.0, 2109.0 ],
    [ 112.5, 223.0, 331.0, 438.0,  541.0,  642.0,  740.0,  837.0,  933.0, 1025.0, 1411.0, 1788.0, 2140.0, 2495.0 ],
    [ 139.0, 274.0, 406.0, 533.0,  658.0,  780.0,  901.0, 1017.0, 1129.0, 1246.0, 1716.0, 2160.0, 2589.0, 3023.0 ],
    [ 165.5, 328.0, 486.0, 638.0,  786.0,  935.0, 1076.0, 1217.0, 1363.0, 1503.0, 2092.0, 2662.0, 3201.0, 3729.0 ],
    [ 185.0, 365.0, 544.0, 714.0,  885.0, 1058.0, 1219.0, 1383.0, 1544.0, 1701.0, 2397.0, 3025.0, 3631.0, 4217.0 ],
    [ 202.5, 401.0, 599.0, 787.0,  972.0, 1158.0, 1334.0, 1509.0, 1686.0, 1861.0, 2633.0, 3301.0, 3954.0, 4589.0 ],
    [ 217.0, 430.0, 641.0, 845.0, 1046.0, 1242.0, 1429.0, 1606.0, 1787.0, 1974.0, 2795.0, 3507.0, 4197.0, 4877.0 ] ]
    

cashSettledPremium = [
    [   5.5,  12.5,  20.5,  29.5,   39.5,   49.5,   61.0,   72.5,   84.0,   95.5,  150.0,  208.0,  262.0,  318.0 ], 
    [   7.5,  17.5,  28.5,  42.0,   56.5,   71.5,   86.5,  102.0,  119.0,  134.0,  211.0,  290.0,  363.0,  434.0 ], 
    [   9.0,  22.5,  37.0,  52.0,   69.5,   86.5,  105.0,  123.0,  142.0,  161.0,  251.0,  340.0,  422.0,  508.0 ], 
    [  13.0,  31.0,  52.5,  74.0,   96.5,  122.0,  147.0,  172.0,  199.0,  225.0,  342.0,  456.0,  564.0,  668.0 ], 
    [  16.5,  38.5,  64.5,  92.0,  120.0,  149.0,  180.0,  209.0,  240.0,  272.0,  410.0,  547.0,  674.0,  795.0 ], 
    [  19.5,  45.0,  77.0, 108.0,  142.0,  176.0,  212.0,  249.0,  283.0,  319.0,  473.0,  623.0,  765.0,  901.0 ], 
    [  27.5,  60.5, 101.0, 140.0,  181.0,  225.0,  271.0,  313.0,  357.0,  403.0,  592.0,  773.0,  945.0, 1112.0 ], 
    [  35.5,  76.0, 124.0, 172.0,  220.0,  272.0,  324.0,  376.0,  428.0,  477.0,  696.0,  906.0, 1107.0, 1296.0 ], 
    [  54.0, 111.0, 174.0, 236.0,  299.0,  363.0,  426.0,  489.0,  550.0,  611.0,  873.0, 1127.0, 1370.0, 1602.0 ], 
    [  71.0, 144.0, 220.0, 295.0,  370.0,  445.0,  520.0,  594.0,  664.0,  730.0, 1026.0, 1315.0, 1584.0, 1846.0 ], 
    [  86.5, 173.0, 261.0, 347.0,  433.0,  517.0,  600.0,  683.0,  762.0,  837.0, 1167.0, 1479.0, 1773.0, 2066.0 ], 
    [ 112.5, 222.0, 330.0, 435.0,  537.0,  637.0,  733.0,  829.0,  923.0, 1014.0, 1392.0, 1764.0, 2110.0, 2456.0 ], 
    [ 139.0, 274.0, 405.0, 531.0,  655.0,  776.0,  895.0, 1009.0, 1120.0, 1236.0, 1700.0, 2136.0, 2555.0, 2975.0 ], 
    [ 165.5, 327.0, 485.0, 637.0,  784.0,  932.0, 1073.0, 1212.0, 1357.0, 1496.0, 2076.0, 2627.0, 3141.0, 3635.0 ], 
    [ 185.0, 364.0, 543.0, 713.0,  883.0, 1055.0, 1215.0, 1376.0, 1536.0, 1691.0, 2368.0, 2964.0, 3528.0, 4063.0 ], 
    [ 202.5, 401.0, 597.0, 786.0,  970.0, 1154.0, 1328.0, 1500.0, 1674.0, 1846.0, 2592.0, 3220.0, 3820.0, 4384.0 ], 
    [ 217.0, 430.0, 639.0, 843.0, 1043.0, 1237.0, 1422.0, 1596.0, 1772.0, 1956.0, 2745.0, 3412.0, 4035.0, 4624.0 ] ]



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
        lowHighs = [ [-0.20,-0.10], [-0.15,-0.10], [-0.10,-0.05], [-0.05,-0.01], [-0.01,0.01], [0.01,0.05], [0.05,0.1], [0.10,0.15] ]
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
#swapPairs = [ [5,15], [10,20], [15,25], [20,30] ]
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
##exit()
#
#vols = [ ]
#for j in range(len(swapTerms)):
#    volsPerTerm = []
#    for i in range(len(expiryTerms)):
#        helper = PremiumHelper(expiryTerms[i], swapTerms[j],
#                     physicalySettledPremium[i][j], cashSettledPremium[i][j])
#        volsPerTerm.append(helper.normalVolatility)
#    vols.append(volsPerTerm)
#expiryTimes = [ (float(term[:-1]) if term[-1]=='Y' else float(term[:-1])/12.0) for term in expiryTerms ]
#swapTimes   = [ int(term[:-1]) for term in swapTerms   ]
#table = [ expiryTimes ] + vols
#labels = [ 'ExpiryTimes' ] + swapTerms
#frame = pandas.DataFrame(table).T
#frame.columns = labels
#frame.to_csv('ImpledATMVolatilities.csv', index=False)
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#X, Y = np.meshgrid(expiryTimes,swapTimes,indexing='ij')
#surf = ax.plot_surface(X, Y, 1e4*np.transpose(np.array(vols)), cmap=cm.coolwarm, linewidth=0, antialiased=False)
#ax.set_xlim(0, 30)
#ax.set_ylim(0, 30)
##ax.set_zlim(50, 150)
#ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
#ax.set_yticks([0, 5, 10, 15, 20, 25, 30])
#ax.set_xlabel('Expiries (y)')
#ax.set_ylabel('Swap terms (y)')
#ax.set_zlabel('Market-implied normal volatility (bp)')
#plt.show()
##exit()
#
# we analyse switch prices for a few examples
#swapPairs = [ ['20Y', '10Y', 'r'], ['25Y', '10Y', 'b'], ['30Y', '10Y', 'g'] ]
#plt.figure(figsize=(6, 4))
#for item in swapPairs:
#    helper = PremiumHelper(item[0], item[1],
#                physicalySettledPremium[expiryTerms.index(item[0])][swapTerms.index(item[1])],
#                cashSettledPremium[expiryTerms.index(item[0])][swapTerms.index(item[1])])
#    meanReversions = np.linspace(-0.01, 0.10, 20)
#    cashPhysicalSw = np.array([helper.cashPhysicalSwitchModel(helper.HullWhiteModel(mr)) for mr in meanReversions ])
#    marketpriceSw  = np.array([helper.physicalStraddle - helper.cashStraddle for mr in meanReversions ])
#    plt.plot(meanReversions*100,cashPhysicalSw, item[2]+'-',label=item[0] + '-' + item[1])
#    plt.plot(meanReversions*100,marketpriceSw,  item[2]+'--')
#    plt.xlim(-6.0,11.0)
#    #plt.ylim(-10.0,5.0)
#    plt.xlabel('mean reversion (%)')
#    plt.ylabel('cash-physical-switch price')
#    plt.legend()
#    #plt.title(item[0] + '-' + item[1] + ' swaption')
#    plt.tight_layout()
#plt.show()
#exit()
#
## how does the physical price impact the switch value???
#swapPairs = [ ['10Y', '10Y', 'r'], ['20Y', '20Y', 'b'] ]
#discount  = 0.8
#plt.figure(figsize=(6, 4))
#for item in swapPairs:
#    meanReversions = np.linspace(-0.05, 0.10, 20)
#    physPrem = physicalySettledPremium[expiryTerms.index(item[0])][swapTerms.index(item[1])]
#    cashPrem = cashSettledPremium[expiryTerms.index(item[0])][swapTerms.index(item[1])]
#    switch   = physPrem - cashPrem
#    helper = PremiumHelper(item[0], item[1], physPrem, cashPrem)
#    cashPhysicalSw = np.array([helper.cashPhysicalSwitchModel(helper.HullWhiteModel(mr)) for mr in meanReversions ])
#    plt.plot(meanReversions*100,cashPhysicalSw, item[2]+'-',label=item[0] + '-' + item[1])
#    # now with discount
#    helper = PremiumHelper(item[0], item[1], discount*physPrem, discount*physPrem-switch)
#    cashPhysicalSw = np.array([helper.cashPhysicalSwitchModel(helper.HullWhiteModel(mr)) for mr in meanReversions ])
#    plt.plot(meanReversions*100,cashPhysicalSw, item[2]+'--')
#    plt.xlim(-6.0,11.0)
#    #plt.ylim(-10.0,5.0)
#    plt.xlabel('mean reversion (%)')
#    plt.ylabel('cash-physical-switch price')
#    plt.legend()
#    #plt.title(item[0] + '-' + item[1] + ' swaption')
#    plt.tight_layout()
#plt.show()
##exit()
#
## also we want to get an idea of how the payoff looks like depending on mean reversion
#meanReversions = [ -0.05, 0.025, 0.10 ]
#colors         = [   'r',   'r',  'r' ]
#item = ['20Y', '20Y']
#timeToExpiry = 20.0   # set this consistent to item
#for meanReversion, color in zip(meanReversions,colors):
#    plt.figure(figsize=(8, 4))
#    ax1 = plt.gca()
#    ax2 = ax1.twinx()
#    helper = PremiumHelper(item[0], item[1],
#                physicalySettledPremium[expiryTerms.index(item[0])][swapTerms.index(item[1])],
#                cashSettledPremium[expiryTerms.index(item[0])][swapTerms.index(item[1])])
#    hwModel = helper.HullWhiteModel(meanReversion)
#    cashPhysicalSwitchPayoff = CashPhysicalSwitchPayoff(helper.payer,hwModel)
#    states = np.linspace(-0.10,0.10,201)
#    payoff = np.array([ cashPhysicalSwitchPayoff.at([x,0.0]) for x in states ])
#    ax1.plot(states*100,payoff,color+'-',label='MR: '+str(meanReversion*100)+'%')
#    # we also need to consider the change in distribution
#    mu    = hwModel.expectationX(0.0,0.0,timeToExpiry)
#    sigma = np.sqrt(hwModel.varianceX(0.0,timeToExpiry))
#    pdf = np.array([ norm.pdf((x-mu)/sigma)/sigma for x in states ])
#    ax2.plot(states*100,pdf,color+'--')
#
#    ax1.set_xlabel('state variable (%)')
#    ax1.set_ylabel('cash-physical-switch payoff')
#    ax1.set_ylim(-500,500)
#    #ax1.legend()
#    ax2.set_ylabel('pdf')
#    ax2.set_ylim(0,35)
#    plt.title('MR: '+str(meanReversion*100)+'%')
#    plt.tight_layout()
#plt.show()
#
#exit()
#
# we calculate implied mean reversion for a couple of swap terms an expiries
expiries = {}
expiries['10Y'] = [                                                           '30Y' ]
expiries['15Y'] = [                         '7Y', '10Y', '15Y', '20Y', '25Y', '30Y' ]
expiries['20Y'] = [             '4Y', '5Y', '7Y', '10Y', '15Y', '20Y', '25Y', '30Y' ]
expiries['25Y'] = [ '2Y', '3Y', '4Y', '5Y', '7Y', '10Y', '15Y', '20Y', '25Y', '30Y' ]
expiries['30Y'] = [ '2Y', '3Y', '4Y', '5Y', '7Y', '10Y', '15Y', '20Y', '25Y', '30Y' ]

expTimes = {}
expTimes['10Y'] = [                                                            30.0 ]
expTimes['15Y'] = [                          7.0,  10.0,  15.0,  20.0,  25.0,  30.0 ]
expTimes['20Y'] = [              4.0,  5.0,  7.0,  10.0,  15.0,  20.0,  25.0,  30.0 ]
expTimes['25Y'] = [  2.0,  3.0,  4.0,  5.0,  7.0,  10.0,  15.0,  20.0,  25.0,  30.0 ]
expTimes['30Y'] = [  2.0,  3.0,  4.0,  5.0,  7.0,  10.0,  15.0,  20.0,  25.0,  30.0 ]

swaps = [ '10Y', '15Y', '20Y', '25Y', '30Y']


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
    plt.plot(frame1.iloc[:,0],frame1.iloc[:,1+2*k],'b-*',label=frame1.columns[1+2*k])
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

