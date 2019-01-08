#!/usr/bin/python

import pandas
import QuantLib as ql

import numpy as np


relStrikes = [  -0.0200,  -0.0100,  -0.0050,  -0.0025,   0.0000,   0.0025,   0.0050,   0.0100,   0.0200 ]

smile01x01 = [           0.002357, 0.001985, 0.002038, 0.002616, 0.003324, 0.004017, 0.005518, 0.008431 ]
smile3mx02 = [                     0.001653, 0.001269, 0.002250, 0.003431, 0.004493, 0.006528, 0.010423 ]
smile02x02 = [           0.003641, 0.003766, 0.003987, 0.004330, 0.004747, 0.005177, 0.006096, 0.008203 ]
smile01x05 = [ 0.003925, 0.004376, 0.004284, 0.004364, 0.004680, 0.005118, 0.005598, 0.006645, 0.008764 ]
smile05x05 = [ 0.005899, 0.005975, 0.006202, 0.006338, 0.006431, 0.006639, 0.006793, 0.007135, 0.007907 ]
smile3mx10 = [ 0.006652, 0.005346, 0.004674, 0.004583, 0.004850, 0.005431, 0.006161, 0.007743, 0.010880 ]
smile01x10 = [ 0.005443, 0.005228, 0.005271, 0.005398, 0.005600, 0.005879, 0.006203, 0.006952, 0.008603 ]
smile02x10 = [ 0.005397, 0.005492, 0.005685, 0.005821, 0.005971, 0.006167, 0.006367, 0.006818, 0.007840 ]
smile05x10 = [ 0.006096, 0.006234, 0.006427, 0.006541, 0.006622, 0.006821, 0.006946, 0.007226, 0.007875 ]
smile10x10 = [ 0.006175, 0.006353, 0.006485, 0.006582, 0.006602, 0.006850, 0.006923, 0.007097, 0.007495 ]
smile05x30 = [ 0.005560, 0.005660, 0.005792, 0.005871, 0.005958, 0.006147, 0.006233, 0.006458, 0.007048 ]




class SwaptionVolatility:

    # Python constructor
    def __init__(self, fileName, projYtsH, discYtsH):
        today = ql.Settings.getEvaluationDate(ql.Settings.instance())
        S0 = 0.0
        extrapolationRelativeStrike = relStrikes[-1] + 0.05
        extrapolationSlope = 0.0

        atmVols = pandas.read_csv(fileName, sep=';', index_col=0 )
        swapTerms = [ ql.Period(p) for p in atmVols.columns.values ]
        expiTerms = [ ql.Period(p) for p in atmVols.index ] 
        valsMatrx = ql.Matrix(atmVols.values.shape[0],atmVols.values.shape[1])
        for i in range(atmVols.values.shape[0]):
            for j in range(atmVols.values.shape[1]):
                valsMatrx[i][j] = atmVols.values[i][j]
        self.atmVTS = ql.SwaptionVolatilityMatrix(ql.TARGET(),ql.Following,expiTerms,swapTerms,valsMatrx,ql.Actual365Fixed(),True,ql.Normal)

        cms01 = []
        cms02 = []
        cms05 = []
        cms10 = []
        cms30 = []

        cms01.append( ql.VanillaLocalVolModelSmileSection(ql.TARGET().advance(today,ql.Period('1y')), S0, relStrikes[1:], smile01x01, extrapolationRelativeStrike, extrapolationSlope) )
        cms02.append( ql.VanillaLocalVolModelSmileSection(ql.TARGET().advance(today,ql.Period('3m')), S0, relStrikes[2:], smile3mx02, extrapolationRelativeStrike, extrapolationSlope) )
        cms02.append( ql.VanillaLocalVolModelSmileSection(ql.TARGET().advance(today,ql.Period('2y')), S0, relStrikes[1:], smile02x02, extrapolationRelativeStrike, extrapolationSlope) )
        cms05.append( ql.VanillaLocalVolModelSmileSection(ql.TARGET().advance(today,ql.Period('1y')), S0, relStrikes[0:], smile01x05, extrapolationRelativeStrike, extrapolationSlope) )
        cms05.append( ql.VanillaLocalVolModelSmileSection(ql.TARGET().advance(today,ql.Period('5y')), S0, relStrikes[0:], smile05x05, extrapolationRelativeStrike, extrapolationSlope) )
        cms10.append( ql.VanillaLocalVolModelSmileSection(ql.TARGET().advance(today,ql.Period('3m')), S0, relStrikes[0:], smile3mx10, extrapolationRelativeStrike, extrapolationSlope) )
        cms10.append( ql.VanillaLocalVolModelSmileSection(ql.TARGET().advance(today,ql.Period('1y')), S0, relStrikes[0:], smile01x10, extrapolationRelativeStrike, extrapolationSlope) )
        cms10.append( ql.VanillaLocalVolModelSmileSection(ql.TARGET().advance(today,ql.Period('2y')), S0, relStrikes[0:], smile02x10, extrapolationRelativeStrike, extrapolationSlope) )
        cms10.append( ql.VanillaLocalVolModelSmileSection(ql.TARGET().advance(today,ql.Period('5y')), S0, relStrikes[0:], smile05x10, extrapolationRelativeStrike, extrapolationSlope) )
        cms10.append( ql.VanillaLocalVolModelSmileSection(ql.TARGET().advance(today,ql.Period('10y')), S0, relStrikes[0:], smile10x10, extrapolationRelativeStrike, extrapolationSlope) )
        cms30.append( ql.VanillaLocalVolModelSmileSection(ql.TARGET().advance(today,ql.Period('5y')), S0, relStrikes[0:], smile05x30, extrapolationRelativeStrike, extrapolationSlope) )

        smileCollection = [ cms01, cms02, cms05, cms10, cms30 ]
        periods = [ ql.Period('1y'), 
                    ql.Period('2y'), 
                    ql.Period('5y'),
                    ql.Period('10y'),
                    ql.Period('30y') ]

        index = ql.EuriborSwapIsdaFixA( ql.Period('1y'), projYtsH, discYtsH )
        h = ql.SwaptionVolatilityStructureHandle(self.atmVTS)
        self.volTS = ql.VanillaLocalVolSwaptionVTS(h,smileCollection,periods,index)

