#!/usr/bin/python

import numpy as np
from scipy.sparse import diags

from ThetaMethod import thetaStep

class PDESolver:

    # Python constructor
    def __init__(self, hwModel, nGridPoints=101, stdDevs=5, theta=0.5, timeStepSize=1.0/12.0, lambda0N=None):
        self.hwModel      = hwModel
        self.nGridPoints  = nGridPoints
        self.stdDevs      = stdDevs
        self.theta        = theta
        self.timeStepSize = timeStepSize
        self.lambda0N     = lambda0N   # for boundary condition

    def xSet(self,expityTime):
        sigma = np.sqrt(self.hwModel.varianceX(0.0,expityTime))
        return np.linspace(-self.stdDevs*sigma,self.stdDevs*sigma,self.nGridPoints)

    def rollBack(self, T0, T1, x1, U1, H1):
        # first we calculate the payoff
        V = np.array([ max(U1[k],H1[k]) for k in range(U1.shape[0]) ])
        # now we need to determine the time grid
        M = int((T1-T0)/self.timeStepSize)
        tGrid = np.linspace(T1,T1-M*self.timeStepSize,M+1)
        if tGrid[-1]>T0: np.append(tGrid,[T0])
        for k in range(tGrid.shape[0]-1):
            # then we roll back individual time steps
            V = self.rollBackOneStep(tGrid[k+1],tGrid[k],x1,V)
        return [x1, V]

    def rollBackOneStep(self, T0, T1, x, V):
        # time and space discretisations
        ht = T1 - T0
        hx = (x[-1]-x[0])/(x.shape[0]-1)  # assume equidistant grid
        # theta estimation point
        t     = self.theta*T0 + (1-self.theta)*T1
        f     = self.hwModel.forwardRate(0.0,0.0,t)
        sigma = self.hwModel.sigma(t)
        y     = self.hwModel.y(t)
        a     = self.hwModel.meanReversion
        # linear operator v' = M v. Note, x is an array
        c = (sigma**2/hx**2 + f) + x
        l = -sigma**2/2.0/hx**2 + (1.0/2.0/hx)*(y-a*x)
        u = -sigma**2/2.0/hx**2 - (1.0/2.0/hx)*(y-a*x)
        # adjust for boundary conditions
        # lanbda approximation
        if self.lambda0N!=None:  # fall-back if provided by user, typically lambda0N=0
            lambda0 = self.lambda0N
            lambdaN = self.lambda0N
        else:
            Vx0  = (V[2]-V[0])/2.0/hx
            Vxx0 = (V[2]-2*V[1]+V[0])/hx/hx
            lambda0 = Vxx0/Vx0 if abs(Vx0)>1.0e-8 else 0.0
            VxN  = (V[-1]-V[-3])/2.0/hx
            VxxN = (V[-1]-2*V[-2]+V[-3])/hx/hx
            lambdaN = VxxN/VxN if abs(VxN)>1.0e-8 else  0.0
            #print('Vx0 = '+str('%10.6f'%Vx0)+', Vxx0 = '+str('%10.6f'%Vxx0)+', l0 = '+str('%10.6f'%lambda0)+ \
            #    ', VxN = '+str('%10.6f'%VxN)+', VxxN = '+str('%10.6f'%VxxN)+', lN = '+str('%10.6f'%lambdaN)  )
        c[0]  =  2.0*(y-a*x[0] +lambda0*sigma**2/2.0)/(2.0+lambda0*hx)/hx + x[0]  + f
        c[-1] = -2.0*(y-a*x[-1]+lambdaN*sigma**2/2.0)/(2.0+lambdaN*hx)/hx + x[-1] + f
        u[0]  = -2.0*(y-a*x[0] +lambda0*sigma**2/2.0)/(2.0+lambda0*hx)/hx
        l[-1] =  2.0*(y-a*x[-1]+lambdaN*sigma**2/2.0)/(2.0+lambdaN*hx)/hx
        # solve one step via theta method
        # M = diags([l[1:], c, u[:-1] ],[-1, 0, 1])
        V = thetaStep(l[1:], c, u[:-1],V,ht,self.theta)
        return V




