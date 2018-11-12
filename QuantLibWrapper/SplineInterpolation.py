#!/usr/bin/python

import numpy as np
from scipy.interpolate import PPoly


def SplineInterpolation(x, y, g):
    # interpolate y = y(x) with y'(x)=g(x)
    # assume x, y, g numpy.array
    # we calculate PPoly coefficients in two steps
    # first represent y(t)=a3 t**3 + a2 t**2 + a1 t + a0 for 0<=t<=1
    # and t = (x-x0) / (x1 - x0)
    a0 = y[:-1]
    a1 = g[:-1]
    a2 =  3.0*(y[1:] - y[:-1]) - (2.0*g[:-1] + g[1:])*(x[1:] - x[:-1])
    a3 = -2.0*(y[1:] - y[:-1]) +     (g[:-1] + g[1:])*(x[1:] - x[:-1])
    # now we set up the PPoly coefficients
    dx = x[1:] - x[:-1]
    c  = np.array([ a3/dx**3, a2/dx**2, a1, a0 ])
    return PPoly(c,x,True)

