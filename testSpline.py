#!/usr/bin/python

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__),'QuantLibWrapper'))

import numpy as np
from scipy.interpolate import PPoly
from scipy.interpolate import CubicSpline

import matplotlib.pyplot as plt

from QuantLibWrapper import SplineInterpolation

x = np.linspace(0,10,6)
#y = np.sin(x)
#g = np.cos(x)
#g = np.zeros(11)
y = abs(x-5)
g = (y[2:] - y[:-2])/(x[2:]-x[:-2])
g = np.append([-1], g)
g = np.append(g, [1])


C = CubicSpline(x,y)
S = SplineInterpolation(x,y,g)

t = np.linspace(-1,11,121)
plt.figure()
plt.plot(t, abs(t-5), label='abs(t-5)')
plt.plot(t, C(t), label='CubicSpline')
plt.plot(t, S(t), label='GSpline')
plt.legend()

plt.figure()
#plt.plot(t, np.cos(t), label='cos()')
plt.plot(t, C.derivative()(t), label='CubicSplinePrime')
plt.plot(t, S.derivative()(t), label='GSplinePrime')
plt.legend()

plt.show()

print(C.c.shape)

