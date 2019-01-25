import sys, os
#sys.path.append(os.path.join(os.path.dirname(__file__),'QuantLibWrapper'))

if locals().get('__file__'):
    sys.path.append(os.path.join(os.path.dirname(__file__),'QuantLibWrapper'))
else:
    sys.path.append(os.path.join(os.getcwd(),'QuantLibWrapper'))

import numpy as np

from scipy.sparse import diags

import QuantLib as ql
from QuantLibWrapper import ThetaMethod, solveTDS

a = np.random.RandomState(123).standard_normal([5])
b = -2*np.random.RandomState(456).standard_normal([6])
c = np.random.RandomState(789).standard_normal([5])

y = np.random.RandomState(101112).standard_normal([6])

A = diags([a,b,c], [-1,0,1])

print(A.toarray())

b2 = np.linalg.inv(A.toarray()).dot(b)

solveTDS(A,b)

print(A.toarray())

print(b)
print(b2)

