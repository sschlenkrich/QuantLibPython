import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__),'QuantLibWrapper'))

import numpy as np

from QuantLibWrapper import MultiIndexSet, Regression

print(MultiIndexSet(3,3))

X = np.random.random(100)
Y = np.random.random(100)

C = np.array([ [x,y] for x in X for y in Y ])

Z = np.array([ 0.5*x**2 + x - 2*y**2 -y + 3*x*y - 3 + 1.0*(np.random.random()-0.5)
        for x in X for y in Y ] )

print(C)
print(Z)
print(C.shape)

R = Regression(C,Z,2)
# beta = [-3.  -1.  -2.   1.   3.   0.5]
print(R.beta)
print(R.multiIdxSet)

