#!/usr/bin/python

import numpy as np
from scipy.linalg import lstsq

def MultiIndexSet(n, k):
    if n==1: return [ [i] for i in range(k) ]
    return [ [i]+s for i in range(k) for s in MultiIndexSet(n-1,k-i)]

class Regression:

    # Python constructor
    def __init__(self, controls, observations, maxPolynomialDegree=2):
        self.maxPolynomialDegree = maxPolynomialDegree
        self.multiIdxSet = np.array(MultiIndexSet(controls.shape[1],maxPolynomialDegree+1))
        A = np.array([ self.monomials(c) for c in controls ])
        p, res, rnk, s = lstsq(A, observations)   # res, rnk, s for debug purposes
        self.beta = p

    def monomials(self, control):
        x = np.ones(self.multiIdxSet.shape[0])
        for i in range(self.multiIdxSet.shape[0]):
            for j in range(self.multiIdxSet.shape[1]):
                x[i] *= control[j]**self.multiIdxSet[i][j]
        return x

    def value(self, control):
        return self.monomials(control).dot(self.beta)

