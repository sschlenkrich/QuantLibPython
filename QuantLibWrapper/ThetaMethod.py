#!/usr/bin/python

from scipy.sparse import diags, identity
import numpy as np


def solveTDS(diagA, y):
    # solve linear systen Ax = y for a tridiagonal matrix A
    a = diagA.diagonal(-1)
    b = diagA.diagonal(0)
    c = diagA.diagonal(1)
    # in place LU decomposition; no error handling if LU decomposition does not exist
    for i in range(1,b.shape[0]):
        a[i-1] /= b[i-1]
        b[i] -= c[i-1]*a[i-1]
    # forward substitution
    z = np.zeros(b.shape[0])
    z[0] = y[0]
    for i in range(1,z.shape[0]):
        z[i] = y[i] - a[i-1]*z[i-1]
    # backward substitution, overwrite input y
    y[-1] = z[-1]/b[-1]
    for i in range(z.shape[0]-1,0,-1):
        y[i-1] = (z[i-1] - c[i-1]*y[i])/b[i-1]
    return


def thetaStep(arrayL, arrayC, arrayU, arrayRHS, stepSize, theta):
    # solve v = [I+h*theta*M]^-1 [I-h(1-theta)M] r
    # where M = diag[l, c, u] and r = RHS
    M = diags([arrayL, arrayC, arrayU], [-1, 0, 1])
    # print(M.toarray())
    I = identity(arrayC.shape[0])
    b = (I - (stepSize*(1.0-theta))*M).dot(arrayRHS)
    if theta==0:  # Explicit Euler
        return b
    A = (I + (stepSize*theta)*M)
    solveTDS(A,b)
    return b
