from __future__ import division
import numpy as np
import scipy.special as special
import itertools

"""
Main module for computing the multiFoxH module. The script test_multiFoxH.py contains an
example of using this module. The method is based on simple rectangular approximation of the multivariate
Fox-H integrals. 

"""

def detBoundaries(params, tol):
    '''This modules attempts to determine an appropriate  rectangular
    boundaries of the integration region of the multivariate Fox H function.'''
    boundary_range = np.arange(0, 50, 0.05)
    dims = len(params[0])
    boundaries = np.zeros(dims)
    for dim_l in range(dims):
        points = np.zeros((boundary_range.shape[0], dims))
        points[:, dim_l] = boundary_range
        abs_integrand = np.abs(compMultiFoxHIntegrand(points, params))
        index = np.max(np.nonzero(abs_integrand > tol * abs_integrand[0]))
        boundaries[dim_l] = boundary_range[index]
    return boundaries


def compMultiFoxHIntegrand(y, params):
    ''' This module computes the complex integrand of the multivariate Fox-H
    function at the points given by the rows of the matrix y.'''
    z, mn, pq, c, d, a, b = params
    m, n = zip(*mn)
    p, q = zip(*pq)
    npoints, dims = y.shape
    s = 1j * y

    # Estimating sigma[l]
    lower = np.zeros(dims)
    upper = np.zeros(dims)
    for dim_l in range(dims):
        if b[dim_l]:
            bj, Bj = zip(*b[dim_l])
            bj = np.array(bj[:m[dim_l + 1]])
            Bj = np.array(Bj[:m[dim_l + 1]])
            lower[dim_l] = -np.min(bj / Bj)
        else:
            lower[dim_l] = -100
        if a[dim_l]:
            aj, Aj = zip(*a[dim_l])
            aj = np.array(aj[:n[dim_l + 1]])
            Aj = np.array(Aj[:n[dim_l + 1]])
            upper[dim_l] = np.min((1 - aj) / Aj)
        else:
            upper[dim_l] = 1
    mindist = np.linalg.norm(upper - lower)
    sigs = 0.5 * (upper + lower)
    for j in range(n[0]):
        num = 1 - c[j][0] - np.sum(c[j][1:] * lower)
        cnorm = np.linalg.norm(c[j][1:])
        newdist = np.abs(num) / (cnorm + np.finfo(float).eps)
        if newdist < mindist:
            mindist = newdist
            sigs = lower + 0.5 * num * np.array(c[j][1:]) / (cnorm * cnorm)
    s += sigs

    # Computing products of Gamma factors on both numeratos and denomerator
    s1 = np.c_[np.ones((npoints, 1)), s]
    prod_gam_num = prod_gam_denom = 1 + 0j
    for j in range(n[0]):
        prod_gam_num *= special.gamma(1 - np.dot(s1, c[j]))
    for j in range(q[0]):
        prod_gam_denom *= special.gamma(1 - np.dot(s1, d[j]))
    for j in range(n[0], p[0]):
        prod_gam_denom *= special.gamma(np.dot(s1, c[j]))
    for dim_l in range(dims):
        for j in range(n[dim_l + 1]):
            prod_gam_num *= special.gamma(1 - a[dim_l][j][0] - a[dim_l][j][1] * s[:, dim_l])
        for j in range(m[dim_l + 1]):
            prod_gam_num *= special.gamma(b[dim_l][j][0] + b[dim_l][j][1] * s[:, dim_l])
        for j in range(n[dim_l + 1], p[dim_l + 1]):
            prod_gam_denom *= special.gamma(a[dim_l][j][0] + a[dim_l][j][1] * s[:, dim_l])
        for j in range(m[dim_l + 1], q[dim_l + 1]):
            prod_gam_denom *= special.gamma(1 - b[dim_l][j][0] - b[dim_l][j][1] * s[:, dim_l])

    # Final integrand
    zs = np.power(z, -s)
    result = (prod_gam_num / prod_gam_denom) * np.prod(zs, axis=1) / (2 * np.pi) ** dims
    # the complex j is not forgotten
    return result


def compMultiFoxH(params, nsubdivisions, boundaryTol=0.0001):
    '''This module estimates a multivariate integral using simple rectangule
    quadrature. In most practical applications, 20 points per dimension provide
    sufficient accuracy.
    Inputs:
    'params': list containing z, mn, pq, c, d, a, b.
    'nsubdivisions': the number of divisions taken along each dimension. Note
    that the total number of points will be nsubdivisions**dim.
    'boundaryTol': tolerance used for determining the boundaries
    Output:
    'result': the estimated value of the multivariate Fox H function...'''
    boundaries = detBoundaries(params, boundaryTol)
    dim = boundaries.shape[0]
    signs = list(itertools.product([1, -1], repeat=dim))
    code = list(itertools.product(range(int(nsubdivisions / 2)), repeat=dim))
    quad = 0
    res = np.zeros((0))
    for sign in signs:
        points = np.array(sign) * (np.array(code) + 0.5) * boundaries * 2 / nsubdivisions
        res = np.r_[res, np.real(compMultiFoxHIntegrand(points, params))]
        quad += np.sum(compMultiFoxHIntegrand(points, params))
    volume = np.prod(2 * boundaries / nsubdivisions)
    result = quad * volume
    return result