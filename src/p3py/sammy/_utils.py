"""Module with basic utility functions for SAMMY methods"""

import numpy as np
import numpy.linalg as la

def calc_m_plus_w(P,M_inv,D,V,T,G, verbose=False):
    """ Function to calculate the M+W scheme of solving the
    Bayesian equations
    
    Parameters
    ----------
    P : np.array
        Prior parameters, as a column matrix

    M_inv : np.array
        Inverse of prior parameter cov matrix

    D : np.array
        Column vector of data points

    T : np.array
        Column vector of theoretical values

    G : np.array
        Sensitivity matrix

    verbose : bool, optional, default: False
        whether or not to print each matrix in the
        calculation

    Returns
    -------
    np.array
        Posterior parameters

    np.array
        Posterior parameter cov matrix
    
    Notes
    -----
    Following the equations from the SAMMY manual Section IV.A.1.
    Derivation of SAMMY's Solution Schemes, M+W Version

    """

    # first calc Y and W, convenience matrices
    Y = G.T @ la.inv(V) @ (D - T)
    if verbose: print("\nY: ", Y)

    W = G.T @ la.inv(V) @ G
    if verbose: print("\nW: ", W)

    # then get posterior covariance matrix
    M_prime = la.inv(M_inv + W)
    if verbose: print("\nM_prime: ", M_prime)

    # posterior parameters
    P_prime = P + M_prime @ Y
    if verbose: print("\nP_prime: ", P_prime)

    # return posterior values
    return P_prime, M_prime