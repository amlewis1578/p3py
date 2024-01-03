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

    Raises
    ------
    numpy will crash if the array sizes are not compatible. #TODO: add
    specific tests for this with useful messages.
    
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


def print_parameters(P, M):
    """ Function to pretty print the parameters and their unc. The
    values are rounded to 4 decimals for ease of comparison with
    the SAMMY manual example. 
    
    Parameters
    ---------
    P : np.array
        Column vector of parameters

    M : np.array
        Parameter covariance matrix

    Returns
    -------
    None, just prints to screen
    """

    for z, dz_sq in zip(P, np.diag(M)):
        print(f"\n{np.around(z[0],4)} +/- {np.around(np.sqrt(dz_sq),4)}")