""" Module that contains the different methods described in the SAMMY manual """

import numpy as np
from p3py.sammy._utils import calc_m_plus_w, print_parameters
from p3py.gls import get_gls_estimate

def method1(r, dr, n, dn, verbose=False):
    """ Function to calculate the Method 1 solution
    
    Parameters
    ----------
    r : list or np.array
        measured data points

    dr : list or np.array
        uncertainty on measured data points

    n : float
        normalization constant value

    dn : float
        normalization constant uncertainty

    verbose: bool, optional, default: False
        whether or not to print the parameters and their
        uncertainties

    Returns
    -------
    np.array 
        Fitted parameter values

    np.array
        Fitting parameter covariance matrix
    
    """

    # first step - average the two R values with GLS

    # set up the D and V matrices
    D = np.array([[r[0]],[r[1]]])
    V = np.diag(np.array(dr)**2)
    D_av, _ = get_gls_estimate(D, V)

    # prior of X is the normalized average
    X = D_av[0,0] / n

    # the theoretical values are T = nX
    T = np.array([[n*X],[n*X]])

    # unc on the averaged D_av is assumed to be inifinite to 
    # allow full-least-squares calculation
    M_inv = np.zeros((2,2))
    M_inv[-1,-1] = 1/dn**2

    # sensitivy matrix
    G = np.hstack((np.array([[n],[n]]),np.array([[X],[X]])))

    # prior parameters
    P = np.array([[X],[n]])

    # solve with M+W schema
    P_prime, M_prime = calc_m_plus_w(P,M_inv,D,V,T,G)

    if verbose:
        print_parameters(P_prime, M_prime)

    return P_prime, M_prime

def method2(r, dr, n, dn, verbose=False):
    """ Function to calculate the Method 2 solution
    
    Parameters
    ----------
    r : list or np.array
        measured data points

    dr : list or np.array
        uncertainty on measured data points

    n : float
        normalization constant value

    dn : float
        normalization constant uncertainty

    verbose: bool, optional, default: False
        whether or not to print the parameters and their
        uncertainties

    Returns
    -------
    np.array 
        Fitted parameter values

    np.array
        Fitting parameter covariance matrix
    
    """

    # in this case, the D values are the individual normalized
    # data points

    # set up the D and V matrices
    D = np.array([[r[0]/n],[r[1]/n]])

    # V is given in the manual Eq.(IV D.5)
    dr_sq = np.array(dr)**2
    n_rel = dn**2/n**2
    V = np.array([
        [dr_sq[0]/n**2 + (n_rel)*D[0,0]**2, n_rel*D[0,0]*D[1,0]],
        [n_rel*D[0,0]*D[1,0], dr_sq[1]/n**2 + (n_rel)*D[1,0]**2]
    ])
    
    # the prior X is the GLS estimate
    X,_ = get_gls_estimate(D, V)

    # the theoretical values are T = X, since X now 
    # incorporates n
    T = np.array([[X],[X]])

    # unc on the averaged X is assumed to be inifinite to 
    # allow full-least-squares calculation
    M_inv = np.zeros((1,1))

    # sensitivy matrix
    G = np.array([[1],[1]])

    # prior parameter is X
    P = np.array([X])

    # solve with M+W schema
    P_prime, M_prime = calc_m_plus_w(P,M_inv,D,V,T,G)

    if verbose:
        print_parameters(P_prime, M_prime)

    return P_prime, M_prime
