""" Module with basic GLS functions """
import numpy as np
import numpy.linalg as la

def get_gls_unc(V, X=None):
    """ Function to calculate the GLS uncertainty, which
    is not a function of the data points
    
    Parameters
    ----------
    V : np.array
        The covariance matrix of the data points. Must be a 2D
        square matrix, the size of the number of data points.

    X : np.array, optional, default: None
        The design matrix/derivatives matrix. It should have
        the same number of rows (and cols) as the covariance 
        matrix, n, and the same number of columns as the number 
        of parameters in the fit. If not given, it is assumed 
        to be a (nx1) column vector of 1's. 

    Returns
    -------
    np.array 
        The uncertainty estimate. Array has the same length
        as the number of parameters, which is the number of 
        columns in X

    Raises
    ------
    AssertionError
        If the covariance matrix is not square and 2D, or if 
        the number of rows in X is not equal to the size of V.

    Notes
    -----
    The design matrix, X, is also often designated as G.
    
    """

    # if X is not given, default is column of 1's
    if X is None:
        X = np.ones((len(V),1))

    # make sure both are numpy arrays, not just lists
    V = np.array(V)
    X = np.array(X)

    # check that the matrices are proper size
    try:
        if not V.shape[0] == V.shape[1]:
            raise AssertionError("Error: Covariance matrix must be square")
    except IndexError: # if a 1-D matrix is given
            raise AssertionError("Error: Covariance matrix must be 2-D") 
    
    if not X.shape[0] == V.shape[0]:
        raise AssertionError("Error: Design matrix must have same number of rows as covariance matrix columns.")


    return np.sqrt((la.inv(X.T @ la.inv(V) @ X)))


