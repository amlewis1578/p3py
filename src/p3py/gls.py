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



def get_gls_estimate(Y, V, X=None, verbose=False):
    """ Function to calculate the GLS estimate and the unc
    
    Parameters
    ----------
    Y : np.array
        The data points for fitting. Must be a column vector.

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
        The fitted parameter(s). Array has the same length
        as the number of parameters, which is the number of 
        columns in X

    np.array 
        The uncertainty estimate. Array has the same length
        as the number of parameters, which is the number of 
        columns in X

    Raises
    ------
    AssertionError
        If the covariance matrix is not square and 2D, if 
        the number of rows in X is not equal to the size of V, 
        if Y is not a column vector, or if the shapes of Y and V
        are not compatible.

    Notes
    -----
    The design matrix, X, is also often designated as G.
    
    """

    # if X is not given, default is column of 1's
    if X is None:
        X = np.ones((len(V),1))

    # first get the uncertainty, which also checks the matrices
    unc = get_gls_unc(V, X)

    # make sure all are numpy arrays
    Y = np.array(Y)
    V = np.array(V)
    X = np.array(X)

    # check shape of Y
    if not Y.shape[1] == 1:
        raise AssertionError("Error: Data points must be given in a column vector.")
    if not Y.shape[0] == V.shape[0]:
        raise AssertionError("Error: Number of rows in Y must equal size of V.")

    
    # print the steps if verbose mode
    if verbose:
        print("\nX:\n", X)
        print("\nY:\n", Y)
        print("\nV:\n", V)
        print("\nX.T @ la.inv(V) :\n", X.T @ la.inv(V))
        print("\nconst:\n", la.inv(X.T @ la.inv(V) @ X))

    # calculate the estimate
    est = la.inv(X.T @ la.inv(V) @ X) @ X.T @ la.inv(V) @ Y
    
    return est, unc


def check_2x2_matrix_for_PPP(V, verbose=True):
    """ Function to test whether a 2x2 covariance matrix will produce a 
    PPP result in GLS fitting.

    Parameters
    ----------
    V : np.array 
        2x2 covariance matrix

    verbose : bool, optional, default: True
        whether or not to print the values and the comparison
    
    Returns
    -------
    bool
        whether or not PPP will occur with GLS fitting

    Raises
    ------
    AssertionError
        If the matrix is not 2x2
    
    Notes
    -----
    The criteria comes from Ref [1] Equation 10 - if the correlation 
    coefficient is greater than the ratio of the smaller to the larger 
    uncertainty, PPP will occur with GLS fitting.

    [1] T. Burr, T. Kawano, P. Talou, F. Pan, and N. Hengartner, “Defense 
        of the Least Squares Solution to Peelle's Pertinent Puzzle,” 
        Algorithms, 4, 28-39 (2011) 10.3390/a4010028.


    """

    # make sure V is a numpy array and is 2x2
    V = np.array(V)
    try:
        assert np.array_equal(V.shape, (2,2))
    except AssertionError:
        print("Error: Covariance matrix must be 2x2 for PPP testing.")
        return None

    # get sigma1 and sigma2 from the diagonal
    sigma1, sigma2 = np.sqrt(np.diag(V))

    # get the ratio of the smaller to the larger
    ratio = min(sigma1/sigma2, sigma2/sigma1)

    # get the corr coefficient from one of the off-diagonals
    rho = V[0,1] / sigma1 / sigma2
    
    # if verbose mode, print the two values and the result
    if verbose: 
        print(f"rho: {rho}, ratio: {ratio}")
        if rho > ratio: print("PPP")
        else: print("no PPP")

    # return True if PPP occurs
    return rho > ratio
