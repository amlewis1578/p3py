import pytest
import numpy as np
from p3py.gls import get_gls_unc, get_gls_estimate

@pytest.fixture
def ds1():
    X = np.array([[1],[1]])
    Y = np.array([[1.0],[1.5]])
    V = np.array([[0.05,0.06],[0.06,0.1125]])
    V_ind = np.array([[0.05,0.0],[0.0,0.1125]])
    return X, Y, V, V_ind

def test_unc_exceptions():

    # not a square matrix
    with pytest.raises(AssertionError):
        get_gls_unc([1,2])
    
    # 1d matrix
    with pytest.raises(AssertionError):
        get_gls_unc([1])

    # mismatch between matrices
    with pytest.raises(AssertionError):
        get_gls_unc([[1,0],[0,1]],[1,1,1])

def test_est_exceptions(ds1):
    X, Y, V, V_ind = ds1

    # mismatch between Y and V
    with pytest.raises(AssertionError):
        get_gls_estimate(Y,V[:1,:1])
    
     # Y is not column vector
    with pytest.raises(AssertionError):
        get_gls_estimate([[1,0],[0,1]],V)   

def test_ind_values(ds1):
    X, Y, _, V_ind = ds1
    gls_unc = get_gls_unc(V_ind,X)
    assert np.isclose(gls_unc, 0.1860521)

    gls_est, gls_unc = get_gls_estimate(Y,V_ind,X)
    assert np.isclose(gls_est, 1.15384615 )
    assert np.isclose(gls_unc, 0.1860521)
    
def test_corr_values(ds1):
    X, Y, V, _ = ds1
    gls_unc = get_gls_unc(V,X)
    assert np.isclose(gls_unc, 0.21828206)

    gls_est, gls_unc = get_gls_estimate(Y,V,X)
    assert np.isclose(gls_est, 0.88235294 )
    assert np.isclose(gls_unc, 0.21828206)

def test_default_X(ds1):
    # should get same answer without giving X
    _, Y, V, _ = ds1
    gls_unc = get_gls_unc(V)
    assert np.isclose(gls_unc, 0.21828206)

    gls_est, gls_unc = get_gls_estimate(Y,V)
    assert np.isclose(gls_est, 0.88235294 )
    assert np.isclose(gls_unc, 0.21828206)