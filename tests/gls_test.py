import pytest
import numpy as np
from p3py.gls import get_gls_unc

@pytest.fixture
def ds1():
    X = np.array([[1],[1]])
    Y = np.array([[1.0],[1.5]])
    V = np.array([[0.05,0.06],[0.06,0.1125]])
    V_ind = np.array([[0.05,0.0],[0.0,0.1125]])
    return X, Y, V, V_ind

def test_exceptions():

    # not a square matrix
    with pytest.raises(AssertionError):
        get_gls_unc([1,2])
    
    # 1d matrix
    with pytest.raises(AssertionError):
        get_gls_unc([1])

    # mismatch between matrices
    with pytest.raises(AssertionError):
        get_gls_unc([[1,0],[0,1]],[1,1,1])

def test_ind_values(ds1):
    X, _, _, V_ind = ds1
    gls_unc = get_gls_unc(V_ind,X)
    assert np.isclose(gls_unc, 0.1860521)
    
def test_corr_values(ds1):
    X, _, V, _ = ds1
    gls_unc = get_gls_unc(V,X)
    assert np.isclose(gls_unc, 0.21828206)

def test_default_X(ds1):
    # should get same answer without giving X
    _, _, V, _ = ds1
    gls_unc = get_gls_unc(V)
    assert np.isclose(gls_unc, 0.21828206)