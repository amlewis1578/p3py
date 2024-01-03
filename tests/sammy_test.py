import pytest
import numpy as np
from p3py.sammy._utils import calc_m_plus_w
from p3py.sammy import method1

@pytest.fixture()
def small_unc_example():
    r = np.array([10000,12100])
    dr = np.array([100,110])
    n = 100
    dn = 0.5
    return r, dr, n, dn

@pytest.fixture()
def large_unc_example():
    r = np.array([10000,12100])
    dr = np.array([100,110])
    n = 100
    dn = 10
    return r, dr, n, dn

def test_m_plus_w_scheme():

    # set up values from Method 1 example
    P = np.array([[109.50226244], [100.]])
    M_inv = np.array([[0,0],[0., 4.]])
    D = np.array([[10000], [12100]])
    V = np.array([[10000, 0], [0, 12100]])
    T = np.array([[10950.22624434], [10950.22624434]])
    G = np.array([[100, 109.50226244],  [100, 109.50226244]])

    p_prime, m_prime = calc_m_plus_w(P, M_inv, D, V, T, G)

    assert np.isclose(p_prime[0,0], 109.50226244)
    assert np.isclose(m_prime[0,0], 0.84727995)
    assert np.isclose(m_prime[0,1], -0.27375566)
    assert np.isclose(m_prime[1,1], 0.25)

def test_method1_small(small_unc_example):
    r, dr, n, dn = small_unc_example
    p_prime, m_prime = method1(r, dr, n, dn)
    assert np.isclose(p_prime[0,0], 109.50226244)
    assert np.isclose(m_prime[0,0], 0.84727995)


def test_method1_large(large_unc_example):
    r, dr, n, dn = large_unc_example
    p_prime, m_prime = method1(r, dr, n, dn)
    assert np.isclose(p_prime[0,0], 109.50226244)
    assert np.isclose(m_prime[0,0], 120.45496611)