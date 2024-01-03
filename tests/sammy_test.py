import pytest
import numpy as np
from p3py.sammy._utils import calc_m_plus_w

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