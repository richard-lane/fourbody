"""
Unit test for util (i.e. for the parameterisation maths)

"""
import numpy as np

from fourbody import util


def test_inv_mass_stationary():
    """
    Test invariant mass of a stationary particle gets calculated correctly

    """
    b0_mass = 5279.65
    momentum = np.array([[0.0]])
    energy = np.array([[b0_mass]])

    assert np.allclose(
        util._invariant_masses(momentum, momentum, momentum, energy),
        np.array([[b0_mass]]),
    )


def test_inv_mass():
    """
    Test invariant mass of a stationary particle gets calculated correctly

    """
    pi_mass = 139.57018

    px = np.array([[-2405.25192233]])
    py = np.array([[1017.71934261]])
    pz = np.array([[-128.6045092]])
    e = np.array([[2618.58901417]])

    assert np.allclose(util._invariant_masses(px, py, pz, e), np.array([[pi_mass]]))


def test_m_plus_minus():
    """
    Test we get the right invariant masses out

    """
    k = np.array([[53.89743437], [-385.779829], [-1782.366763], [2116.4343876]])
    pi1 = np.array([[-241.70298533], [-13.62889997], [1166.66733619], [1199.66603207]])
    pi2 = np.array([[-1176.18918034], [612.05350502], [482.97638869], [1418.01851474]])
    pi3 = np.array([[463.9947313], [-212.64477997], [132.72300175], [545.53106553]])

    mplus, mminus = util.m_plus_minus(k, pi1, pi2, pi3)

    assert np.allclose(mplus, util._invariant_masses(*np.add(k, pi3)))
    assert np.allclose(mminus, util._invariant_masses(*np.add(pi1, pi2)))


def test_phi_range():
    """
    Check we get the right phi back when we pass -ve or +ve phis

    """
    phis = np.linspace(-np.pi, np.pi)
    cosphis = np.cos(phis)
    sinphis = np.sin(phis)

    calculated_phis = util.phi(cosphis, sinphis)

    assert np.allclose(calculated_phis, phis)
