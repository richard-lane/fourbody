"""
Fcn for doing the parameterisation

"""
import numpy as np

from . import util


def _verify_args(h1, h2, h3, h4):
    # Check they're all arrays of 4 arrays
    assert h1.shape[0] == 4, "h1_plus should be a shape (4, N) array"
    assert h2.shape[0] == 4, "h2_minus should be a shape (4, N) array"
    assert h3.shape[0] == 4, "h3_minus should be a shape (4, N) array"
    assert h4.shape[0] == 4, "h4_plus should be a shape (4, N) array"

    # Check they all contain the same number of particles
    n_particles = len(h1[0])
    assert h2.shape[1] == n_particles, "h2_minus and h1_plus are different lengths"
    assert h3.shape[1] == n_particles, "h3_minus and h1_plus are different lengths"
    assert h4.shape[1] == n_particles, "h4_minus and h1_plus are different lengths"


def helicity_param(
    h1_plus: np.ndarray, h2_minus: np.ndarray, h3_minus: np.ndarray, h4_plus: np.ndarray
) -> np.ndarray:
    """
    Find 5 dimensional four-body phase space parameterisation using invariant masses and helicity angles

    Our decay is `X -> h1+ h2- h3- h4+`

    Parameterisation comes from the original paper by Cabibbo and Maksymowicz- definitions are in the full documentation.
    In brief, our parameters are:
        Invariant mass of (h1, h4)
        Invariant mass of (h2, h3)
        Cosine angle of h1 wrt parent particle, in the CoM frame of + charged particle system
        Cosine angle of h2 wrt parent particle, in the CoM frame of - charged particle system
        Angle between + and - system decay planes

    :param h1_plus: array of +ve charged hadron parameters, (px, py, pz, energy).
                    Each entry in this array should be an N-length array of momenta; overall shape is (4, N) for N particles.
    :param h2_minus: array of -ve charged hadron parameters, (px, py, pz, energy).
                     Each entry in this array should be an N-length array of momenta; overall shape is (4, N) for N particles.
    :param h3_minus: array of -ve charged hadron parameters, (px, py, pz, energy).
                     Each entry in this array should be an N-length array of momenta; overall shape is (4, N) for N particles.
    :param h3_plus: array of +ve charged hadron parameters, (px, py, pz, energy).
                    Each entry in this array should be an N-length array of momenta; overall shape is (4, N) for N particles.

    :return: shape (N, 5) array of points in 5d phase space.

    """
    _verify_args(h1_plus, h2_minus, h3_minus, h4_plus)

    # Find invariant masses
    m_plus, m_minus = util.m_plus_minus(h1_plus, h2_minus, h3_minus, h4_plus)

    # Find costheta + and -
    d = np.add(h1_plus, np.add(h2_minus, np.add(h3_minus, h4_plus)))
    cos_theta_plus = util.cos_theta(h1_plus, h4_plus, d)
    cos_theta_minus = util.cos_theta(h2_minus, h3_minus, d)

    # Find phi
    cosphi = util.cos_phi(h1_plus, h2_minus, h3_minus, h4_plus)

    # Return
    return np.column_stack(
        (m_plus, m_minus, cos_theta_plus, cos_theta_minus, np.arccos(cosphi))
    )
