"""
Fcns for doing the actual maths when we work out our parameterisation

Called `util` because I couldn't think of a better name. Maybe `geometry` or `maths`?

"""
import numpy as np
import pylorentz

from .boosts import boost


def _invariant_masses(
    px: np.ndarray, py: np.ndarray, pz: np.ndarray, energy: np.ndarray
):
    """
    Find the invariant masses of a collection of particles represented by their kinematic data

    :param px: particle x momenta
    :param py: particle y momenta
    :param pz: particle z momenta
    :param energy: particle energies
    :returns: array of particle invariant masses

    """
    return np.sqrt(energy ** 2 - px ** 2 - py ** 2 - pz ** 2)


def m_plus_minus(k, pi1, pi2, pi3):
    """
    Invariant masses of + and - systems

    """
    return _invariant_masses(*np.add(k, pi3)), _invariant_masses(*np.add(pi1, pi2))


def cos_theta(hadron1, hadron2, d):
    """
    Cosine of helicity angle

    """
    # Boost hadron1 and d into the CoM frame of hadron1 + hadron2
    com = np.add(hadron1, hadron2)
    boosted_hadron1, boosted_d = boost(com, hadron1, d)

    # Evaluate + return
    return (
        boosted_hadron1.p_x * boosted_d.p_x
        + boosted_hadron1.p_y * boosted_d.p_y
        + boosted_hadron1.p_z * boosted_d.p_z
    ) / (boosted_hadron1.p * boosted_d.p)


def sin_phi(k, pi1, pi2, pi3):
    """
    Sin of helicity angle
    """
    # Find D 4 momentum
    d = np.add(k, np.add(pi1, np.add(pi2, pi3)))
    d_4v = pylorentz.Momentum4(d[3], *d[0:3])

    # Find momenta of all particles in the D frame
    k_4v, pi1_4v, pi2_4v, pi3_4v = boost(d, k, pi1, pi2, pi3)

    # Find three momenta
    k_p = np.column_stack((k_4v.p_x, k_4v.p_y, k_4v.p_z))
    pi1_p = np.column_stack((pi1_4v.p_x, pi1_4v.p_y, pi1_4v.p_z))
    pi2_p = np.column_stack((pi2_4v.p_x, pi2_4v.p_y, pi2_4v.p_z))
    pi3_p = np.column_stack((pi3_4v.p_x, pi3_4v.p_y, pi3_4v.p_z))

    # Take cross products + find magnitudes
    plus_cross = np.cross(k_p, pi3_p)
    minus_cross = np.cross(pi1_p, pi2_p)
    plus_cross_mag = np.linalg.norm(plus_cross, axis=1)
    minus_cross_mag = np.linalg.norm(minus_cross, axis=1)

    # Take cross products of cross products and scale
    sf = plus_cross_mag * minus_cross_mag
    prod = np.cross(plus_cross, minus_cross) / sf[:, None]

    return (prod * (pi1_p + pi2_p)).sum(1) / np.linalg.norm(pi1_p + pi2_p, axis=1)


def cos_phi(k, pi1, pi2, pi3):
    # Find D 4 momentum
    d = np.add(k, np.add(pi1, np.add(pi2, pi3)))

    # Find momenta of all particles in the D frame
    k_4v, pi1_4v, pi2_4v, pi3_4v = boost(d, k, pi1, pi2, pi3)

    # Find three momenta
    k_p = np.column_stack((k_4v.p_x, k_4v.p_y, k_4v.p_z))
    pi1_p = np.column_stack((pi1_4v.p_x, pi1_4v.p_y, pi1_4v.p_z))
    pi2_p = np.column_stack((pi2_4v.p_x, pi2_4v.p_y, pi2_4v.p_z))
    pi3_p = np.column_stack((pi3_4v.p_x, pi3_4v.p_y, pi3_4v.p_z))

    # Take cross products + find magnitudes
    plus_cross = np.cross(k_p, pi3_p)
    minus_cross = np.cross(pi1_p, pi2_p)
    plus_cross_mag = np.linalg.norm(plus_cross, axis=1)
    minus_cross_mag = np.linalg.norm(minus_cross, axis=1)

    # Dot them together by multiplying elementwise then summing
    return (plus_cross * minus_cross).sum(1) / (plus_cross_mag * minus_cross_mag)
