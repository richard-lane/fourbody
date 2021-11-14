"""
Fcns for doing the parameterisation

"""
import numpy as np
import pylorentz


def invariant_masses(
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


def _m_plus_minus(k, pi1, pi2, pi3, verbose=False):
    """
    Invariant masses of + and - systems

    """
    if verbose:
        print("finding invariant masses")
    return invariant_masses(*np.add(k, pi3)), invariant_masses(*np.add(pi1, pi2))


def _boost(target, *particles):
    """
    Boost particles into the frame of target

    """
    # Create 4 vectors
    target_4v = pylorentz.Momentum4(target[3], *target[0:3])
    particles_4v = (pylorentz.Momentum4(p[3], *-p[0:3]) for p in particles)

    # Boost
    return (p.boost_particle(target_4v) for p in particles_4v)


def _cos_theta(hadron1, hadron2, d, verbose=False):
    """
    Cosine of helicity angle

    """
    if verbose:
        print("Finding theta")

    # Boost hadron1 and d into the CoM frame of hadron1 + hadron2
    com = np.add(hadron1, hadron2)
    boosted_hadron1, boosted_d = _boost(com, hadron1, d)

    # Evaluate + return
    return (
        boosted_hadron1.p_x * boosted_d.p_x
        + boosted_hadron1.p_y * boosted_d.p_y
        + boosted_hadron1.p_z * boosted_d.p_z
    ) / (boosted_hadron1.p * boosted_d.p)


def _cos_phi(k, pi1, pi2, pi3, verbose=False):
    if verbose:
        print("finding cos phi")

    # Find D 4 momentum
    d = np.add(k, np.add(pi1, np.add(pi2, pi3)))
    d_4v = pylorentz.Momentum4(d[3], *d[0:3])

    # Find momenta of all particles in the D frame
    k_4v, pi1_4v, pi2_4v, pi3_4v = _boost(d, k, pi1, pi2, pi3)

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


def helicity_param(k, pi1, pi2, pi3, t, verbose=False):
    """
    Parameterisation using helicity angles

    Uses:
        invariant mass of (k, pi3)
        invariant mass of (pi1, pi2)
        cosine helicity angle of pi3
        cosine helicity angle of pi1
        phi
        t

    Slightly computationally wasteful but idc

    """
    # Find invariant masses
    m_plus, m_minus = _m_plus_minus(k, pi1, pi2, pi3, verbose)

    # Find costheta + and -
    d = np.add(k, np.add(pi1, np.add(pi2, pi3)))
    cos_theta_plus = _cos_theta(k, pi3, d, verbose=True)
    cos_theta_minus = _cos_theta(pi1, pi2, d, verbose=True)

    # Find phi
    cosphi = _cos_phi(k, pi1, pi2, pi3, verbose)

    # Return
    return np.column_stack(
        (m_plus, m_minus, cos_theta_plus, cos_theta_minus, np.arccos(cosphi), t)
    )
