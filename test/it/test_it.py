"""
Integration tests

"""
import pytest
import numpy as np
import phasespace as ps

from fourbody import boosts
from fourbody import util


@pytest.fixture(scope="session")
def _phsp():
    """
    Make some phsp evts

    """
    a_mass = 0.139570
    c_mass = 0.493677
    b_mass = a_mass
    d_mass = c_mass

    x_mass = 1.86484

    generator = ps.nbody_decay(
        x_mass, (a_mass, b_mass, c_mass, d_mass), names=("a", "b", "c", "d")
    )

    N = 1000  # Number of evts to generate, but some will get thrown away
    weights, particles = generator.generate(N, normalize_weights=True)

    # Create a mask for accept-reject based on weights
    keep_mask = (np.max(weights) * np.random.random(N)) < weights
    n_kept = np.sum(keep_mask)

    a = particles["a"].numpy()[keep_mask].T
    b = particles["b"].numpy()[keep_mask].T
    c = particles["c"].numpy()[keep_mask].T
    d = particles["d"].numpy()[keep_mask].T

    return a, b, c, d


def test_boost(_phsp):
    """
    Boost into the rest frame of a particle, check its three momenta are zero

    """

    a, _, _, _ = _phsp

    (boosted_a,) = boosts.boost(a, a)

    assert np.allclose(boosted_a.p_x, np.zeros_like(boosted_a.p_x))
    assert np.allclose(boosted_a.p_y, np.zeros_like(boosted_a.p_y))
    assert np.allclose(boosted_a.p_z, np.zeros_like(boosted_a.p_z))


def test_boost_sum(_phsp):
    """
    Boost into the rest frame of a system of two particles, check that our combined three momenta are 0

    """
    a, b, _, _ = _phsp
    n_particles = len(a[0])

    boosted_a, boosted_b = boosts.boost(np.add(a, b), a, b)

    assert np.allclose(boosted_a.p_x + boosted_b.p_x, np.zeros(n_particles))
    assert np.allclose(boosted_a.p_y + boosted_b.p_y, np.zeros(n_particles))
    assert np.allclose(boosted_a.p_z + boosted_b.p_z, np.zeros(n_particles))


def test_sin_cos_phi_consistency(_phsp):
    """
    Check that sin2 + cos2 phi = 1 for our phsp evts

    """
    a, b, c, d = _phsp

    sin_phi = util.sin_phi(a, b, c, d)
    cos_phi = util.cos_phi(a, b, c, d)

    sum = sin_phi**2 + cos_phi**2

    assert np.allclose(sum, np.ones_like(a[0]))
