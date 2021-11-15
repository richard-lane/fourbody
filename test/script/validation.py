"""
Check how the helicity parameterisation looks with phsp

"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
import phasespace as ps

from helicity_param import parameterisation


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

    # Decay times will just be an exponential
    times = np.random.exponential(size=n_kept)

    return a, b, c, d, times


def test_projections(_phsp):
    # Generate phsp
    k, pi1, pi2, pi3, t = _phsp

    # Parametrise
    points = parameterisation.helicity_param(k, pi1, pi2, pi3, t, verbose=True)

    # Plot projections
    fig, ax = plt.subplots(2, 3, figsize=(12, 6))
    kw = {"histtype": "step"}
    labels = (
        r"$M(K^+\pi^+) /GeV$",
        r"$M(\pi_1^-\pi_2^-)$",
        r"$cos(\theta^+)$",
        r"$cos(\theta^-)$",
        r"$\phi$",
        r"$t /ps$",
    )
    bins = (
        np.linspace(0.5, 1.7),
        np.linspace(0.2, 1.5),
        np.linspace(0.0, 1.0),
        np.linspace(0.0, 1.0),
        np.linspace(0.0, np.pi),
        np.linspace(0.0, 3.0),
    )

    for i, (a, l, b) in enumerate(zip(ax.flatten(), labels, bins)):
        kw["bins"] = b
        a.hist(points[:, i], **kw, label="Phsp")
        a.set_xlabel(l)

    ax.flatten()[-1].legend()
    fig.set_tight_layout(True)

    phi = points[:, 4]

    path = "helicity_phsp.png"
    plt.savefig(path)


def test_phi_consistency():
    """
    Check that sin2 + cos2 phi = 1

    """
    ...


def test_boosts():
    """
    Check that boosts correctly take us into the rest frame of our particle

    """
    ...
