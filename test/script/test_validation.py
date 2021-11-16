"""
Check how the helicity parameterisation looks with phsp

"""

import pytest
import pylorentz
import numpy as np
import matplotlib.pyplot as plt
import phasespace as ps

from fourbody import param
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

    N = 50000  # Number of evts to generate, but some will get thrown away
    weights, particles = generator.generate(N, normalize_weights=True)

    # Create a mask for accept-reject based on weights
    keep_mask = (np.max(weights) * np.random.random(N)) < weights
    n_kept = np.sum(keep_mask)

    a = particles["a"].numpy()[keep_mask].T
    b = particles["b"].numpy()[keep_mask].T
    c = particles["c"].numpy()[keep_mask].T
    d = particles["d"].numpy()[keep_mask].T

    return a, b, c, d


def test_projections(_phsp):
    # Generate phsp
    pi1, pi2, k1, k2 = _phsp

    # Parametrise
    points = param.helicity_param(k1, pi1, k2, pi2)

    # Plot projections
    fig, ax = plt.subplots(2, 3, figsize=(12, 6))
    kw = {"histtype": "step"}
    labels = (
        r"$M(K^+\pi^+) /GeV$",
        r"$M(K^-\pi^-)$",
        r"$cos(\theta^+)$",
        r"$cos(\theta^-)$",
        r"$\phi$",
    )
    bins = (
        np.linspace(0.6, 1.4),
        np.linspace(0.6, 1.4),
        np.linspace(0.0, 1.0),
        np.linspace(0.0, 1.0),
        np.linspace(0.0, np.pi),
    )

    for i, (a, l, b) in enumerate(zip(ax.flatten()[:-1], labels, bins)):
        kw["bins"] = b
        a.hist(points[:, i], **kw, label="Phsp")
        a.set_xlabel(l)

    ax.flatten()[-1].legend()
    fig.set_tight_layout(True)

    path = "helicity_phsp.png"
    plt.savefig(path)
    plt.clf()


def test_phi_consistency(_phsp):
    """
    Check that sin2 + cos2 phi = 1

    """
    pi1, pi2, k1, k2 = _phsp

    sin = util.sin_phi(k1, pi1, k2, pi2)
    cos = util.cos_phi(k1, pi1, k2, pi2)

    sum = sin ** 2 + cos ** 2

    plt.hist(sum, bins=np.linspace(0.5, 1.5))
    plt.title(r"$sin^2(\phi) + cos^2(\phi)$")

    plt.savefig("sin_cos.png")
    plt.clf()


def test_boosts(_phsp):
    """
    Check that boosts correctly take us into the rest frame of our particle

    """
    pi1, pi2, k1, k2 = _phsp

    def _3momm(particle):
        if isinstance(particle, pylorentz.Momentum4):
            return np.sqrt(particle.p_x ** 2 + particle.p_y ** 2 + particle.p_z ** 2)
        return np.sqrt(particle[0] ** 2 + particle[1] ** 2 + particle[2] ** 2)

    def _plot(a, k, pi, title):
        """
        Plot particles on an axis
        """
        kw = {"histtype": "step", "bins": np.linspace(0.0, 2.0)}

        a.hist(_3momm(k), label=r"$K$", **kw)
        a.hist(_3momm(pi), label=r"$\pi$", **kw)

        combined = k + pi if isinstance(k, pylorentz.Momentum4) else np.add(k, pi)
        a.hist(_3momm(combined), label=r"$K\pi$", **kw)

        a.set_title(title)
        a.legend()
        a.set_xlabel(r"$|\vec{p}|$")
        a.set_ylabel("Count")

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    _plot(ax[0, 0], k1, pi1, "Lab Frame")

    # Boost particles into the k frame
    boosted_k, boosted_pi1 = boosts.boost(k1, k1, pi1)
    _plot(ax[0, 1], boosted_k, boosted_pi1, "K Frame")

    # Boost particles into the pi frame
    boosted_k, boosted_pi1 = boosts.boost(pi1, k1, pi1)
    _plot(ax[1, 0], boosted_k, boosted_pi1, title=r"$\pi$ Frame")

    # Boost particles into the k+pi frame
    boosted_k, boosted_pi1 = boosts.boost(np.add(pi1, k1), k1, pi1)
    _plot(ax[1, 1], boosted_k, boosted_pi1, title=r"$K\pi$ Frame")

    fig.suptitle(r"3 Momenta of K, $\pi$ in different frames")

    plt.savefig("boosts.png")


def test_correlation(_phsp):
    """
    Plot correlations

    """
    pi1, pi2, k1, k2 = _phsp
    points = param.helicity_param(k1, pi1, k2, pi2)

    d = len(points[0])
    corr = np.ones((d, d))

    for i in range(d):
        for j in range(d):
            corr[i, j] = np.abs(np.corrcoef(points[:, i], points[:, j])[0, 1])

    fig, ax = plt.subplots()
    labels = (
        r"$M(K^+\pi^+)$",
        r"$M(K^-\pi^-)$",
        r"$cos(\theta^+)$",
        r"$cos(\theta^-)$",
        r"$\phi$",
    )

    im = ax.imshow(corr)
    ax.set_title("Correlations")
    ax.set_xticks([i for i in range(d)])
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticks([i for i in range(d)])
    ax.set_yticklabels(labels)

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    fig.subplots_adjust(bottom=0.25)
    fig.subplots_adjust(left=0.1)

    plt.savefig("correlations.png")
