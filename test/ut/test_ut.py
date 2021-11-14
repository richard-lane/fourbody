"""
Unit tests

"""
import numpy as np

from helicity_param import parameterisation


def test_inv_mass_stationary():
    """
    Test invariant mass of a stationary particle gets calculated correctly

    """
    b0_mass = 5279.65
    momentum = np.array([[0.0]])
    energy = np.array([[b0_mass]])

    assert np.allclose(
        parameterisation.invariant_masses(momentum, momentum, momentum, energy),
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

    assert np.allclose(
        parameterisation.invariant_masses(px, py, pz, e), np.array([[pi_mass]])
    )


def test_m_plus_minus():
    """
    Test we get the right invariant masses out

    """
    k = np.array([[53.89743437], [-385.779829], [-1782.366763], [2116.4343876]])
    pi1 = np.array([[-241.70298533], [-13.62889997], [1166.66733619], [1199.66603207]])
    pi2 = np.array([[-1176.18918034], [612.05350502], [482.97638869], [1418.01851474]])
    pi3 = np.array([[463.9947313], [-212.64477997], [132.72300175], [545.53106553]])

    mplus, mminus = parameterisation._m_plus_minus(k, pi1, pi2, pi3)

    assert np.allclose(mplus, parameterisation.invariant_masses(*np.add(k, pi3)))
    assert np.allclose(mminus, parameterisation.invariant_masses(*np.add(pi1, pi2)))


def test_boost_no_op():
    """
    Test boosting a particle to the current frame

    """
    particle = np.array([[53.89743437], [-385.779829], [-1782.366763], [2116.4343876]])

    # Target particle has no 3-momentum; it's stationary, so our boost shouldn't do anything
    target = np.array([[0.0], [0.0], [0.0], [1.8]])

    boosted_particle, = parameterisation._boost(target, particle)

    assert np.allclose(
        [
            boosted_particle.p_x[0],
            boosted_particle.p_y[0],
            boosted_particle.p_z[0],
            boosted_particle.e[0],
        ],
        particle.flatten(),
    )


def test_boost():
    """
    Test boosting a particle

    """
    particle = np.array([[53.89743437], [-385.779829], [-1782.366763], [2116.4343876]])

    # Target particle moving in x-direction with gamma = 2.0
    target = np.array([[1.0], [0.0], [0.0], [2.0]])

    expected = np.array(
        [
            -0.375 * particle[3][0] + 2.0 * particle[0][0],  # px
            particle[1][0],
            particle[2][0],
            2.0 * particle[3][0] - 0.375 * particle[0][0],  # energy
        ]
    )

    boosted_particle, = parameterisation._boost(target, particle)

    assert np.allclose(
        [
            boosted_particle.p_x[0],
            boosted_particle.p_y[0],
            boosted_particle.p_z[0],
            boosted_particle.e[0],
        ],
        expected,
    )


def test_boost_multiple():
    """
    Test boosting multiple particles

    """
    a = np.array([[53.89743437], [-385.779829], [-1782.366763], [2116.4343876]])
    b = np.array([[-241.70298533], [-13.62889997], [1166.66733619], [1199.66603207]])

    # Target particles moving in x-direction with gamma = 2.0
    target = np.array([[1.0, 0.5], [0.0, 0.0], [0.0, 0.0], [2.0, 1.0]])
    expected_a, expected_b = (
        np.array(
            [[0.375 * x[3] + 2.0 * x[0]], [x[1]], [x[2]], [2.0 * x[3] + 0.375 * x[0]]]
        )
        for x in (a, b)
    )

    boosted_a, boosted_b = parameterisation._boost(target, a, b)

    assert np.allclose(
        [boosted_a.p_x[0], boosted_a.p_y[0], boosted_a.p_z[0], boosted_a.e[0]],
        expected_a,
    )
    assert np.allclose(
        [boosted_b.p_x[0], boosted_b.p_y[0], boosted_b.p_z[0], boosted_b.e[0]],
        expected_b,
    )

