"""
Unit tests

"""
import numpy as np
import pytest
import pylorentz

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


def test_stationary():
    """
    Check if we can identify whether particles are stationary

    """
    particles = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [1.0, 25.0]])

    stationary = parameterisation._stationary(particles)

    assert stationary[0] == True
    assert stationary[1] == False


def test_convert_one():
    """
    Check converting arrays for one particle to pylorentz gives us an obj of the correct type

    """
    particles = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [1.0, 25.0]])

    converted, = parameterisation._convert_to_pylorentz(particles)

    assert isinstance(converted, pylorentz.Momentum4)


def test_convert_multiple():
    """
    Check converting arrays for two particles to pylorentz gives us an obj of the correct type

    """
    particles = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [1.0, 25.0]])

    a, b = parameterisation._convert_to_pylorentz(particles, particles)

    assert isinstance(a, pylorentz.Momentum4)
    assert isinstance(b, pylorentz.Momentum4)


def test_boost_not_stationary():
    """
    Test boosting a particle if we assume it is not stationary

    """
    particle = np.array([[53.89743437], [-385.779829], [-1782.366763], [2116.4343876]])

    # Target particle moving in x-direction with gamma = 2.0
    m = 1.0
    gamma, beta = 2.0, np.sqrt(3) / 2.0
    target = np.array([[gamma * beta * m], [0.0], [0.0], [gamma * m]])

    # Might have accidentally done the boost backwards. We'll see if it affects anything...
    expected = np.array(
        [
            beta * gamma * particle[3][0] + gamma * particle[0][0],  # px
            particle[1][0],
            particle[2][0],
            gamma * particle[3][0] + beta * gamma * particle[0][0],  # energy
        ]
    )

    boosted_particle, = parameterisation._boost_not_stationary(target, particle)

    assert np.allclose(
        [
            boosted_particle.p_x[0],
            boosted_particle.p_y[0],
            boosted_particle.p_z[0],
            boosted_particle.e[0],
        ],
        expected,
    )


def test_no_op_boost():
    """
    Test boosting a particle works when we're already in the target frame

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


def test_all_moving_boost():
    """
    Test boost where all target particles are moving

    """
    # Target particles moving in x-direction with gamma = 2.0
    particle = np.array(
        [
            [53.89743437, -53.89743437],
            [-385.779829, -385.779829],
            [-1782.366763, -1782.366763],
            [2116.4343876, 2116.4343876],
        ]
    )

    m1, m2 = 1.0, 5.0
    gamma, beta = 2.0, np.sqrt(3) / 2.0
    target = np.array(
        [
            [gamma * beta * m1, gamma * beta * m2],
            [0.0, 0.0],
            [0.0, 0.0],
            [gamma * m1, gamma * m2],
        ]
    )

    # Might have accidentally done the boost backwards. We'll see if it affects anything...
    expected_1 = np.array(
        [
            beta * gamma * particle[3][0] + gamma * particle[0][0],  # px
            particle[1][0],
            particle[2][0],
            gamma * particle[3][0] + beta * gamma * particle[0][0],  # energy
        ]
    )
    expected_2 = np.array(
        [
            beta * gamma * particle[3][1] + gamma * particle[0][1],  # px
            particle[1][1],
            particle[2][1],
            gamma * particle[3][1] + beta * gamma * particle[0][1],  # energy
        ]
    )

    boosted, = parameterisation._boost(target, tuple(particle))

    assert np.allclose(
        [boosted.p_x[0], boosted.p_y[0], boosted.p_z[0], boosted.e[0]], expected_1
    )
    assert np.allclose(
        [boosted.p_x[1], boosted.p_y[1], boosted.p_z[1], boosted.e[1]], expected_2
    )


def test_some_moving_boost():
    particle = np.array(
        [
            [53.89743437, -53.89743437],
            [-385.779829, -385.779829],
            [-1782.366763, -1782.366763],
            [2116.4343876, 2116.4343876],
        ]
    )

    m1, m2 = 1.0, 5.0
    gamma, beta = 2.0, np.sqrt(3) / 2.0
    target = np.array(
        [[gamma * beta * m1, 0], [0.0, 0.0], [0.0, 0.0], [gamma * m1, m2]]
    )
    with pytest.raises(NotImplementedError):
        parameterisation._boost(target, particle)
