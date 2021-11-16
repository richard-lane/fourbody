"""
Fcns for doing things with Lorentz boosts

"""
import numpy as np
import pylorentz


def _stationary(particles):
    """
    Returns bool mask of whether particles are stationary

    """
    # Check if 3 momenta components are close to 0
    zeros = np.isclose(particles[:3], 0.0)

    # If all three components are close to 0 then our particle is stationary
    return np.all(zeros, axis=0)


def _convert_to_pylorentz(*particles):
    """
    Just converts our particles to pylorentz Momentum4 instances

    """
    return (pylorentz.Momentum4(p[3], *p[0:3]) for p in particles)


def _boost_not_stationary(target, *particles):
    """
    Boost particles into another's rest frame, assuming the target particles are not stationary

    """
    (target_4v,) = _convert_to_pylorentz(target)
    particles_4v = _convert_to_pylorentz(*particles)

    target_4v = pylorentz.Momentum4(
        target_4v.e, -target_4v.p_x, -target_4v.p_y, -target_4v.p_z
    )

    return (p.boost_particle(target_4v) for p in particles_4v)


def boost(target, *particles):
    """
    Boost particles into the frame of target

    NB this returns a generator, so call it like:

    boosted_particle, = _boost(target, particle)
    (boosted_k, boosted_pi) = _boost(target, k, pi)

    note the comma

    """
    # Find whether our target particles are moving
    stationary = _stationary(target)

    # No particles are stationary - probably the mainline case
    if np.all(~stationary):
        return _boost_not_stationary(target, *particles)

    # All particles are stationary - also relatively common if our particles were e.g. generated from an amplitude model
    elif np.all(stationary):
        return _convert_to_pylorentz(*particles)

    # Boosting particles into a stationary frame can cause NaNs to appear
    # If we get here, it means some but not all of the target particles are stationary
    # If they are, undo the boost by replacing the calculated particle momenta with their original values
    # Evaluted afterwards so we can use the pylorentz array functionality
    raise NotImplementedError(
        "I haven't got round to implementing the case where only some of the target particles are moving"
    )
