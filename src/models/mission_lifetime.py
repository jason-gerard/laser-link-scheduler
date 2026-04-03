import numpy as np


def mission_lifetime(P0: float, decay_constant: float, P_min: float) -> float:
    """
    Compute mission lifetime under an RTG power model with exponential decay.

    The radioisotope thermoelectric generator (RTG) is modeled as an exponentially decaying power source:
        P(t) = P0 * exp(-λ * t)
        where P(t) is the available power at time t,
        P0 is the initial power, and λ is the decay constant.

    Mission lifetime L_m is defined as the duration for which P(t) remains >= P_min.

    Parameters
    ----------
    P0 : float
        Initial available power at t = 0. Units: power (e.g., W).
    decay_constant : float
        Exponential decay constant λ. Units: 1 / time (must be > 0 for finite decay).
        The time unit of the returned lifetime matches the inverse of this unit.
    P_min : float
        Minimum operational power threshold. Units: power (same as P0).

    Returns
    -------
    float
        Mission lifetime L_m in the same time units as 1/λ.
        For λ > 0 and P0 >= P_min: L_m = (1/λ) * ln(P0 / P_min)
        If P0 < P_min, returns 0.0.

    Raises
    ------
    ValueError
        If P0 <= 0, P_min <= 0, or decay_constant <= 0.
    """

    if P0 <= 0:
        raise ValueError("Initial power P0 must be greater than 0.")
    if P_min <= 0:
        raise ValueError(
            "Minimum operational power P_min must be greater than 0."
        )
    if decay_constant <= 0:
        raise ValueError(
            "Decay constant must be greater than 0 for finite decay."
        )
    if P0 < P_min:
        return 0.0

    lifetime = (1 / decay_constant) * np.log(P0 / P_min)
    return lifetime


def generating_power(
    time: float,  # t
    initial_power: float,  # P_0
    decay_constant: float,  # λ
):
    """
    Energy Source and Generation Modeling
    -----
    The radioisotope thermoelectric generator (RTG) is modeled as an exponentially decaying power source:
        P(t) = P0 * exp(-λ * t)
        where P(t) is the available power at time t,
        P0 is the initial power, and λ is the decay constant.
    """

    return initial_power * np.exp(decay_constant * time)


def generated_energy(
    from_time: float,
    to_time: float,
    initial_power: float | np.ndarray,
    decay_constant: float,
):
    """
    Compute the generated energy during a time interval [from_time, to_time] under the RTG power model.

    The generated energy E during the interval can be computed as the integral of the power function P(t) over that interval:
        E = ∫[from_time, to_time] P(t) dt
          = ∫[from_time, to_time] P0 * exp(-λ * t) dt
          = (P0 / λ) * [exp(-λ * from_time) - exp(-λ * to_time)]

    Parameters
    ----------
    from_time : float
        Start of the time interval. Units: time (same as 1/λ).
    to_time : float
        End of the time interval. Units: time (same as 1/λ).
    initial_power : float
        Initial available power at t = 0. Units: power (e.g., W).
    decay_constant : float
        Exponential decay constant λ. Units: 1 / time (must be > 0 for finite decay).

    Returns
    -------
    float
        Generated energy E during the interval [from_time, to_time]. Units: energy (e.g., J).
    """
    if decay_constant <= 0:
        raise ValueError(
            "Decay constant must be greater than 0 for finite decay."
        )
    if to_time < from_time:
        raise ValueError("to_time must be greater than or equal to from_time.")
    if np.any(initial_power <= 0):
        raise ValueError("Initial power must be greater than 0.")

    energy = (initial_power / decay_constant) * (
        np.exp(-decay_constant * from_time) - np.exp(-decay_constant * to_time)
    )
    return energy


if __name__ == "__main__":
    # TODO: Re do.
    P0 = 250.0  # Initial power in Watts
    decay_constant = 0.013  # Decay constant in 1/years
    P_min = 60.0  # Minimum operational power in Watts
    lifetime = mission_lifetime(P0, decay_constant, P_min)
    print(f"Mission lifetime under RTG power model: {lifetime:.2f} years")
