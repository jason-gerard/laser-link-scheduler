def survivability(E_b: float, P_c: float, P_g: float) -> float:
    """
    Calculate the survivability (L) of a satellite or node based on available stored
    and harvested energy.

    Survivability is defined as:
        L = E_b / (P_c − P_g), if P_c > P_g
        L = ∞, otherwise

    Parameters
    ----------
    E_b : float
        Total battery energy in Joules.
    P_c : float
        Average power consumption in Watts, derived from the OCT and RTG
        energy models.
    P_g : float
        Average power generation in Watts, derived from solar panels or
        other energy sources.

    Notes
    ----------
    If the node is purely RTG-powered (no recharge capability), P_g = 0.

    A node is considered survivable if its power consumption does not exceed its
    power generation (P_c ≤ P_g).

    This metric is useful for making scheduling decisions while ensuring
    operational continuity.

    Returns
    -------
    float
        The survivability (L) of the node. Returns a finite value if P_c > P_g,
        otherwise returns infinity.
    """
    if P_c > P_g:
        return E_b / (P_c - P_g)
    else:
        return float("inf")


if __name__ == "__main__":
    E_b = 5000.0  # Joules
    P_c = 50.0  # Watts
    P_g = 20.0  # Watts

    L = survivability(E_b, P_c, P_g)
    print("Parameters:")
    print(f"\tBattery Energy (E_b): {E_b} J")
    print(f"\tPower Consumption (P_c): {P_c} W")
    print(f"\tPower Generation (P_g): {P_g} W")

    print(f"Survivability L: {L} seconds")
