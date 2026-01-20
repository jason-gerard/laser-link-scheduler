def transmission_energy(power: float, duration: float) -> float:
    """
    Compute the energy consumed during transmission.

    Parameters
    ----------
    power : float
        Transmission power in watts (W).
    duration : float
        Duration of transmission in seconds (s).

    Returns
    -------
    float
        Energy consumed during transmission in joules (J).
    """
    return power * duration


# TODO: implement this function
def optical_trasnmit_power() -> float: ...


def transmission_duration(
    N_bits: int,
    M: int,
    L: int,
    T_chip: float,
    T_guard: float,
) -> float:
    """
    Compute the total transmission duration for pulse position modulation (PPM).

    Parameters
    ----------
    N_bits : int
        Number of bits in the message.
    M : int
        Bits encoded per slot (modulation order).
    L : int
        Number of slots.
    T_chip : float
        Duration of each slot in seconds.
    T_guard : float
        Inter-slot guard time in seconds.

    Returns
    -------
    float
        Total transmission duration (T_tx) in seconds.
    """
    if L != 2**M:
        raise ValueError("L must be equal to 2^M")

    return (N_bits / (2**M)) * (M / (L * T_chip + T_guard))


def bit_rate(
    M: int,
    L: int,
    T_chip: float,
    T_guard: float,
) -> float:
    """
    Compute the bit rate for pulse position modulation (PPM).

    Parameters
    ----------
    M : int
        Bits encoded per slot (modulation order).
    L : int
        Number of slots.
    T_chip : float
        Duration of each slot in seconds.
    T_guard : float
        Inter-slot guard time in seconds.

    Returns
    -------
    float
        Bit rate (R_b) in bits per second (bps).
    """
    if L != 2**M:
        raise ValueError("L must be equal to 2^M")

    return M / (L * T_chip + T_guard)


if __name__ == "__main__":
    power = 5.0  # watts
    duration = 10.0  # seconds

    energy = transmission_energy(power, duration)
    print("Parameters:")
    print(f"\tPower: {power} W")
    print(f"\tDuration: {duration} s")
    print(f"Transmission Energy: {energy} J\n")

    N_bits = 1e6  # bits
    M = 40  # bits per slot
    L = 2**M  # slots
    T_chip = 1e-6  # seconds
    T_guard = 1e-7  # seconds

    duration_tx = transmission_duration(N_bits, M, L, T_chip, T_guard)
    print("Parameters:")
    print(f"\tNumber of Bits (N_bits): {N_bits} bits")
    print(f"\tBits per Slot (M): {M} bits")
    print(f"\tNumber of Slots (L): {L}")
    print(f"\tSlot Duration (T_chip): {T_chip} s")
    print(f"\tGuard Time (T_guard): {T_guard} s")
    print(f"Transmission Duration: {duration_tx} s\n")

    R_b = bit_rate(M, L, T_chip, T_guard)
    print("Parameters:")
    print(f"\tBits per Slot (M): {M} bits")
    print(f"\tNumber of Slots (L): {L}")
    print(f"\tSlot Duration (T_chip): {T_chip} s")
    print(f"\tGuard Time (T_guard): {T_guard} s")
    print(f"Bit Rate: {R_b} bps")
