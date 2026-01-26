import math

import numpy as np


IPN_CPA_SLEW_AZ = 0.0
IPN_CPA_SLEW_EL = 0.0
IPN_FSM_TIP = 0.114592  # deg/s, 2 milliradian/s
IPN_FSM_TILT = 0.114592  # deg/s, 2 milliradian/s
IPN_DWELL_TIME = 0.5  # sec
IPN_BEAM_WIDTH = 0.2  # deg
IPN_TX_OUTPUT_PWR = 0.0
IPN_FOU = 2.0  # deg
IPN_FOU_R = IPN_FOU / 2  # deg
IPN_QC_FOV = 0.0  # Must be larger than the field of uncertainty

LEO_CPA_SLEW_AZ = 0.0
LEO_CPA_SLEW_EL = 0.0
LEO_FSM_TIP = 0.5  # deg/s, 2 milliradian/s
LEO_FSM_TILT = 0.5  # deg/s, 2 milliradian/s
LEO_DWELL_TIME = 0.5  # sec
LEO_BEAM_WIDTH = 0.2  # deg
LEO_TX_OUTPUT_PWR = 0.0
LEO_FOU = 1.5  # deg
LEO_FOU_R = LEO_FOU / 2  # deg
LEO_QC_FOV = 0.0


def link_acq_delay(
    R: float, d: float, tip_rate: float, tilt_rate: float, dwell_time: float
) -> float:
    def seek_stare_arch_hex_spiral_acq_delay() -> float:
        N = (2 * math.pi * math.pow(R, 2)) / (math.sqrt(3) * math.pow(d, 2))
        N = math.ceil(N)

        N_revolutions = math.ceil(N / 6)
        N_diag = N_revolutions * 4
        N_az = N_revolutions * 2

        T_slew = (N_diag * (d / min(tip_rate, tilt_rate))) + (
            N_az * (d / tip_rate)
        )

        return T_slew + (N * dwell_time)

    def acq_to_track_delay() -> float:
        return 1.0

    return seek_stare_arch_hex_spiral_acq_delay() + acq_to_track_delay()


def link_acq_delay_ipn() -> float:
    return link_acq_delay(
        IPN_FOU_R,
        IPN_BEAM_WIDTH,
        IPN_FSM_TIP,
        IPN_FSM_TILT,
        IPN_DWELL_TIME,
    )


def link_acq_delay_leo() -> float:
    return link_acq_delay(
        LEO_FOU_R,
        LEO_BEAM_WIDTH,
        LEO_FSM_TIP,
        LEO_FSM_TILT,
        LEO_DWELL_TIME,
    )


np.random.seed(42)


def link_acq_delay_ipn_rand() -> float:
    fou_r = np.clip(
        np.random.normal(loc=IPN_FOU_R, scale=0.010), 0.85, 1.15
    )  # Clamp to [0.5°, 1.5°]
    beam_width = np.clip(
        np.random.normal(loc=IPN_BEAM_WIDTH, scale=0.005), 0.18, 0.22
    )  # Clamp to [0.1°, 0.3°]
    return link_acq_delay(
        fou_r,
        beam_width,
        IPN_FSM_TIP,
        IPN_FSM_TILT,
        IPN_DWELL_TIME,
    )


def link_acq_delay_leo_rand() -> float:
    fou_r = np.clip(
        np.random.normal(loc=LEO_FOU_R, scale=0.01), 0.6, 0.9
    )  # Clamp to [0.6°, 0.9°]
    beam_width = np.clip(
        np.random.normal(loc=LEO_BEAM_WIDTH, scale=0.005), 0.15, 0.25
    )  # Clamp to [0.15°, 0.25°]
    return link_acq_delay(
        fou_r,
        beam_width,
        LEO_FSM_TIP,
        LEO_FSM_TILT,
        LEO_DWELL_TIME,
    )


def link_acq_delay_ipn_fou(fou_r) -> float:
    return link_acq_delay(
        fou_r,
        IPN_BEAM_WIDTH,
        IPN_FSM_TIP,
        IPN_FSM_TILT,
        IPN_DWELL_TIME,
    )


def link_acq_delay_leo_fou(fou_r) -> float:
    return link_acq_delay(
        fou_r,
        LEO_BEAM_WIDTH,
        LEO_FSM_TIP,
        LEO_FSM_TILT,
        LEO_DWELL_TIME,
    )


if __name__ == "__main__":
    D_acq_ipn = link_acq_delay_ipn()  # seconds
    print(
        f"IPN link acq takes {D_acq_ipn} seconds or {D_acq_ipn / 60} minutes"
    )

    D_acq_LEO = link_acq_delay_leo()  # seconds
    print(
        f"LEO optical link acq takes {D_acq_LEO} seconds or {D_acq_LEO / 60} minutes"
    )
