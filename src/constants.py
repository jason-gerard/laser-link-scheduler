import os
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SOURCES_ROOT = os.path.join(REPO_ROOT, "scenarios", "experiments")
REPORTS_ROOT = os.path.join(REPO_ROOT, "output", "reports")
PLOTS_ROOT = os.path.join(REPO_ROOT, "output", "plots")

# Minimum duration an edge i,j in state k can have
# There is a minimum amount of time required for acquisition, tracking, and pointing (ATP). If there is a contact in the
# contact topology, P, that is shorter than this, the contact most likely is not long enough to establish the link and
# send a meaningful amount of data and so it should be filtered out.
# d_min = t_atp + t_useful
#
# t_atp is then the delay imposed by the mechanical movement of the optical telescope gimbal from its current position
# to the direction of the target satellite, plus the link acquisition latency until bit-lock is achieved between the
# devices.
d_min = 5
# Maximum duration an edge i,j in state k can have
# This value decides the maximum duration of a state k. If a contact spans longer than t_max it will be split into
# multiple contacts of duration t_max, which then translates into multiple k states. There is a tradeoff with this
# variable between the overhead of setting up the link and being able to transition to contacts with different nodes
# to improve the fairness.
d_max = 600

# The default communication interface to use, a = 1 is the default laser that each node is equipped with.
# The default interface, a = 1, has the default bit rate of 100 mbps.
default_a = 1

# TODO: change descrition for general propourse.
alpha = 0.99
"""
    Alpha is a weighting factor that scales how much impact fairness has on the algorithm. If alpha is low it will only
    be used for tie breaking when multiple options have the same change in capacity. If alpha is high then increasing
    fairness will be used over increasing capacity in some cases i.e. if a node has little impact on capacity but has
    not been given an opportunity to transmit
    alpha must be set greater than or equal to 0 and less than or equal to 1, i.e. [0, 1]
    if alpha = 1 then only consider fairness
    if alpha = 0 then only consider capacity
"""


# The matrix A, contains the integer IDs of the communication interfaces for each node. Each laser
# communication interface is associated in an integer ID, a, where a >= 1.
# For nodes 0, 1, 2, 3
# A = [[1], [1, 3], [1, 1, 1, 1], [2, 2]]

# The list I, contains the number of communication interfaces each node has.
# For nodes 0, 1, 2, 3
# I = [1, 2, 4, 2]

# For now, we are assuming a constant and symmetric bitrate across all links, units are bits per second (bps). This
# value is taken from the NASA DSOC Mars communication demonstration where they achieved a 267 megabit per second
# bit rate, https://www.jpl.nasa.gov/news/nasas-tech-demo-streams-first-video-from-deep-space-via-laser.
# TODO update the code to us megabit as the base unit, switching now will cause integer overflow errors
# default_BIT_RATE = 267_000_000
default_BIT_RATE = 1000  # 1 kilobit per second
# This list R, contains the bit rates for each communication interface.
# Since a = 0 doesn't apply to any interface the bit_rate is just set to 0.
R = [0, default_BIT_RATE]

# The matrix P, represents the contact topology of the network. This contains for each state k, all possible contacts.
# This is the input to the algorithm.
# P[k][i][j]

# The matrix L, represents the contact plan of the network. This contains for each state k, the selects contacts. This
# is the output to the algorithm.
# L[k][i][j]

# The matrix W, represents the weights for each edge for each state k. This can be used by the max-weight or min-weight
# matching algorithm
# W[k][i][j]

# The list T, contains the durations for each k state
# For K = 3
# T = [2000, 3000, 800]

# The list X, contains the interplanetary node central body mapping, where x = 0 means that the node is not an IPN node.
# For nodes 0, 1, 2, 3
# X = [1, 3, 0, 3]

# The interplanetary range we define is any contact over 100,000 kilometers. We then convert this to light seconds to
# follow the ION contact plan standard
INTERPLANETARY_RANGE = 100_000 * 1_000 / 299_792_458

EARTH = "EARTH"
MARS = "MARS"

# Optical ground stations, sink nodes, T, from 9001 to 9012
DESTINATION_NODES = [str(i) for i in range(9001, 9013)]

# Mars Science sats from 2001 to 2064
SOURCE_NODES = [str(i) for i in range(2001, 2065)]

# Relay nodes from 1001 to 1012
RELAY_NODES = [str(i) for i in range(1001, 1013)]

# Relay nodes from 1001 to 1012 (EARTH or MARS, depends the scenario)
# Source nodes from 2001 to 2064 (MARS)
# Ground stations, destination nodes, from 9001 to 9012 (EARTH)
NODE_TO_PLANET_MAP = {
    # **{str(i): EARTH for i in range(1001, 1013)},
    **{str(i): MARS for i in range(1001, 1013)},
    **{str(i): MARS for i in range(2001, 2065)},
    **{str(i): EARTH for i in range(9001, 9013)},
}


# # Destination nodes from 1001 to 1008
# DESTINATION_NODES = [str(i) for i in range(1001, 1009)]

# # from 2001 to 2192
# SOURCE_NODES = [str(i) for i in range(2001, 2193)]

# # Relay nodes from 3001 to 3016
# RELAY_NODES = [str(i) for i in range(3001, 3017)]

# # Destination nodes from 1001 to 1008 (EARTH)
# # Source nodes from 2001 to 2192 (MARS)
# # Relay nodes from 3001 to 3016 (MARS)
# NODE_TO_PLANET_MAP = {
#     **{node_id: EARTH for node_id in DESTINATION_NODES},
#     **{node_id: MARS for node_id in SOURCE_NODES},
#     **{node_id: MARS for node_id in RELAY_NODES},
# }

SOURCE_NODE_BIT_RATE = 187  # DSOC Psyche @ 100 million km 50 mbps
# SOURCE_NODE_BIT_RATE = 1000  # DSOC Psyche @ 33 million km 267 mbps
RELAY_NODE_BIT_RATE = 4495  # LCRD @ 1.2 gbps
GS_NODE_BIT_RATE = 4495  # LCRD @ 1.2 gbps
BIT_RATES = {
    node_id: SOURCE_NODE_BIT_RATE
    if node_id in SOURCE_NODES
    else RELAY_NODE_BIT_RATE
    for node_id in NODE_TO_PLANET_MAP
}


def get_num_lasers(node_id: str):
    if node_id in SOURCE_NODES:
        return 1
    elif node_id in RELAY_NODES:
        # return 1
        return 2
    elif node_id in DESTINATION_NODES:
        return 1
        # return 2


class MLConfig:
    """
    MISSION LIFETIME
    -----
    We took as reference the NASA New Horizons spacecraft RTG:

        'Besides its suite of scientific instruments, New Horizons carries a cylindrical
        radioisotope thermoelectric generator (a spare from the Cassini mission)
        that provided about 250 W of power at launch (decaying to 200 W by the
        time of the Pluto encounter).' (11kg of plutonium oxide fuel)

        'https://science.nasa.gov/mission/new-horizons/#:~:text=Besides,encounter%29%2E'

        The minimum it is set at 69.9 W to function until the Low-Power Helio Science

        'https://www.jhuapl.edu/sites/default/files/2024-09/37-01-Hersman.pdf'

        The approximate rate of decay in power output is currently about 1.6% per year.

        'https://ntrs.nasa.gov/api/citations/20160001769/downloads/20160001769.pdf'

                λ = 0.016

    """

    SOURCE_NODE_INITIAL_POWER = 250.0  # Watts
    RELAY_NODE_INITIAL_POWER = 250.0  # Watts
    GS_NODE_INITIAL_POWER = float("inf")
    # TODO: Change it to 1.6% per year to match the decay of the New Horizons RTG,
    #       but for now we set it to 0.016e-3 per second to match the units of the calculations.
    DECAY_RATE = (
        0.016e-3  # per year, converted to per second for the calculations
    )

    @classmethod
    def get_initial_power(cls, node_id: str):
        # Obtain the mission lifetime by satellite type
        return (
            cls.RELAY_NODE_INITIAL_POWER
            if node_id in RELAY_NODES
            else cls.SOURCE_NODE_INITIAL_POWER
            if node_id in SOURCE_NODES
            else cls.GS_NODE_INITIAL_POWER
        )

    @classmethod
    def get_initial_powers(cls, node_ids: np.ndarray) -> np.ndarray:
        node_ids = np.asarray(node_ids)

        return np.select(
            [
                np.isin(node_ids, RELAY_NODES),
                np.isin(node_ids, SOURCE_NODES),
            ],
            [
                cls.RELAY_NODE_INITIAL_POWER,
                cls.SOURCE_NODE_INITIAL_POWER,
            ],
            default=cls.GS_NODE_INITIAL_POWER,
        )


class OPTConfig:
    """
    OPTICAL COMUNICATION TERMINAL
    ------------
    The optical communication terminal (OCT) energy consumption is defined across three
    states: idle, acquisition (PAT), and transmission.
    - 16-PPM -> 2⁴ -> 4 bits of data
    - 4W average transmit power.
    - Forward Error Correction (FEC) code rate (r = 1 if no coding), generally 2/3
    - Data rate of 267 mbps (Net information bit rate)
    - Duration of a single PPM time slot
    """

    AVG_TRANSMISSION_POWER = 4  # P_avg
    PEAK_TRANSMISSION_POWER = 64  # P_peak
    PAYLOAD_BITS = 4  # m = log2(L)
    SLOTS_PER_SYMBOL = 16  # L = 2^m
    FEC = 2 / 3  # r
    BIT_RATE = 267  # R_b
    TIME_SLOT = (FEC * PAYLOAD_BITS) / (BIT_RATE * (SLOTS_PER_SYMBOL + 1 / 4))
    GUARD_TIME = (  #  (L / 4) · T_slot
        (SLOTS_PER_SYMBOL / 4) * TIME_SLOT
    )
    SYMBOL_DURATION = (  # (L × T_slot) + T_guard
        (SLOTS_PER_SYMBOL * TIME_SLOT) + GUARD_TIME
    )
