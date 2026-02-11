SOURCES_ROOT = "experiments"
REPORTS_ROOT = "reports"

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

# Alpha is a weighting factor that scales how much impact fairness has on the algorithm. If alpha is low it will only
# be used for tie breaking when multiple options have the same change in capacity. If alpha is high then increasing
# fairness will be used over increasing capacity in some cases i.e. if a node has little impact on capacity but has
# not been given an opportunity to transmit
# alpha must be set greater than or equal to 0 and less than or equal to 1, i.e. [0, 1]
# if alpha = 1 then only consider fairness
# if alpha = 0 then only consider capacity

# SCENARIO: mars_relay_earth_scenario
# lls_alpha = 0.3
# otls_alpha = None

# SCENARIO: mars_earth_relay_scenario_inc
lls_alpha = 0.5
otls_alpha = 0.35

# SCENARIO: dte_mars_earth_scenario_inc
# lls_alpha = 0.4
# otls_alpha = 0.35

# SCENARIO: dte_realistic_scenario
# lls_alpha = 0.90
# otls_alpha = 0.90

# The matrix A, contains the integer IDs of the communication interfaces for each node. Each laser
# communication interface is associated in an integer ID, a, where a >= 1.
# For nodes 0, 1, 2, 3
# A = [[1], [1, 3], [1, 1, 1, 1], [2, 2]]

# The list I, contains the number of communication interfaces each node has.
# For nodes 0, 1, 2, 3
# I = [1, 2, 4, 2]

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

should_bypass_retargeting_time = False

# The interplanetary range we define is any contact over 100,000 kilometers. We then convert this to light seconds to
# follow the ION contact plan standard
INTERPLANETARY_RANGE = 100_000 * 1_000 / 299_792_458

EARTH = "EARTH"
MARS = "MARS"

# Optical ground stations, sink nodes, T
DESTINATION_NODES = [
    "9001", "9002", "9003", "9004", "9005", "9006", "9007", "9008", "9009", "9010", "9011", "9012",
]

# Source nodes, S
SOURCE_NODES = [
    "2001", "2002", "2003", "2004", "2005", "2006", "2007", "2008", "2009", "2010",
    "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020",
    "2021", "2022", "2023", "2024", "2025", "2026", "2027", "2028", "2029", "2030",
    "2031", "2032", "2033", "2034", "2035", "2036", "2037", "2038", "2039", "2040",
    "2041", "2042", "2043", "2044", "2045", "2046", "2047", "2048", "2049", "2050",
    "2051", "2052", "2053", "2054", "2055", "2056", "2057", "2058", "2059", "2060",
    "2061", "2062", "2063", "2064"
]

# Relay nodes, R
RELAY_NODES = [
    "1001", "1002", "1003", "1004", "1005", "1006", "1007", "1008", "1009", "1010", "1011", "1012"
]

NODE_TO_PLANET_MAP = {
    # SCENARIO: Earth Relay
    "1001": EARTH, "1002": EARTH, "1003": EARTH, "1004": EARTH, "1005": EARTH, "1006": EARTH, "1007": EARTH, "1008": EARTH,
    # SCENARIO: Mars Relay
    # "1001": MARS, "1002": MARS, "1003": MARS, "1004": MARS, "1005": MARS,
    # "1006": MARS, "1007": MARS, "1008": MARS, "1009": MARS, "1010": MARS,
    # "1011": MARS, "1012": MARS,

    "2001": MARS, "2002": MARS, "2003": MARS, "2004": MARS, "2005": MARS, "2006": MARS, "2007": MARS, "2008": MARS,
    "2009": MARS, "2010": MARS, "2011": MARS, "2012": MARS, "2013": MARS, "2014": MARS, "2015": MARS, "2016": MARS,
    "2017": MARS, "2018": MARS, "2019": MARS, "2020": MARS, "2021": MARS, "2022": MARS, "2023": MARS, "2024": MARS,
    "2025": MARS, "2026": MARS, "2027": MARS, "2028": MARS, "2029": MARS, "2030": MARS, "2031": MARS, "2032": MARS,
    "2033": MARS, "2034": MARS, "2035": MARS, "2036": MARS, "2037": MARS, "2038": MARS, "2039": MARS, "2040": MARS,
    "2041": MARS, "2042": MARS, "2043": MARS, "2044": MARS, "2045": MARS, "2046": MARS, "2047": MARS, "2048": MARS,
    "2049": MARS, "2050": MARS, "2051": MARS, "2052": MARS, "2053": MARS, "2054": MARS, "2055": MARS, "2056": MARS,
    "2057": MARS, "2058": MARS, "2059": MARS, "2060": MARS, "2061": MARS, "2062": MARS, "2063": MARS, "2064": MARS,

    "9001": EARTH, "9002": EARTH, "9003": EARTH,
    "9004": EARTH, "9005": EARTH, "9006": EARTH,
    "9007": EARTH, "9008": EARTH, "9009": EARTH,
    "9010": EARTH, "9011": EARTH, "9012": EARTH,
}

# SCENARIO: Earth Relay
# For now, we are assuming a constant and symmetric bitrate across all links, units are bits per second (bps). This
# value is taken from the NASA DSOC Mars communication demonstration where they achieved a 267 megabit per second
# bit rate, https://www.jpl.nasa.gov/news/nasas-tech-demo-streams-first-video-from-deep-space-via-laser.
# SOURCE_NODE_BIT_RATE = 267  # DSOC Psyche @ 33 million km 267 mbps
ORBITER_NODE_BIT_RATE = 50  # DSOC Psyche @ 100 million km 50 mbps
LANDER_NODE_BIT_RATE = 25  # 25 mbps
RELAY_NODE_BIT_RATE = 1200  # LCRD @ 1.2 gbps
OGS_NODE_BIT_RATE = 1200  # LCRD @ 1.2 gbps

# SCENARIO: Mars relay to Earth only link data rates
# ORBITER_NODE_BIT_RATE = 50
# OGS_NODE_BIT_RATE = 50

BIT_RATES = {
    # SCENARIO: Earth Relay
    "1001": OGS_NODE_BIT_RATE, "1002": OGS_NODE_BIT_RATE, "1003": OGS_NODE_BIT_RATE, "1004": OGS_NODE_BIT_RATE,
    "1005": OGS_NODE_BIT_RATE, "1006": OGS_NODE_BIT_RATE, "1007": OGS_NODE_BIT_RATE, "1008": OGS_NODE_BIT_RATE, 
    # SCENARIO: Mars Relay
    # "1001": ORBITER_NODE_BIT_RATE, "1002": ORBITER_NODE_BIT_RATE, "1003": ORBITER_NODE_BIT_RATE,
    # "1004": ORBITER_NODE_BIT_RATE, "1005": ORBITER_NODE_BIT_RATE, "1006": ORBITER_NODE_BIT_RATE,
    # "1007": ORBITER_NODE_BIT_RATE, "1008": ORBITER_NODE_BIT_RATE, "1009": ORBITER_NODE_BIT_RATE,
    # "1010": ORBITER_NODE_BIT_RATE, "1011": ORBITER_NODE_BIT_RATE, "1012": ORBITER_NODE_BIT_RATE,

    # SCENARIO: Realistic scenario
    # "2001": LANDER_NODE_BIT_RATE, "2002": LANDER_NODE_BIT_RATE, "2003": LANDER_NODE_BIT_RATE,
    # SCENARIO: Scalability scenarios
    "2001": ORBITER_NODE_BIT_RATE, "2002": ORBITER_NODE_BIT_RATE, "2003": ORBITER_NODE_BIT_RATE,
    # SCENARIO: Common
    "2004": ORBITER_NODE_BIT_RATE, "2005": ORBITER_NODE_BIT_RATE, "2006": ORBITER_NODE_BIT_RATE, "2007": ORBITER_NODE_BIT_RATE, "2008": ORBITER_NODE_BIT_RATE,
    "2009": ORBITER_NODE_BIT_RATE, "2010": ORBITER_NODE_BIT_RATE, "2011": ORBITER_NODE_BIT_RATE, "2012": ORBITER_NODE_BIT_RATE, "2013": ORBITER_NODE_BIT_RATE, "2014": ORBITER_NODE_BIT_RATE, "2015": ORBITER_NODE_BIT_RATE, "2016": ORBITER_NODE_BIT_RATE,
    "2017": ORBITER_NODE_BIT_RATE, "2018": ORBITER_NODE_BIT_RATE, "2019": ORBITER_NODE_BIT_RATE, "2020": ORBITER_NODE_BIT_RATE, "2021": ORBITER_NODE_BIT_RATE, "2022": ORBITER_NODE_BIT_RATE, "2023": ORBITER_NODE_BIT_RATE, "2024": ORBITER_NODE_BIT_RATE,
    "2025": ORBITER_NODE_BIT_RATE, "2026": ORBITER_NODE_BIT_RATE, "2027": ORBITER_NODE_BIT_RATE, "2028": ORBITER_NODE_BIT_RATE, "2029": ORBITER_NODE_BIT_RATE, "2030": ORBITER_NODE_BIT_RATE, "2031": ORBITER_NODE_BIT_RATE, "2032": ORBITER_NODE_BIT_RATE,
    "2033": ORBITER_NODE_BIT_RATE, "2034": ORBITER_NODE_BIT_RATE, "2035": ORBITER_NODE_BIT_RATE, "2036": ORBITER_NODE_BIT_RATE, "2037": ORBITER_NODE_BIT_RATE, "2038": ORBITER_NODE_BIT_RATE, "2039": ORBITER_NODE_BIT_RATE, "2040": ORBITER_NODE_BIT_RATE,
    "2041": ORBITER_NODE_BIT_RATE, "2042": ORBITER_NODE_BIT_RATE, "2043": ORBITER_NODE_BIT_RATE, "2044": ORBITER_NODE_BIT_RATE, "2045": ORBITER_NODE_BIT_RATE, "2046": ORBITER_NODE_BIT_RATE, "2047": ORBITER_NODE_BIT_RATE, "2048": ORBITER_NODE_BIT_RATE,
    "2049": ORBITER_NODE_BIT_RATE, "2050": ORBITER_NODE_BIT_RATE, "2051": ORBITER_NODE_BIT_RATE, "2052": ORBITER_NODE_BIT_RATE, "2053": ORBITER_NODE_BIT_RATE, "2054": ORBITER_NODE_BIT_RATE, "2055": ORBITER_NODE_BIT_RATE, "2056": ORBITER_NODE_BIT_RATE,
    "2057": ORBITER_NODE_BIT_RATE, "2058": ORBITER_NODE_BIT_RATE, "2059": ORBITER_NODE_BIT_RATE, "2060": ORBITER_NODE_BIT_RATE, "2061": ORBITER_NODE_BIT_RATE, "2062": ORBITER_NODE_BIT_RATE, "2063": ORBITER_NODE_BIT_RATE, "2064": ORBITER_NODE_BIT_RATE,

    "9001": OGS_NODE_BIT_RATE, "9002": OGS_NODE_BIT_RATE, "9003": OGS_NODE_BIT_RATE,
    "9004": OGS_NODE_BIT_RATE, "9005": OGS_NODE_BIT_RATE, "9006": OGS_NODE_BIT_RATE,
    "9007": OGS_NODE_BIT_RATE, "9008": OGS_NODE_BIT_RATE, "9009": OGS_NODE_BIT_RATE,
    "9010": OGS_NODE_BIT_RATE, "9011": OGS_NODE_BIT_RATE, "9012": OGS_NODE_BIT_RATE,
}


def get_num_lasers(node_id: str):
    if node_id in SOURCE_NODES:
        return 1
    elif node_id in RELAY_NODES:
        # SCENARIO: Mars relay
        # return 1
        # SCENARIO: Earth relay
        return 2
    elif node_id in DESTINATION_NODES:
        return 1
        # return 2
    else:
        print(f"Node id not mapped {node_id}")
        return None
