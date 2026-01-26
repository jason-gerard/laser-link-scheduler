import os


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SOURCES_ROOT = os.path.join(
    REPO_ROOT, "scenarios", "experiments", "experiments"
)
REPORTS_ROOT = os.path.join(REPO_ROOT, "output", "reports")

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
alpha = 0.99

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

should_bypass_retargeting_time = False

# The interplanetary range we define is any contact over 100,000 kilometers. We then convert this to light seconds to
# follow the ION contact plan standard
INTERPLANETARY_RANGE = 100_000 * 1_000 / 299_792_458

EARTH = "EARTH"
MARS = "MARS"

# GS
DESTINATION_NODES = ["9001", "9002", "9003"]

# Mars Science sats
SOURCE_NODES = [
    "2001",
    "2002",
    "2003",
    "2004",
    "2005",
    "2006",
    "2007",
    "2008",
    "2009",
    "2010",
    "2011",
    "2012",
    "2013",
    "2014",
    "2015",
    "2016",
    "2017",
    "2018",
    "2019",
    "2020",
    "2021",
    "2022",
    "2023",
    "2024",
    "2025",
    "2026",
    "2027",
    "2028",
    "2029",
    "2030",
    "2031",
    "2032",
    "2033",
    "2034",
    "2035",
    "2036",
    "2037",
    "2038",
    "2039",
    "2040",
    "2041",
    "2042",
    "2043",
    "2044",
    "2045",
    "2046",
    "2047",
    "2048",
    "2049",
    "2050",
    "2051",
    "2052",
    "2053",
    "2054",
    "2055",
    "2056",
    "2057",
    "2058",
    "2059",
    "2060",
    "2061",
    "2062",
    "2063",
    "2064",
]

# Earth satellites
RELAY_NODES = ["1001", "1002", "1003", "1004", "1005", "1006"]

NODE_TO_PLANET_MAP = {
    "1001": EARTH,
    "1002": EARTH,
    "1003": EARTH,
    "1004": EARTH,
    "1005": EARTH,
    "1006": EARTH,
    "2001": MARS,
    "2002": MARS,
    "2003": MARS,
    "2004": MARS,
    "2005": MARS,
    "2006": MARS,
    "2007": MARS,
    "2008": MARS,
    "2009": MARS,
    "2010": MARS,
    "2011": MARS,
    "2012": MARS,
    "2013": MARS,
    "2014": MARS,
    "2015": MARS,
    "2016": MARS,
    "2017": MARS,
    "2018": MARS,
    "2019": MARS,
    "2020": MARS,
    "2021": MARS,
    "2022": MARS,
    "2023": MARS,
    "2024": MARS,
    "2025": MARS,
    "2026": MARS,
    "2027": MARS,
    "2028": MARS,
    "2029": MARS,
    "2030": MARS,
    "2031": MARS,
    "2032": MARS,
    "2033": MARS,
    "2034": MARS,
    "2035": MARS,
    "2036": MARS,
    "2037": MARS,
    "2038": MARS,
    "2039": MARS,
    "2040": MARS,
    "2041": MARS,
    "2042": MARS,
    "2043": MARS,
    "2044": MARS,
    "2045": MARS,
    "2046": MARS,
    "2047": MARS,
    "2048": MARS,
    "2049": MARS,
    "2050": MARS,
    "2051": MARS,
    "2052": MARS,
    "2053": MARS,
    "2054": MARS,
    "2055": MARS,
    "2056": MARS,
    "2057": MARS,
    "2058": MARS,
    "2059": MARS,
    "2060": MARS,
    "2061": MARS,
    "2062": MARS,
    "2063": MARS,
    "2064": MARS,
    "9001": EARTH,
    "9002": EARTH,
    "9003": EARTH,
}

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


# DESTINATION_NODES = [
#     "1001", "1002", "1003", "1004", "1005", "1006", "1007", "1008",
# ]
#
# SOURCE_NODES = [
#     "2001", "2002", "2003", "2004", "2005", "2006", "2007", "2008", "2009", "2010", "2011", "2012",
#     "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024",
#     "2025", "2026", "2027", "2028", "2029", "2030", "2031", "2032", "2033", "2034", "2035", "2036",
#     "2037", "2038", "2039", "2040", "2041", "2042", "2043", "2044", "2045", "2046", "2047", "2048",
#     "2049", "2050", "2051", "2052", "2053", "2054", "2055", "2056", "2057", "2058", "2059", "2060",
#     "2061", "2062", "2063", "2064", "2065", "2066", "2067", "2068", "2069", "2070", "2071", "2072",
#     "2073", "2074", "2075", "2076", "2077", "2078", "2079", "2080", "2081", "2082", "2083", "2084",
#     "2085", "2086", "2087", "2088", "2089", "2090", "2091", "2092", "2093", "2094", "2095", "2096",
#     "2097", "2098", "2099", "2100", "2101", "2102", "2103", "2104", "2105", "2106", "2107", "2108",
#     "2109", "2110", "2111", "2112", "2113", "2114", "2115", "2116", "2117", "2118", "2119", "2120",
#     "2121", "2122", "2123", "2124", "2125", "2126", "2127", "2128", "2129", "2130", "2131", "2132",
#     "2133", "2134", "2135", "2136", "2137", "2138", "2139", "2140", "2141", "2142", "2143", "2144",
#     "2145", "2146", "2147", "2148", "2149", "2150", "2151", "2152", "2153", "2154", "2155", "2156",
#     "2157", "2158", "2159", "2160", "2161", "2162", "2163", "2164", "2165", "2166", "2167", "2168",
#     "2169", "2170", "2171", "2172", "2173", "2174", "2175", "2176", "2177", "2178", "2179", "2180",
#     "2181", "2182", "2183", "2184", "2185", "2186", "2187", "2188", "2189", "2190", "2191", "2192",
# ]
#
# RELAY_NODES = [
#     "3001", "3002", "3003", "3004", "3005", "3006", "3007", "3008", "3009", "3010",
#     "3011", "3012", "3013", "3014", "3015", "3016",
# ]
#
# NODE_TO_PLANET_MAP = {
#     "1001": EARTH, "1002": EARTH, "1003": EARTH, "1004": EARTH, "1005": EARTH, "1006": EARTH, "1007": EARTH, "1008": EARTH,
#
#     "2001": MARS, "2002": MARS, "2003": MARS, "2004": MARS, "2005": MARS, "2006": MARS, "2007": MARS, "2008": MARS,
#     "2009": MARS, "2010": MARS, "2011": MARS, "2012": MARS, "2013": MARS, "2014": MARS, "2015": MARS, "2016": MARS,
#     "2017": MARS, "2018": MARS, "2019": MARS, "2020": MARS, "2021": MARS, "2022": MARS, "2023": MARS, "2024": MARS,
#     "2025": MARS, "2026": MARS, "2027": MARS, "2028": MARS, "2029": MARS, "2030": MARS, "2031": MARS, "2032": MARS,
#     "2033": MARS, "2034": MARS, "2035": MARS, "2036": MARS, "2037": MARS, "2038": MARS, "2039": MARS, "2040": MARS,
#     "2041": MARS, "2042": MARS, "2043": MARS, "2044": MARS, "2045": MARS, "2046": MARS, "2047": MARS, "2048": MARS,
#     "2049": MARS, "2050": MARS, "2051": MARS, "2052": MARS, "2053": MARS, "2054": MARS, "2055": MARS, "2056": MARS,
#     "2057": MARS, "2058": MARS, "2059": MARS, "2060": MARS, "2061": MARS, "2062": MARS, "2063": MARS, "2064": MARS,
#     "2065": MARS, "2066": MARS, "2067": MARS, "2068": MARS, "2069": MARS, "2070": MARS, "2071": MARS, "2072": MARS,
#     "2073": MARS, "2074": MARS, "2075": MARS, "2076": MARS, "2077": MARS, "2078": MARS, "2079": MARS, "2080": MARS,
#     "2081": MARS, "2082": MARS, "2083": MARS, "2084": MARS, "2085": MARS, "2086": MARS, "2087": MARS, "2088": MARS,
#     "2089": MARS, "2090": MARS, "2091": MARS, "2092": MARS, "2093": MARS, "2094": MARS, "2095": MARS, "2096": MARS,
#     "2097": MARS, "2098": MARS, "2099": MARS, "2100": MARS, "2101": MARS, "2102": MARS, "2103": MARS, "2104": MARS,
#     "2105": MARS, "2106": MARS, "2107": MARS, "2108": MARS, "2109": MARS, "2110": MARS, "2111": MARS, "2112": MARS,
#     "2113": MARS, "2114": MARS, "2115": MARS, "2116": MARS, "2117": MARS, "2118": MARS, "2119": MARS, "2120": MARS,
#     "2121": MARS, "2122": MARS, "2123": MARS, "2124": MARS, "2125": MARS, "2126": MARS, "2127": MARS, "2128": MARS,
#     "2129": MARS, "2130": MARS, "2131": MARS, "2132": MARS, "2133": MARS, "2134": MARS, "2135": MARS, "2136": MARS,
#     "2137": MARS, "2138": MARS, "2139": MARS, "2140": MARS, "2141": MARS, "2142": MARS, "2143": MARS, "2144": MARS,
#     "2145": MARS, "2146": MARS, "2147": MARS, "2148": MARS, "2149": MARS, "2150": MARS, "2151": MARS, "2152": MARS,
#     "2153": MARS, "2154": MARS, "2155": MARS, "2156": MARS, "2157": MARS, "2158": MARS, "2159": MARS, "2160": MARS,
#     "2161": MARS, "2162": MARS, "2163": MARS, "2164": MARS, "2165": MARS, "2166": MARS, "2167": MARS, "2168": MARS,
#     "2169": MARS, "2170": MARS, "2171": MARS, "2172": MARS, "2173": MARS, "2174": MARS, "2175": MARS, "2176": MARS,
#     "2177": MARS, "2178": MARS, "2179": MARS, "2180": MARS, "2181": MARS, "2182": MARS, "2183": MARS, "2184": MARS,
#     "2185": MARS, "2186": MARS, "2187": MARS, "2188": MARS, "2189": MARS, "2190": MARS, "2191": MARS, "2192": MARS,
#
#     "3001": MARS, "3002": MARS, "3003": MARS, "3004": MARS, "3005": MARS, "3006": MARS, "3007": MARS, "3008": MARS,
#     "3009": MARS, "3010": MARS, "3011": MARS, "3012": MARS, "3013": MARS, "3014": MARS, "3015": MARS, "3016": MARS,
# }
