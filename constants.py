SOURCES_ROOT = "experiments"
REPORTS_ROOT = "reports"

# Minimum duration an edge i,j in state k can have
# If there is a contact in the contact topology, P, that is shorter than this, since we can't extend the duration we
# will filter it out. This contact most likely is not long enough to establish the link and send a meaningful amount
# of data.
d_min = 5
# Maximum duration an edge i,j in state k can have
# This value decides the maximum duration of a state k. If a contact spans longer than t_max it will be split into
# multiple contacts of duration t_max, which then translates into multiple k states. There is a tradeoff with this
# variable between the overhead of setting up the link and being able to transition to contacts with different nodes
# to improve the fairness.
d_max = 100

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
alpha = 0.1

# The matrix A, contains the integer IDs of the communication interfaces for each node. Each laser
# communication interface is associated in an integer ID, a, where a >= 1.
# For nodes 0, 1, 2, 3
# A = [[1], [1, 3], [1, 1, 1, 1], [2, 2]]

# The list I, contains the number of communication interfaces each node has.
# For nodes 0, 1, 2, 3
# I = [1, 2, 4, 2]

# For now, we are assuming a constant and symmetric bitrate across all links at 100 mbps.
default_BIT_RATE = 100
# This list B, contains the bit rates for each communication interface.
# Since a = 0 doesn't apply to any interface the bit_rate is just set to 0.
B = [0, default_BIT_RATE]

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
