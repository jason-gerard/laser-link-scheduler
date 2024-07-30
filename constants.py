SOURCES_ROOT = "experiments"

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
default_a = 1

# The matrix A, contains the integer IDs of the communication interfaces for each node. Each laser
# communication interface is associated in an integer ID, a, where a >= 1.
# For nodes 0, 1, 2, 3
# A = [[1], [1, 3], [1, 1, 1, 1], [2, 2]]

# The list I, contains the of communication interfaces each node has.
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
