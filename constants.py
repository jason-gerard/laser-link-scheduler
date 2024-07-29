SOURCES_ROOT = "experiments"

# Minimum duration an edge i,j in state k can have
# If there is a contact in the contact topology, P, that is shorter than this, since we can't extend the duration we
# will filter it out. This contact most likely is not long enough to establish the link and send a meaningful amount
# of data.
t_min = 5
# Maximum duration an edge i,j in state k can have
# This value decides the maximum duration of a state k. If a contact spans longer than t_max it will be split into
# multiple contacts of duration t_max, which then translates into multiple k states. There is a tradeoff with this
# variable between the overhead of setting up the link and being able to transition to contacts with different nodes
# to improve the fairness.
t_max = 100

# This list A, contains the integer IDs of all the different available communication interfaces. Each laser
# communication interface is associated in an integer ID, a, where a >= 1.
A = [1]
# The default communication interface to use.
default_a = A[0]
