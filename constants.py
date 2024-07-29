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

# The maximum number of simultaneous interfaces. For this work interfaces corresponds to the number of lasers onboard
# the satellite. In future work if we want to model satellites with multiple lasers, this value can be adjusted. If we
# want to model satellites with varying number of lasers this should be dynamically read by the input file for each
# satellite.
I = 1
