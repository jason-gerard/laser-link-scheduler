import numpy as np


def pat_delay_single_node(src_node, curr_dst_node, new_dst_node) -> float:
    # Parameters are coordinates with respect to some central point
    # so convert everything to have the src node as the central point
    # by subtracting the src node coordinates from all the nodes
    # coordinates which will create two 3d distance vectors for the current beam and the next beam
    rel_curr = curr_dst_node - src_node
    rel_new = new_dst_node - src_node

    # Compute the angle, theta, between the two vectors
    v1_u = rel_curr / np.linalg.norm(rel_curr)
    v2_u = rel_new / np.linalg.norm(rel_new)
    theta = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))  # radians

    # Based on the slew rate, angular velocity, of the CPA compute the time it
    # takes to make that rotation
    SLEW_RATE = 0.0349066  # radians or 2 deg/s
    return theta / SLEW_RATE


def pat_delay(node_set1, node_set2) -> float:
    # Compute the PAT delay for node sets 1 and 2
    pat_delay_1 = pat_delay_single_node(*node_set1)
    pat_delay_2 = pat_delay_single_node(*node_set2)

    # The max between them is the actual delay since both must be finished pointing
    # before starting acquisition
    return max(pat_delay_1, pat_delay_2)


if __name__ == "__main__":
    # dst12 is src2
    # src1 is dst22
    node_set_1 = np.array([
        np.array([1, 0, 0]),  # src1
        np.array([1, 1, 1]),  # dst11
        np.array([0, 2, 1]),  # dst12
    ])
    node_set_2 = np.array([
        np.array([0, 2, 1]),  # src2
        np.array([2, 0, 0]),  # dst21
        np.array([1, 0, 0]),  # dst22
    ])
    delay = pat_delay(node_set_1, node_set_2)
    print(delay)
