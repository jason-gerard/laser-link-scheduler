import numpy as np
import constants
from pat_delay_model import pat_delay
from dataclasses import dataclass


@dataclass
class NodeCapacity:
    id: int
    # This is the amount of data that the node can receive from local nodes i.e. nodes on the same planet
    capacity_in: float
    # This is the amount of data that the node can transmit to nodes on other planets
    capacity_out: float


def delta_capacity(
        contact_topology_k: np.ndarray,
        scheduled_contact_topology: np.ndarray,
        node_capacities: list[NodeCapacity],
        nodes: list[str],
        state_duration: int,
        positions: np.ndarray,
) -> np.ndarray:
    """
    The delta cap method can be used with two different sub routines. Each of the sub routines will be applied to the
    current contact topology and applied to a new graph where each edge is selected in isolation.The choice of sub
    routine can either be the increase in capacity which is used for max weight maximal matching or it can compute the
    increase in wasted capacity for min weight maximal matching.
    """
    num_nodes = len(contact_topology_k)

    # Compute network capacity with current node_capacities list
    current_capacity = compute_capacity(node_capacities)

    delta_capacities = np.zeros((num_nodes, num_nodes), dtype='int64')
    for tx_idx in range(num_nodes):
        for rx_idx in range(num_nodes):
            # The id of the optical communication interface that the edge uses for the contact. An id of 0 corresponds
            # to no contact
            interface_id: int = contact_topology_k[tx_idx][rx_idx]
            # For each edge in topology_k that is active i.e. graph[i][j] >= 1, create a new graph where only that edge
            # is active and compute the capacity if that edge was selected
            if interface_id >= 1:
                # Compute the node capacity from the single edge graph
                single_edge_node_capacity = compute_node_capacity_by_single_edge_graph(
                    tx_idx,
                    rx_idx,
                    interface_id,
                    state_duration,
                    nodes,
                    scheduled_contact_topology,
                    positions)

                # Since the delta caps matrix is already filled with zeros, if the single edge node capacity returns
                # None i.e. there was no new capacity then we can just leave the delta as 0, otherwise compute
                # the new total capacity then the new delta
                if single_edge_node_capacity is not None:
                    # Merge it with the node_capacities list and compute the network capacity with the new list
                    new_node_capacities = merge_many_node_capacities(node_capacities + [single_edge_node_capacity])
                    new_capacity = compute_capacity(new_node_capacities)

                    # Take the difference and that is the new weight
                    delta_capacities[tx_idx][rx_idx] = new_capacity - current_capacity

    return delta_capacities


def disabled_contact_time(contact_topology_k: np.ndarray, contact_plan_k: np.ndarray,
                          state_duration: int) -> np.ndarray:
    """
    Compute the disabled contact time for each node. This is calculated by taking the state duration and adding it to
    the index in the matrix if the edge was active in the topology but inactive in the plan. The disabled contact time
    for an edge i,j, or the sum of the duration that edge i,j could have been enabled but was not due to previous
    contact plan selections.
    """

    num_nodes = len(contact_topology_k)
    disabled_contact_times = np.zeros((num_nodes, num_nodes), dtype='int64')

    for tx_idx in range(num_nodes):
        for rx_idx in range(num_nodes):
            # There are two ways to decide to increment the disabled contact time.
            #
            # The first method would be to increment the DCT if the link (edge) existed in the contact topology but was
            # not selected in the contact plan
            # if contact_topology_k[tx_idx][rx_idx] >= 1 and contact_plan_k[tx_idx][rx_idx] == 0:
            #
            # The second method is to increment any contact that was not in the contact plan regardless if it was a
            # feasible contact in the original contact topology.
            # if contact_plan_k[tx_idx][rx_idx] == 0:
            #
            # After some testing it seems that the second method produces the same or higher capacity for both LLS and
            # FCP due to a higher prioritization of links which are scarce in the initial contact topology. This way
            # when they do show up in the contact topology there is a much higher chance for it to be selected. Since
            # these links are often intra-constellation or interplanetary links, it improves the capacity of the network
            if contact_plan_k[tx_idx][rx_idx] == 0:
                # Increment the disabled contact time by the state duration i.e. the amount of time it was turned
                # off
                disabled_contact_times[tx_idx][rx_idx] = state_duration

    return disabled_contact_times


def merge_many_node_capacities(capacities: list[NodeCapacity]) -> list[NodeCapacity]:
    """
    Merges capacities for many different node ids
    """
    # Convert to dict of node to list of capacities
    ipn_node_ids = set([capacity.id for capacity in capacities])
    node_capacities_dict = {node_id: [] for node_id in ipn_node_ids}
    for node_capacity in capacities:
        node_capacities_dict[node_capacity.id].append(node_capacity)

    # Merge node capacities calculated over all graphs in the TEG to a single capacity, this gives a list of node
    # capacities where each is the total capacity in and out over all graphs for a single IPN node
    return [merge_node_capacities(node_capacities, node_idx) for node_idx, node_capacities in
            node_capacities_dict.items()]


def merge_node_capacities(capacities: list[NodeCapacity], node_idx: int) -> NodeCapacity:
    """
    Merges capacities for a single node id
    """
    # Sum the capacity in and capacity out for the IPN node
    total_capacity_in = sum([capacity.capacity_in for capacity in capacities])
    total_capacity_out = sum([capacity.capacity_out for capacity in capacities])

    # We will assume all the capacities in the list are for the same node
    return NodeCapacity(
        id=node_idx,
        capacity_in=total_capacity_in,
        capacity_out=total_capacity_out,
    )


def compute_node_capacities(
        graphs: np.ndarray,
        state_durations: np.ndarray,
        K: int,
        nodes: list[str],
        scheduled_contact_topology: np.ndarray,
        positions: np.ndarray,
) -> list[NodeCapacity]:
    # For each graph in the TEG, compute the capacity
    node_capacities_by_graph = [
        compute_node_capacity_by_graph(graphs[k], state_durations[k], nodes, scheduled_contact_topology[:k], positions) for k in range(K)]

    node_capacities = np.array(node_capacities_by_graph).flatten().tolist()
    return merge_many_node_capacities(node_capacities)


def compute_node_capacity_by_single_edge_graph(
        tx_idx: int,
        rx_idx: int,
        interface_id: int,
        duration: int,
        nodes: list[str],
        scheduled_contact_topology: np.ndarray,
        positions: np.ndarray,
) -> NodeCapacity | None:
    bit_rate = constants.R[interface_id]
    
    effective_contact_duration = compute_effective_contact_time(
        tx_idx,
        rx_idx,
        scheduled_contact_topology,
        duration,
        positions,
    )

    # Only one of these two conditions can ever be true since we don't count contacts with the same node as
    # the tx and rx
    # Inflow for single hop and two hop
    if nodes[tx_idx] in constants.SOURCE_NODES and nodes[rx_idx] in constants.RELAY_NODES or nodes[rx_idx] in constants.DESTINATION_NODES:
        return NodeCapacity(
            id=rx_idx,
            capacity_in=effective_contact_duration * bit_rate,
            capacity_out=0 if nodes[rx_idx] in constants.RELAY_NODES else float("inf"))
    # Outflow for two hop
    elif (nodes[tx_idx] in constants.RELAY_NODES
          and nodes[rx_idx] in constants.DESTINATION_NODES):
        return NodeCapacity(
            id=tx_idx,
            capacity_in=0,
            capacity_out=effective_contact_duration * bit_rate)
    else:
        return None


def compute_node_capacity_by_graph(
        graph: np.ndarray,
        duration: int,
        nodes: list[str],
        scheduled_contact_topology: np.ndarray,
        positions: np.ndarray,
) -> list[NodeCapacity]:
    num_nodes = len(graph)

    capacities = []
    for node_idx, node in enumerate(nodes):
        if node not in constants.RELAY_NODES and node not in constants.DESTINATION_NODES:
            continue
        
        node_capacity = NodeCapacity(
            id=node_idx,
            capacity_in=0,
            capacity_out=0 if node in constants.RELAY_NODES else float("inf"))

        # The list if ipn_nodes contains the node idx i.e. its index in the adjacency matrix
        for tx_idx in range(num_nodes):
            for rx_idx in range(num_nodes):
                # If there was no contact between those two nodes then skip
                if graph[tx_idx][rx_idx] == 0:
                    continue

                effective_contact_duration = compute_effective_contact_time(
                    tx_idx,
                    rx_idx,
                    scheduled_contact_topology,
                    duration,
                    positions,
                )
                
                # Get the bit_rate from the communication interface ID
                a = graph[tx_idx][rx_idx]
                bit_rate = constants.R[a]

                # Only one of these two conditions can ever be true since we don't count contacts with the same node as
                # the tx and rx
                if nodes[tx_idx] in constants.SOURCE_NODES and rx_idx == node_idx:
                    # Compute the amount of data transmitted to the IPN node from a non-IPN node.
                    # ipn node == rx_node
                    node_capacity.capacity_in += effective_contact_duration * bit_rate
                elif (tx_idx == node_idx
                      and nodes[rx_idx] in constants.DESTINATION_NODES):
                    # Compute the amount of data transmitted by the IPN node to an IPN node that is orbiting the
                    # destination planet.
                    # ipn node == tx_node
                    node_capacity.capacity_out += effective_contact_duration * bit_rate

        capacities.append(node_capacity)

    # Return capacity in and capacity out for each ipn node
    return capacities


def compute_capacity(capacities: list[NodeCapacity]) -> float:
    # Take the min between the capacity in and capacity out for each IPN node
    max_capacities = [min(capacity.capacity_in, capacity.capacity_out) for capacity in capacities]

    # This metric should be maximized
    # Sum the capacities for each IPN nodes to get the network capacity
    return sum(max_capacities)


def compute_wasted_capacity(capacities: list[NodeCapacity]) -> float:
    # Take the absolute difference between the capacity in and capacity out for each IPN node
    wasted_capacities = [abs(capacity.capacity_in - capacity.capacity_out) for capacity in capacities if capacity.capacity_out != float("inf")]

    # This metric should be minimized
    # Sum the wasted capacities for each IPN nodes to get the network wasted capacity
    return sum(wasted_capacities)


def compute_wasted_buffer(capacities: list[NodeCapacity]) -> float:
    # To compute the wasted buffer we first compute the difference between how much data was scheduled as inflow and
    # outflow. Excess outflow data does not count as wasted buffer capacity. The main difference between wasted buffer
    # and wasted network capacity is that wasted network capacity includes excess inflow and outflow in the computation
    # where buffer only includes excess inflow.
    wasted_capacities = [max(capacity.capacity_in - capacity.capacity_out, 0) for capacity in capacities]

    return sum(wasted_capacities)


def compute_jains_fairness_index(
        graphs: np.ndarray,
        state_durations: np.ndarray,
        nodes: list[str],
        K: int,
        N: int
) -> float:
    """
    Compute the fairness based on the scheduled duration a contact is enabled and the bit rate of the laser, which gives
    the amount of data scheduled to be transferred over that period, this is the resource we are evaluating the fairness
    of. This fairness evaluation is specifically looking at orbiter to relay transmissions and does not include orbiter
    to orbiter or relay to relay transmissions. This metrics tries to understand of fair/even the distribution of access
    opportunities is between the science satellites (orbiters) and the relay satellites.
    """
    source_nodes = [node_idx for node_idx in range(N) if nodes[node_idx] in constants.SOURCE_NODES]  # Mars orbiter nodes
    # Create a single graph the sums the enabled contact times of all k states
    enabled_contact_times = np.zeros((K, N, N), dtype="int64")
    for k in range(K):
        for tx_idx in range(N):
            for rx_idx in range(N):
                if (graphs[k][tx_idx][rx_idx] >= 1
                        and tx_idx in source_nodes
                        and (nodes[rx_idx] in constants.RELAY_NODES or nodes[rx_idx] in constants.DESTINATION_NODES)):
                    interface_id = graphs[k][tx_idx][rx_idx]
                    bit_rate = constants.R[interface_id]

                    enabled_contact_times[k][tx_idx][rx_idx] = state_durations[k] * bit_rate

    enabled_contact_time_graph = np.sum(enabled_contact_times, axis=0)
    # Compute a list of the amount of data each Mars orbiter transmitted to a Mars relay
    enabled_contact_time_by_node = np.array([np.sum(enabled_contact_time_graph[node_idx]) for node_idx in source_nodes])

    # Solve for the network level Jain's fairness index with x as the throughput of the Mars orbiter nodes
    return (np.sum(enabled_contact_time_by_node) ** 2) / (len(source_nodes) * np.sum(enabled_contact_time_by_node ** 2))


def compute_scheduled_delay(
        graphs: np.ndarray,
        state_durations: np.ndarray,
        nodes: list[str],
        K: int,
        N: int
) -> float:
    """
    Compute the average delay between when an orbiter can transmit to a relay. This does not include orbiter to orbiter
    or relay to relay transmissions. This tries to understand the delay for Mars orbiter to Mars relay transmissions.
    """
    orbiter_nodes = [node_idx for node_idx in range(N) if nodes[node_idx] in constants.SOURCE_NODES]
    non_source_nodes = [node_idx for node_idx in range(N) if nodes[node_idx] in constants.RELAY_NODES or nodes[node_idx] in constants.DESTINATION_NODES]
    node_delays = {node_idx: {"total_delay": 0, "num_contacts": 0} for node_idx in orbiter_nodes}

    # This list keeps track of the delays each orbiter node has between contact opportunities with the relay satellites.
    node_contact_delays = np.zeros(N, dtype="int64")
    # For each orbiter compute the average delay between relay contact times
    for k in range(K):
        for tx_idx in orbiter_nodes:
            is_in_contact = len([rx_idx for rx_idx in non_source_nodes if graphs[k][tx_idx][rx_idx] >= 1]) > 0
            if is_in_contact and (node_contact_delays[tx_idx] > 0 or k == 0):
                node_delays[tx_idx]["total_delay"] += node_contact_delays[tx_idx]
                node_delays[tx_idx]["num_contacts"] += 1
                node_contact_delays[tx_idx] = 0
            elif not is_in_contact:
                node_contact_delays[tx_idx] += state_durations[k]
    
    # Since we increment the delay when a contact opportunity occurs, for the nodes that are not in communication with
    # an orbiter at the end of the contact plan this will add those delays to the running total.
    for tx_idx, delay in enumerate(node_contact_delays):
        if delay > 0:
            node_delays[tx_idx]["total_delay"] += node_contact_delays[tx_idx]

    # Take the average delay of all orbiters
    avg_delay_by_node = [node_delay["total_delay"] / max(node_delay["num_contacts"], 1)
                         for node_delay in node_delays.values()]
    return sum(avg_delay_by_node) / len(avg_delay_by_node)


# L1 cache effective contact time for same nodes idx1, idx2, k
effective_contact_time_cache = {}

# L2 cache coordinates for a specific node
coordinate_cache = {}

def compute_effective_contact_time(
    idx1: int,
    idx2: int,
    scheduled_contact_topology: np.ndarray,
    state_duration: int,
    positions: np.ndarray,
) -> float:
    if constants.should_bypass_retargeting_time:
        return state_duration
    
    curr_k = min(len(scheduled_contact_topology), len(positions) - 1)
    
    if ((idx1, idx2, curr_k) in effective_contact_time_cache or (idx2, idx1, curr_k) in effective_contact_time_cache) and curr_k != len(positions) - 1:
        return effective_contact_time_cache[(idx1, idx2, curr_k)]
    
    # For node1 check in the scheduled topology the last time it had a contact and with which node
    # Take that k and check the coordinates of it and the rx at that time, this will give the previous coordinates
    idx1_rx = -1
    if (idx1, curr_k) in coordinate_cache:
        idx1_rx = coordinate_cache[(idx1, curr_k)]
    else:
        for k, _ in reversed(list(enumerate(scheduled_contact_topology))):
            for rx_idx in range(len(scheduled_contact_topology[k])):
                if scheduled_contact_topology[k][idx1][rx_idx] >= 1:
                    idx1_rx = rx_idx
                    coordinate_cache[(idx1, curr_k)] = rx_idx
                    break

    # Do the same for node 2
    idx2_rx = -1
    if (idx2, curr_k) in coordinate_cache:
        idx2_rx = coordinate_cache[(idx2, curr_k)]
    else:
        for k, _ in reversed(list(enumerate(scheduled_contact_topology))):
            for rx_idx in range(len(scheduled_contact_topology[k])):
                if scheduled_contact_topology[k][idx2][rx_idx] >= 1:
                    idx2_rx = rx_idx
                    coordinate_cache[(idx2, curr_k)] = rx_idx
                    break

    # For node1 check in the scheduled topology the last time it had a contact and with which node
    # Take that k and check the coordinates of it and the rx at that time, this will give the previous coordinates
    # idx1_rx = -1
    # for k, _ in reversed(list(enumerate(scheduled_contact_topology))):
    #     for rx_idx in range(len(scheduled_contact_topology[k])):
    #         if scheduled_contact_topology[k][idx1][rx_idx] >= 1:
    #             idx1_rx = rx_idx
    #             break

    # Do the same for node 2
    # idx2_rx = -1
    # for k, _ in reversed(list(enumerate(scheduled_contact_topology))):
    #     for rx_idx in range(len(scheduled_contact_topology[k])):
    #         if scheduled_contact_topology[k][idx2][rx_idx] >= 1:
    #             idx2_rx = rx_idx
    #             break

    # Use PAT lib to compute delay
    if idx1_rx != -1 and idx2_rx != -1:
        idx1_coords = np.array(positions[curr_k][idx1])
        idx1_rx_coords = np.array(positions[curr_k][idx1_rx])

        idx2_coords = np.array(positions[curr_k][idx2])
        idx2_rx_coords = np.array(positions[curr_k][idx2_rx])
        
        PAT_delay = pat_delay(
            np.array([idx1_coords, idx1_rx_coords, idx2_coords]),
            np.array([idx2_coords, idx2_rx_coords, idx1_coords]),
        )
    elif idx1_rx != -1 and idx2_rx == -1:
        # idx2 first contact
        idx1_coords = np.array(positions[curr_k][idx1])
        idx1_rx_coords = np.array(positions[curr_k][idx1_rx])

        idx2_coords = np.array(positions[curr_k][idx2])

        PAT_delay = pat_delay(
            np.array([idx1_coords, idx1_rx_coords, idx2_coords]),
            np.array([idx1_coords, idx1_rx_coords, idx2_coords]),
        )
    elif idx1_rx == -1 and idx2_rx != -1:
        # idx1 first contact
        idx2_coords = np.array(positions[curr_k][idx2])
        idx2_rx_coords = np.array(positions[curr_k][idx2_rx])

        idx1_coords = np.array(positions[curr_k][idx1])

        PAT_delay = pat_delay(
            np.array([idx2_coords, idx2_rx_coords, idx1_coords]),
            np.array([idx2_coords, idx2_rx_coords, idx1_coords]),
        )
    else:
        PAT_delay = 0

    # Subtract with state duration, bind it to a floor of 0, this will give
    # effective contact duration = contact duration - PAT_delay
    effective_contact_time = max(state_duration - PAT_delay, 0)
    
    effective_contact_time_cache[(idx1, idx2, curr_k)] = effective_contact_time
    effective_contact_time_cache[(idx2, idx1, curr_k)] = effective_contact_time

    return effective_contact_time
