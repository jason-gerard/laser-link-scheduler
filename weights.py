import numpy as np
import constants
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
        node_capacities: list[NodeCapacity],
        ipn_node_to_planet_map: dict[int, str],
        state_duration: int
) -> np.ndarray:
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
                    ipn_node_to_planet_map)
                
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


def delta_time(contact_topology_k: np.ndarray, contact_plan_k: np.ndarray, state_duration: int) -> np.ndarray:
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
            # If the contact in the contact topology was >= 1, then there was a possible contact, but if in the
            # contact plan it is equal to 0 it means it was not enabled for the kth state.
            if contact_topology_k[tx_idx][rx_idx] >= 1 and contact_plan_k[tx_idx][rx_idx] == 0:
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
    return [merge_node_capacities(node_capacities, node_idx) for node_idx, node_capacities in node_capacities_dict.items()]


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


def compute_node_capacities(graphs: np.ndarray, state_durations: np.ndarray, K: int, ipn_node_to_planet_map: dict[int, str]) -> list[NodeCapacity]:
    # For each graph in the TEG, compute the capacity
    node_capacities_by_graph = [
        compute_node_capacity_by_graph(graphs[k], state_durations[k], ipn_node_to_planet_map) for k in range(K)]
    
    node_capacities = np.array(node_capacities_by_graph).flatten().tolist()
    return merge_many_node_capacities(node_capacities)


def compute_node_capacity_by_single_edge_graph(
        tx_idx: int,
        rx_idx: int,
        interface_id: int,
        duration: int,
        ipn_node_to_planet_map: dict[int, str]
) -> NodeCapacity | None:
    bit_rate = constants.B[interface_id]

    # Only one of these two conditions can ever be true since we don't count contacts with the same node as
    # the tx and rx
    if tx_idx not in ipn_node_to_planet_map.keys() and rx_idx in ipn_node_to_planet_map.keys():
        # Capacity in is defined for the rx IPN node by the amount of data transmitted to the IPN node from a non-IPN
        # node
        return NodeCapacity(
            id=rx_idx,
            capacity_in=duration * bit_rate,
            capacity_out=0)
    elif (tx_idx in ipn_node_to_planet_map.keys()
          and rx_idx in ipn_node_to_planet_map.keys()
          and ipn_node_to_planet_map[tx_idx] != ipn_node_to_planet_map[rx_idx]):
        # Capacity out is defined for the tx IPN node by the amount of data transmitted by the tx IPN node to an rx IPN
        # node where the tx and rx nodes are orbiting different central bodies
        return NodeCapacity(
            id=tx_idx,
            capacity_in=0,
            capacity_out=duration * bit_rate)
    else:
        return None


def compute_node_capacity_by_graph(
        graph: np.ndarray,
        duration: int,
        ipn_node_to_planet_map: dict[int, str]
) -> list[NodeCapacity]:
    num_nodes = len(graph)
    
    capacities = []
    for ipn_node_idx in ipn_node_to_planet_map.keys():
        node_capacity = NodeCapacity(
            id=ipn_node_idx,
            capacity_in=0,
            capacity_out=0)

        # The list if ipn_nodes contains the node idx i.e. its index in the adjacency matrix
        for tx_idx in range(num_nodes):
            for rx_idx in range(num_nodes):
                # If there was no contact between those two nodes then skip
                if graph[tx_idx][rx_idx] == 0:
                    continue

                # Get the bit_rate from the communication interface ID
                a = graph[tx_idx][rx_idx]
                bit_rate = constants.B[a]

                # Only one of these two conditions can ever be true since we don't count contacts with the same node as
                # the tx and rx
                if tx_idx not in ipn_node_to_planet_map.keys() and rx_idx == ipn_node_idx:
                    # Compute the amount of data transmitted to the IPN node from a non-IPN node.
                    # ipn node == rx_node
                    node_capacity.capacity_in += duration * bit_rate
                elif (tx_idx == ipn_node_idx
                      and rx_idx in ipn_node_to_planet_map.keys()
                      and ipn_node_to_planet_map[tx_idx] != ipn_node_to_planet_map[rx_idx]):
                    # Compute the amount of data transmitted by the IPN node to an IPN node that is orbiting the
                    # destination planet.
                    # ipn node == tx_node
                    node_capacity.capacity_out += duration * bit_rate

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
    wasted_capacities = [abs(capacity.capacity_in - capacity.capacity_out) for capacity in capacities]

    # This metric should be minimized
    # Sum the wasted capacities for each IPN nodes to get the network wasted capacity
    return sum(wasted_capacities)
