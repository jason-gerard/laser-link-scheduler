from dataclasses import dataclass

from time_expanded_graph import TimeExpandedGraph


@dataclass
class NodeCapacity:
    id: int
    # This is the amount of data that the node can receive from local nodes i.e. nodes on the same planet
    capacity_in: float
    # This is the amount of data that the node can transmit to nodes on other planets
    capacity_out: float


def compute_node_capacities(time_expanded_graph: TimeExpandedGraph) -> list[NodeCapacity]:
    # For each graph in the TEG, compute the capacity
    node_capacities_by_graph = [
        compute_node_capacity_by_graph(
            graph.adj_matrix,
            graph.state_duration,
            time_expanded_graph.interplanetary_nodes,
            time_expanded_graph.ipn_node_to_planet_map)
        for graph
        in time_expanded_graph.graphs]

    # Convert to dict of node to list of capacities
    node_capacities_dict = {node_id: [] for node_id in time_expanded_graph.interplanetary_nodes}
    for graph_capacities in node_capacities_by_graph:
        for node_capacity in graph_capacities:
            node_capacities_dict[node_capacity.id].append(node_capacity)

    # Merge node capacities calculated over all graphs in the TEG to a single capacity, this gives a list of node
    # capacities where each is the total capacity in and out over all graphs for a single IPN node
    return [merge_node_capacities(node_capacities) for node_capacities in node_capacities_dict.values()]


def merge_node_capacities(capacities: list[NodeCapacity]) -> NodeCapacity:
    # Sum the capacity in and capacity out for the IPN node
    total_capacity_in = sum([capacity.capacity_in for capacity in capacities])
    total_capacity_out = sum([capacity.capacity_out for capacity in capacities])

    # We will assume all the capacities in the list are for the same node
    return NodeCapacity(
        id=capacities[0].id,
        capacity_in=total_capacity_in,
        capacity_out=total_capacity_out,
    )


def compute_node_capacity_by_graph(
        adj_matrix: list[list[int]],
        duration: int,
        interplanetary_nodes: list[int],
        ipn_node_to_planet_map: dict[int, str]
) -> list[NodeCapacity]:
    # For now, we are assuming a constant and symmetric bitrate across all links at 100 mbps.
    BIT_RATE = 100

    capacities = []
    for ipn_node_idx in interplanetary_nodes:
        node_capacity = NodeCapacity(
            id=ipn_node_idx,
            capacity_in=0,
            capacity_out=0)

        # The list if ipn_nodes contains the node idx i.e. its index in the adjacency matrix
        for tx_idx in range(len(adj_matrix)):
            for rx_idx in range(len(adj_matrix[tx_idx])):
                # If there was no contact between those two nodes then skip
                if adj_matrix[tx_idx][rx_idx] == 0:
                    continue

                # Only one of these two conditions can ever be true since we don't count contacts with the same node as
                # the tx and rx
                if tx_idx not in interplanetary_nodes and rx_idx == ipn_node_idx:
                    # Compute the amount of data transmitted to the IPN node from a non-IPN node.
                    # ipn node == rx_node
                    node_capacity.capacity_in += duration * BIT_RATE
                elif (tx_idx == ipn_node_idx
                      and rx_idx in interplanetary_nodes
                      and ipn_node_to_planet_map[tx_idx] != ipn_node_to_planet_map[rx_idx]):
                    # Compute the amount of data transmitted by the IPN node to an IPN node that is orbiting the
                    # destination planet.
                    # ipn node == tx_node
                    node_capacity.capacity_out += duration * BIT_RATE

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
