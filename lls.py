import argparse
import pprint
from dataclasses import dataclass

from contact_plan import IONContactPlanParser
from time_expanded_graph import build_time_expanded_graph, write_time_expanded_graph, \
    TimeExpandedGraph


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file_name', help='Contact plan file input name ')
    return parser.parse_args()


def main(file_name):
    """
    Assumptions: A key assumption we make is that contacts are bidirectional i.e. if there is a contact from A -> B
    then there is also a contact from B -> A. These contacts can also exist at the same time. Since we are dealing
    with lasers this might not be a valid assumption but since our data is unidirectional i.e. we are only
    transmitting data from Mars -> Earth, it doesn't matter anyway. This assumption should be revisited in the future
    as the scenarios become more complex. This data is embedded in the contact plan and not the code, so the code should
    already be written to support unidirectional contacts but logically, for now, all contacts are bidirectional.
    """

    # Read contact plan from disk
    contact_plan_parser = IONContactPlanParser()
    contact_plan = contact_plan_parser.read(file_name)

    # Split long contacts into multiple smaller contacts, a maximum duration of 100s
    # Write split contact plan back to disk
    # TODO

    # Convert split contact plan into a time expanded graph (TEG)
    time_expanded_graph = build_time_expanded_graph(contact_plan)
    write_time_expanded_graph(time_expanded_graph, file_name)

    # Iterate through each of the k graphs and compute the maximal matching
    # As we step through the k graphs we want to optimize for our metric
    # This will produce the TEG with the maximum Earth-bound network capacity
    capacities = compute_node_capacities(time_expanded_graph)
    pprint.pprint(capacities)

    # I initially thought this might be needed but leaving the capacities or filtering them out shouldn't do anything
    # because they cannot be changed. Might as well leave them to not create confusion.
    # We will filter out capacities with a capacity_in equal to 0 since those are nodes that don't generate data or have
    # any satellites orbiting the same planet transmitting data to it. An example of this is a relay satellite orbiting
    # Earth that doesn't transmit any command data and only receives data. We don't have to include this in our
    # calculation. If that satellite started sending some amount of data then it would be included.
    # capacities = [capacity for capacity in capacities if capacity.capacity_in > 0]

    network_capacity = compute_capacity(capacities)
    wasted_network_capacity = compute_wasted_capacity(capacities)
    print("cap", network_capacity)
    print("wasted cap", wasted_network_capacity)

    # Convert the TEG back to a contact plan
    # TODO

    # Write scheduled contact plan to disk
    contact_plan_parser.write(file_name, contact_plan)


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


if __name__ == "__main__":
    args = get_args()
    main(args.file_name)
