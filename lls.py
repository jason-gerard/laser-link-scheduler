import networkx as nx
import argparse
import pprint
from timeit import default_timer as timer

from contact_plan import IONContactPlanParser, contact_plan_splitter
from metrics import compute_node_capacities, compute_capacity, compute_wasted_capacity
from time_expanded_graph import build_time_expanded_graph, write_time_expanded_graph, \
    TimeExpandedGraph, convert_time_expanded_graph_to_contact_plan, Graph
from utils import FileType


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--experiment_name', help='Contact plan file input name ')
    return parser.parse_args()


def main(experiment_name):
    start = timer()

    # Read contact plan from disk
    contact_plan_parser = IONContactPlanParser()
    contact_plan = contact_plan_parser.read(experiment_name)
    print("Finished reading contact plan")

    # Split long contacts into multiple smaller contacts, a maximum duration of 100s
    split_contact_plan = contact_plan_splitter(contact_plan)
    contact_plan_parser.write(experiment_name, split_contact_plan, FileType.SPLIT)
    print("Finished splitting contact plan")

    # Convert split contact plan into a time expanded graph (TEG)
    time_expanded_graph = build_time_expanded_graph(split_contact_plan)
    write_time_expanded_graph(experiment_name, time_expanded_graph, FileType.TEG)
    print("Finished converting contact plan to time expanded graph")
    
    # Iterate through each of the k graphs and compute the maximal matching
    # As we step through the k graphs we want to optimize for our metric
    # This will produce the TEG with the maximum Earth-bound network capacity
    scheduled_time_expanded_graph = lls(time_expanded_graph)
    write_time_expanded_graph(experiment_name, scheduled_time_expanded_graph, FileType.SCHEDULED_TEG)
    print("Finished scheduling contacts")

    # Convert the TEG back to a contact plan
    scheduled_contact_plan = convert_time_expanded_graph_to_contact_plan(scheduled_time_expanded_graph)
    contact_plan_parser.write(experiment_name, scheduled_contact_plan, FileType.SCHEDULED)
    print("Finished converting time expanded graph to contact plan")
    
    end = timer()
    print(f"Elapsed time: {end - start} seconds")


def lls(time_expanded_graph: TimeExpandedGraph) -> TimeExpandedGraph:
    """
    This algorithm is a max-weight maximal matching

    The weights of the matrix W_k should be calculated such that the weight of each edge should correspond to the
    delta capacity or delta wasted capacity if that edge was selected plus the sum of time that contact was
    previously disabled for

    delta_cap -> The increased network capacity (max weight maximal matching) or wasted capacity (min weight maximal
    matching) if the edge i,j is selected for step k.
    delta_time -> The disabled contact time for an edge i,j, or the sum of the duration that edge i,j could have been
    enabled but was not due to previous contact plan selections.
    
    Inputs: contact topology [P] of size K x N x N
            IPN node mappings [X] of size N
            state durations [T] of size K
    Outputs: contact plan [L] of size K x N x N
    
    for k <- 0 to K do
      [W]_k,i,j <- delta_capacity([P]_k, [L], [X])
                   + alpha * delta_time([L], [T]) for all i,j
      Blossom([P]_k, [L]_k, [W]_k)
    """
    
    L = []
    for graph in time_expanded_graph.graphs:
        # Compute weights based on the current contact topology, P_k, the total contact plan up to this point, L,
        # and the list of interplanetary nodes, X.
        W_k = []
        # Create list of edges, represented by three-tuple of (tx_idx, rx_idx, weight) based on the contact topology P_k
        # and computed weights based on delta_capacity + alpha * delta_time
        edges = []
        
        # Create graph containing edges from P_k
        G = nx.Graph()
        G.add_weighted_edges_from(edges)
        matched_edges = nx.max_weight_matching(G)
        
        # Compute L_k from the matched edges
        included_contacts = []
        adj_matrix = []
        L_k = Graph(
            contacts=included_contacts,
            adj_matrix=adj_matrix,
            k=graph.k,
            state_duration=graph.state_duration,
            state_start_time=graph.state_start_time,
        )
        L.append(L_k)
        
    return TimeExpandedGraph(
        graphs=L,
        nodes=time_expanded_graph.nodes,
        interplanetary_nodes=time_expanded_graph.interplanetary_nodes,
        ipn_node_to_planet_map=time_expanded_graph.ipn_node_to_planet_map,
        node_map=time_expanded_graph.node_map,
        start_time=time_expanded_graph.start_time,
        end_time=time_expanded_graph.end_time,
    )
        
    # capacities = compute_node_capacities(time_expanded_graph)
    # pprint.pprint(capacities)
    # 
    # network_capacity = compute_capacity(capacities)
    # wasted_network_capacity = compute_wasted_capacity(capacities)
    # print("cap", network_capacity)
    # print("wasted cap", wasted_network_capacity)


def fair_contact_plan(time_expanded_graph: TimeExpandedGraph) -> TimeExpandedGraph:
    """
    Max-weight maximal matching
    Inputs: contact topology [P] of size K x N x N
            state durations [T] of size K
    Outputs: contact plan [L] of size K x N x N

    DCT_i,j <- 0 for all i,j
    for k <- 0 to K do
      [W]_k,i,j <- DCT_i,j for all i,j
      Blossom([P]_k, [L]_k, [W]_k)
      if [L]_k,i,j = 0 then
        DCT_i,j <- DCT_i,j + [T]_k for all i,j
    """
    pass


if __name__ == "__main__":
    args = get_args()
    main(args.experiment_name)
