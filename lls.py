import argparse
import pprint
from timeit import default_timer as timer

from contact_plan import IONContactPlanParser, contact_plan_splitter
from metrics import compute_node_capacities, compute_capacity, compute_wasted_capacity
from time_expanded_graph import build_time_expanded_graph, write_time_expanded_graph, \
    TimeExpandedGraph, convert_time_expanded_graph_to_contact_plan
from utils import FileType


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--experiment_name', help='Contact plan file input name ')
    return parser.parse_args()


def main(experiment_name):
    """
    Assumptions: A key assumption we make is that contacts are bidirectional i.e. if there is a contact from A -> B
    then there is also a contact from B -> A. These contacts can also exist at the same time. Since we are dealing
    with lasers this might not be a valid assumption but since our data is unidirectional i.e. we are only
    transmitting data from Mars -> Earth, it doesn't matter anyway. This assumption should be revisited in the future
    as the scenarios become more complex. This data is embedded in the contact plan and not the code, so the code should
    already be written to support unidirectional contacts but logically, for now, all contacts are bidirectional.
    """
    
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


def lls(time_expanded_graph) -> TimeExpandedGraph:
    # Inputs: contact topology [P] of size K x N x N
    #         state durations [T] of size K
    # Outputs: contact plan [L] of size K x N x N
    # Initialize matrix [C] containing the weights, [C]_i,j <- 0 for all i,j
    # for k <- 0 to K do
    #   [W]_k,i,j <- [C]_i,j for all i,j
    #   Blossom([P]_k, [L]_k, [W]_k)
    #   if [L]_k,i,j = 0 then
    #     [C]_i,j <- [C]_i,j + t_k for all i,j
    
    # TODO: we need to modify this algorithm such that after each iteration of blossom we update the matrix of weights
    # with the node capacity values
    
    capacities = compute_node_capacities(time_expanded_graph)
    pprint.pprint(capacities)

    network_capacity = compute_capacity(capacities)
    wasted_network_capacity = compute_wasted_capacity(capacities)
    print("cap", network_capacity)
    print("wasted cap", wasted_network_capacity)

    return time_expanded_graph


if __name__ == "__main__":
    args = get_args()
    main(args.experiment_name)
