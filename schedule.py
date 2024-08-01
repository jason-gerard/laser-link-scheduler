import argparse
from timeit import default_timer as timer

from contact_plan import IONContactPlanParser
from lls import lls
from time_expanded_graph import build_time_expanded_graph, write_time_expanded_graph, \
    convert_time_expanded_graph_to_contact_plan, time_expanded_graph_splitter
from utils import FileType
from weights import compute_node_capacities, compute_capacity, compute_wasted_capacity


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--experiment_name', help='Name of experiment folder')
    return parser.parse_args()


def main(experiment_name):
    start = timer()

    # Read contact plan from disk
    contact_plan_parser = IONContactPlanParser()
    contact_plan = contact_plan_parser.read(experiment_name)
    print("Finished reading contact plan")

    # Convert contact plan into a time expanded graph (TEG)
    time_expanded_graph = build_time_expanded_graph(contact_plan)
    write_time_expanded_graph(experiment_name, time_expanded_graph, FileType.TEG)
    print("Finished converting contact plan to time expanded graph")

    # Split long contacts in the TEG into multiple smaller contacts, this will result in each k state having a maximum
    # duration of d_max
    split_time_expanded_graph = time_expanded_graph_splitter(time_expanded_graph)
    write_time_expanded_graph(experiment_name, split_time_expanded_graph, FileType.SPLIT)
    print("Finished splitting time expanded graph")

    # Iterate through each of the k graphs and compute the maximal matching
    # As we step through the k graphs we want to optimize for our metric
    # This will produce the TEG with the maximum Earth-bound network capacity
    print("Starting scheduling contacts")
    scheduled_time_expanded_graph = lls(split_time_expanded_graph)
    write_time_expanded_graph(experiment_name, scheduled_time_expanded_graph, FileType.TEG_SCHEDULED)
    print("Finished scheduling contacts")

    node_capacities = compute_node_capacities(
        scheduled_time_expanded_graph.graphs,
        scheduled_time_expanded_graph.state_durations,
        scheduled_time_expanded_graph.K,
        scheduled_time_expanded_graph.ipn_node_to_planet_map)
    network_capacity = compute_capacity(node_capacities)
    print(f"Scheduled network capacity: {network_capacity:,}")
    network_wasted_capacity = compute_wasted_capacity(node_capacities)
    print(f"Scheduled network wasted capacity: {network_wasted_capacity:,}")

    # Convert the TEG back to a contact plan
    scheduled_contact_plan = convert_time_expanded_graph_to_contact_plan(scheduled_time_expanded_graph)
    contact_plan_parser.write(experiment_name, scheduled_contact_plan, FileType.SCHEDULED)
    print("Finished converting time expanded graph to contact plan")

    end = timer()
    print(f"Elapsed time: {(end - start):.4f} seconds")


if __name__ == "__main__":
    args = get_args()
    main(args.experiment_name)
