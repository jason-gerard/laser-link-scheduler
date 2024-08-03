import argparse
import numpy as np
from timeit import default_timer as timer

from contact_plan import IONContactPlanParser
from report_generator import Reporter
from scheduler import LaserLinkScheduler, FairContactPlan, RandomScheduler
from time_expanded_graph import convert_contact_plan_to_time_expanded_graph, write_time_expanded_graph, \
    convert_time_expanded_graph_to_contact_plan, graph_fractionation
from utils import FileType


def experiment_driver(experiment_name: str, scheduler_name: str, reporter: Reporter):
    start = timer()

    # Read contact plan from disk
    contact_plan_parser = IONContactPlanParser()
    contact_plan = contact_plan_parser.read(experiment_name)
    print("Finished reading contact plan")

    # Convert contact plan into a time expanded graph (TEG)
    time_expanded_graph = convert_contact_plan_to_time_expanded_graph(contact_plan)
    write_time_expanded_graph(experiment_name, time_expanded_graph, FileType.TEG)
    print("Finished converting contact plan to time expanded graph")

    # Split long contacts in the TEG into multiple smaller contacts, this will result in each k state having a maximum
    # duration of d_max. This process is referred to as fractionation.
    fractionated_time_expanded_graph = graph_fractionation(time_expanded_graph)
    write_time_expanded_graph(experiment_name, fractionated_time_expanded_graph, FileType.SPLIT)
    print("Finished splitting time expanded graph")

    print("Starting scheduling contacts")
    if scheduler_name == "lls":
        # Iterate through each of the k graphs and compute the maximal matching
        # As we step through the k graphs we want to optimize for our metric
        # This will produce the TEG with the maximum Earth-bound network capacity
        scheduled_time_expanded_graph = LaserLinkScheduler().schedule(fractionated_time_expanded_graph)
    elif scheduler_name == "fcp":
        scheduled_time_expanded_graph = FairContactPlan().schedule(fractionated_time_expanded_graph)
    elif scheduler_name == "random":
        scheduled_time_expanded_graph = RandomScheduler().schedule(fractionated_time_expanded_graph)
    else:
        print(f"No scheduler selected, scheduler with name {scheduler_name} is unknown")
        scheduled_time_expanded_graph = fractionated_time_expanded_graph

    write_time_expanded_graph(experiment_name, scheduled_time_expanded_graph, FileType.TEG_SCHEDULED)
    print("Finished scheduling contacts")

    # Convert the TEG back to a contact plan
    scheduled_contact_plan = convert_time_expanded_graph_to_contact_plan(scheduled_time_expanded_graph)
    contact_plan_parser.write(experiment_name, scheduled_contact_plan, FileType.SCHEDULED)
    print("Finished converting time expanded graph to contact plan")

    reporter.generate_report(
        experiment_name,
        scheduler_name,
        timer() - start,
        scheduled_time_expanded_graph)


def multi_experiment_driver(experiment_names: list[str], scheduler_names: list[str]):
    reporter = Reporter(debug=True)

    for experiment_name in experiment_names:
        for scheduler_name in scheduler_names:
            print(f"Starting execution of experiment: {experiment_name}, with scheduler: {scheduler_name}")
            experiment_driver(experiment_name, scheduler_name, reporter)
            print("\n\n")

    reporter.write_report()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment_names', help='Name of experiment folder', nargs="+")
    parser.add_argument('-s', '--scheduler_names', help='Name of scheduler algorithm to use', nargs="+")
    return parser.parse_args()


if __name__ == "__main__":
    np.random.seed(42)
    args = get_args()
    multi_experiment_driver(args.experiment_names, args.scheduler_names)
