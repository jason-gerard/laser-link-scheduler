import argparse
from timeit import default_timer as timer

import numpy as np

from contact_plan import IONContactPlanParser, IPNDContactPlanParser
from report_generator import Reporter
from scheduler import LaserLinkScheduler, FairContactPlan, RandomScheduler, AlternatingScheduler
from time_expanded_graph import convert_contact_plan_to_time_expanded_graph, write_time_expanded_graph, \
    convert_time_expanded_graph_to_contact_plan
from utils import FileType
import constants


def experiment_driver(experiment_name: str, scheduler_name: str, reporter: Reporter):
    constants.should_bypass_retargeting_time = False
    start = timer()

    # Read contact plan from disk
    contact_plan_parser = IONContactPlanParser()
    contact_plan = contact_plan_parser.read(experiment_name)
    print("Finished reading contact plan")

    # Convert contact plan into a time expanded graph (TEG). From our testing on the Fair Contact Plan algorithm
    # benefits from graph fractionation.
    time_expanded_graph = convert_contact_plan_to_time_expanded_graph(
        contact_plan,
        should_fractionate=True)
    write_time_expanded_graph(experiment_name, time_expanded_graph, FileType.TEG)
    print("Finished converting contact plan to time expanded graph")

    print("Starting contact scheduling")
    if scheduler_name == "lls":
        scheduled_time_expanded_graph = LaserLinkScheduler().schedule(time_expanded_graph)
    elif scheduler_name == "lls_pat_unaware":
        constants.should_bypass_retargeting_time = True
        scheduled_time_expanded_graph = LaserLinkScheduler().schedule(time_expanded_graph)
        constants.should_bypass_retargeting_time = False
    elif scheduler_name == "fcp":
        scheduled_time_expanded_graph = FairContactPlan().schedule(time_expanded_graph)
    elif scheduler_name == "random":
        scheduled_time_expanded_graph = RandomScheduler().schedule(time_expanded_graph)
    elif scheduler_name == "alternating":
        scheduled_time_expanded_graph = AlternatingScheduler().schedule(time_expanded_graph)
    else:
        print(f"No scheduler selected, scheduler with name {scheduler_name} is unknown")
        raise Exception("No scheduler selected")

    write_time_expanded_graph(experiment_name, scheduled_time_expanded_graph, FileType.TEG_SCHEDULED)
    print("Finished contact scheduling")

    # Convert the TEG back to a contact plan
    scheduled_contact_plan = convert_time_expanded_graph_to_contact_plan(scheduled_time_expanded_graph)
    contact_plan_parser.write(experiment_name, scheduled_contact_plan, FileType.SCHEDULED)
    print("Finished converting time expanded graph to contact plan")
    
    # Write contact plan to disk as IPN-D contact plan, so we can visualize the output
    ipnd_contact_plan_parser = IPNDContactPlanParser()
    ipnd_contact_plan_parser.write(experiment_name, scheduled_contact_plan)

    constants.should_bypass_retargeting_time = False
    reporter.generate_report(
        experiment_name,
        scheduler_name,
        timer() - start,
        scheduled_time_expanded_graph)


def multi_experiment_driver(experiment_names: list[str], scheduler_names: list[str]):
    reporter = Reporter(write_pkl=False)

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
