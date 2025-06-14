import argparse
import traceback
from timeit import default_timer as timer

import numpy as np

from LLS_milp import LLSModel
from path_solver import PathSchedulerModel
from contact_plan import IONContactPlanParser, IPNDContactPlanParser
from report_generator import Reporter
from scheduler import LaserLinkScheduler, FairContactPlan, RandomScheduler, AlternatingScheduler
from time_expanded_graph import convert_contact_plan_to_time_expanded_graph, write_time_expanded_graph, \
    convert_time_expanded_graph_to_contact_plan
from utils import FileType
import constants
import weights
import pointing_delay_model


def experiment_driver(experiment_name: str, scheduler_name: str, reporter: Reporter):
    # Clear all caches
    weights.effective_contact_time_cache = {}
    weights.coordinate_cache = {}
    pointing_delay_model.retargeting_delay_cache = {}

    start = timer()

    # Read contact plan from disk
    contact_plan_parser = IONContactPlanParser()
    contact_plan = contact_plan_parser.read(experiment_name)
    print("Finished reading contact plan")

    # Convert contact plan into a time expanded graph (TEG). From our testing on the Fair Contact Plan algorithm
    # benefits from graph fractionation.
    should_reduce = scheduler_name == "lls_mip" or scheduler_name == "lls_lp"
    time_expanded_graph = convert_contact_plan_to_time_expanded_graph(
        contact_plan,
        should_fractionate=True,
        should_reduce=should_reduce
    )
    write_time_expanded_graph(experiment_name, time_expanded_graph, FileType.TEG)
    print("Finished converting contact plan to time expanded graph")

    try:
        print("Starting contact scheduling")
        if scheduler_name == "lls":
            scheduled_time_expanded_graph = LaserLinkScheduler().schedule(time_expanded_graph)
        elif scheduler_name == "lls_pat_unaware":
            constants.should_bypass_retargeting_time = True
            scheduled_time_expanded_graph = LaserLinkScheduler().schedule(time_expanded_graph)
            constants.should_bypass_retargeting_time = False
        elif scheduler_name == "lls_mip":
            scheduled_time_expanded_graph = LLSModel(time_expanded_graph, is_mip=True).solve()
        elif scheduler_name == "lls_lp":
            scheduled_time_expanded_graph = LLSModel(time_expanded_graph, is_mip=False).solve()
        elif scheduler_name == "lls_path":
            scheduled_time_expanded_graph = PathSchedulerModel(time_expanded_graph).solve()
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

        reporter.generate_report(
            experiment_name,
            scheduler_name,
            timer() - start,
            scheduled_time_expanded_graph)
    except Exception as e:
        print(f"Execution of experiment: {experiment_name}, with scheduler: {scheduler_name} failed from {e}")
        traceback.print_exc()
        if scheduler_name == "lls_mip":
            raise e


def multi_experiment_driver(experiment_names: list[str], scheduler_names: list[str]):
    reporter = Reporter(write_pkl=True)

    for experiment_name in experiment_names:
        try:
            for scheduler_name in scheduler_names:
                print(f"Starting execution of experiment: {experiment_name}, with scheduler: {scheduler_name}")
                experiment_driver(experiment_name, scheduler_name, reporter)
                print("\n\n")
        except Exception as e:
            break

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
