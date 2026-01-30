from timeit import default_timer as timer
import traceback
import typer

import numpy as np

from laser_link_scheduler import constants
from laser_link_scheduler.graph.path_solver import PathSchedulerModel
from laser_link_scheduler.graph.time_expanded_graph import (
    convert_contact_plan_to_time_expanded_graph,
    convert_time_expanded_graph_to_contact_plan,
    write_time_expanded_graph,
)
from laser_link_scheduler.models import pointing_delay as pointing_delay_model
from laser_link_scheduler.reporting.report_generator import Reporter
from laser_link_scheduler.scheduling.milp_lls import LLSModel
from laser_link_scheduler.scheduling.scheduler import (
    AlternatingScheduler,
    FairContactPlan,
    LaserLinkScheduler,
    RandomScheduler,
)
from laser_link_scheduler.topology import weights
from laser_link_scheduler.topology.contact_plan import (
    IONContactPlanParser,
    IPNDContactPlanParser,
)
from laser_link_scheduler.utils import FileType


def experiment_driver(
    experiment_name: str, scheduler_name: str, reporter: Reporter
):
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
        contact_plan, should_fractionate=True, should_reduce=should_reduce
    )
    write_time_expanded_graph(
        experiment_name, time_expanded_graph, FileType.TEG
    )
    print("Finished converting contact plan to time expanded graph")

    try:
        print("Starting contact scheduling")
        if scheduler_name == "lls":
            scheduled_time_expanded_graph = LaserLinkScheduler().schedule(
                time_expanded_graph
            )
        elif scheduler_name == "lls_pat_unaware":
            constants.should_bypass_retargeting_time = True
            scheduled_time_expanded_graph = LaserLinkScheduler().schedule(
                time_expanded_graph
            )
            constants.should_bypass_retargeting_time = False
        elif scheduler_name == "lls_mip":
            scheduled_time_expanded_graph = LLSModel(
                time_expanded_graph, is_mip=True
            ).solve()
        elif scheduler_name == "lls_lp":
            scheduled_time_expanded_graph = LLSModel(
                time_expanded_graph, is_mip=False
            ).solve()
        elif scheduler_name == "lls_path":
            scheduled_time_expanded_graph = PathSchedulerModel(
                time_expanded_graph
            ).solve()
        elif scheduler_name == "fcp":
            scheduled_time_expanded_graph = FairContactPlan().schedule(
                time_expanded_graph
            )
        elif scheduler_name == "random":
            scheduled_time_expanded_graph = RandomScheduler().schedule(
                time_expanded_graph
            )
        elif scheduler_name == "alternating":
            scheduled_time_expanded_graph = AlternatingScheduler().schedule(
                time_expanded_graph
            )
        else:
            print(
                f"No scheduler selected, scheduler with name {scheduler_name} is unknown"
            )
            raise Exception("No scheduler selected")

        write_time_expanded_graph(
            experiment_name,
            scheduled_time_expanded_graph,
            FileType.TEG_SCHEDULED,
        )
        print("Finished contact scheduling")

        # Convert the TEG back to a contact plan
        scheduled_contact_plan = convert_time_expanded_graph_to_contact_plan(
            scheduled_time_expanded_graph
        )
        contact_plan_parser.write(
            experiment_name,
            scheduled_contact_plan,
            FileType.CONTACT_PLAN_SCHEDULED,
        )
        print("Finished converting time expanded graph to contact plan")

        # Write contact plan to disk as IPN-D contact plan, so we can visualize the output
        ipnd_contact_plan_parser = IPNDContactPlanParser()
        ipnd_contact_plan_parser.write(experiment_name, scheduled_contact_plan)

        reporter.generate_report(
            experiment_name,
            scheduler_name,
            timer() - start,
            scheduled_time_expanded_graph,
        )
    except Exception as e:
        print(
            f"Execution of experiment: {experiment_name}, with scheduler: {scheduler_name} failed from {e}"
        )
        traceback.print_exc()
        if scheduler_name == "lls_mip":
            raise e


def multi_experiment_driver(
    experiment_names: list[str], scheduler_names: list[str]
):
    reporter = Reporter(write_pkl=True)

    for experiment_name in experiment_names:
        try:
            for scheduler_name in scheduler_names:
                print(
                    f"Starting execution of experiment: {experiment_name}, with scheduler: {scheduler_name}"
                )
                experiment_driver(experiment_name, scheduler_name, reporter)
                print("\n\n")
        except Exception:
            break

    reporter.write_report()


app = typer.Typer()


@app.command()
def main(
    experiment_names: list[str] = typer.Option(
        ...,
        "--experiment-names",
        "-e",
        help="Name of experiment folder",
    ),
    scheduler_names: list[str] = typer.Option(
        ...,
        "--scheduler-names",
        "-s",
        help="Name of scheduler algorithm to use",
    ),
):
    np.random.seed(42)
    multi_experiment_driver(experiment_names, scheduler_names)


if __name__ == "__main__":
    app()
