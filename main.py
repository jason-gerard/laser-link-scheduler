from timeit import default_timer as timer
import traceback
import typer

import numpy as np

from src.time_expanded_graph.time_expanded_graph import (
    TimeExpandedGraph,
    convert_time_expanded_graph_to_contact_plan,
    write_time_expanded_graph,
)
from src.models.pointing_delay import retargeting_delay_cache
from src.reporting.report_generator import Reporter
from src.schedulers import (
    BaseScheduler,
    LaserLinkScheduler,
    LLSModel,
    PathSchedulerModel,
    RandomScheduler,
    AlternatingScheduler,
    FairContactPlan,
    LifespanAware,
)
from src.topology import weights
from src.topology.contact_plan import (
    IONContactPlanParser,
    IPNDContactPlanParser,
)
from src.utils import FileType

SCHEDULER: dict[str, BaseScheduler] = {
    "lls": LaserLinkScheduler(),
    "lls_pat_unaware": LaserLinkScheduler(should_bypass_retargeting_time=True),
    "lls_mip": LLSModel(is_mip=True),
    "lls_lp": LLSModel(is_mip=False),
    "lls_path": PathSchedulerModel(),
    "fcp": FairContactPlan(),
    "random": RandomScheduler(),
    "alternating": AlternatingScheduler(),
    "lifespan_aware": LifespanAware(),
}


def experiment_driver(
    experiment_name: str, scheduler_name: str, reporter: Reporter
):
    # Clear all caches
    weights.effective_contact_time_cache = {}
    weights.coordinate_cache = {}
    retargeting_delay_cache = {}

    start = timer()

    # Read contact plan from disk
    contact_plan_parser = IONContactPlanParser()
    contact_plan = contact_plan_parser.read(experiment_name)
    print("Finished reading contact plan")

    # Convert contact plan into a time expanded graph (TEG). From our testing on the Fair Contact Plan algorithm
    # benefits from graph fractionation.
    should_reduce = scheduler_name in ["lls_mip", "lls_lp"]
    time_expanded_graph = TimeExpandedGraph.from_contact_plan(
        contact_plan=contact_plan,
        should_fractionate=True,
        should_reduce=should_reduce,
    )
    write_time_expanded_graph(
        experiment_name, time_expanded_graph, FileType.TEG
    )
    print("Finished converting contact plan to time expanded graph")

    try:
        print("Starting contact scheduling")
        if scheduler_name not in SCHEDULER:
            raise ValueError(f"Unknown scheduler name: {scheduler_name}")

        scheduled_time_expanded_graph = SCHEDULER[scheduler_name].schedule(
            time_expanded_graph
        )

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
        for scheduler_name in scheduler_names:
            print(
                f"Starting execution of experiment: {experiment_name}, with scheduler: {scheduler_name}"
            )
            experiment_driver(experiment_name, scheduler_name, reporter)
            print("\n\n")

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
