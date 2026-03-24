from timeit import default_timer as timer
import copy
import traceback
import typer

import numpy as np

from src.time_expanded_graph.time_expanded_graph import (
    TimeExpandedGraph,
    convert_time_expanded_graph_to_contact_plan,
    write_time_expanded_graph,
)
from src.models.pointing_delay import RETARGETING_DELAY_CACHE
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
from src.topology.weights import EFFECTIVE_CONTACT_TIME_CACHE, COORDINATE_CACHE
from src.topology.contact_plan import (
    IONContactPlanParser,
    IPNDContactPlanParser,
)
from src.utils import FileType, RunTablePrinter, make_fanout_progress_callback

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
    experiment_name: str,
    scheduler_name: str,
    reporter: Reporter,
    time_expanded_graph: TimeExpandedGraph,
    run_table: RunTablePrinter,
) -> dict[str, str | float | int]:
    start = timer()
    contact_plan_parser = IONContactPlanParser()

    try:
        if scheduler_name not in SCHEDULER:
            raise ValueError(f"Unknown scheduler name: {scheduler_name}")

        progress_callback = run_table.make_progress_callback(
            experiment_name, scheduler_name
        )
        scheduler_input_teg = copy.deepcopy(time_expanded_graph)

        teg = (
            scheduler_input_teg.dag_reduction(progress_callback)
            if scheduler_name in ["lls_lp", "lls_mip"]
            else scheduler_input_teg
        )
        if scheduler_name not in ["lls_lp", "lls_mip"]:
            progress_callback("schedule", 0, 1)

        scheduled_time_expanded_graph = SCHEDULER[scheduler_name].schedule(
            teg, progress_callback
        )

        write_time_expanded_graph(
            experiment_name,
            scheduled_time_expanded_graph,
            FileType.TEG_SCHEDULED,
        )

        # Convert the TEG back to a contact plan
        scheduled_contact_plan = convert_time_expanded_graph_to_contact_plan(
            scheduled_time_expanded_graph,
            progress_callback,
        )
        contact_plan_parser.write(
            experiment_name,
            scheduled_contact_plan,
            FileType.CONTACT_PLAN_SCHEDULED,
        )

        # Write contact plan to disk as IPN-D contact plan, so we can visualize the output
        ipnd_contact_plan_parser = IPNDContactPlanParser()
        ipnd_contact_plan_parser.write(experiment_name, scheduled_contact_plan)

        progress_callback("report", 1, 1)
        run_data = reporter.generate_report(
            experiment_name,
            scheduler_name,
            timer() - start,
            scheduled_time_expanded_graph,
        )
        return run_data
    except Exception as e:
        traceback.print_exc()
        if scheduler_name == "lls_mip":
            raise e
        return {
            "progress": "failed",
            "duration": timer() - start,
            "network_capacity": 0,
            "network_wasted_capacity": 0,
            "wasted_buffer_capacity": 0,
            "jains_fairness_index": 0.0,
            "scheduled_delay": 0.0,
        }


def multi_experiment_driver(
    experiment_names: list[str],
    scheduler_names: list[str],
    run_table: RunTablePrinter,
):
    reporter = Reporter(write_pkl=True)

    for experiment_name in experiment_names:
        # Clear caches once before building the shared TEG for this experiment.
        EFFECTIVE_CONTACT_TIME_CACHE.clear()
        COORDINATE_CACHE.clear()
        RETARGETING_DELAY_CACHE.clear()

        contact_plan_parser = IONContactPlanParser()
        contact_plan = contact_plan_parser.read(experiment_name)

        build_callbacks = []
        # Build and write the input TEG.
        for scheduler_name in scheduler_names:
            run_table.update_progress(experiment_name, scheduler_name, "0%")
            build_callbacks.append(
                run_table.make_progress_callback(
                    experiment_name, scheduler_name
                )
            )

        build_progress_callback = make_fanout_progress_callback(
            build_callbacks
        )

        time_expanded_graph = TimeExpandedGraph.from_contact_plan(
            contact_plan=contact_plan,
            should_fractionate=True,
            progress_callback=build_progress_callback,
        )
        write_time_expanded_graph(
            experiment_name, time_expanded_graph, FileType.TEG
        )

        for scheduler_name in scheduler_names:
            # Reset caches before each scheduler run since scheduling/reporting uses global caches.
            EFFECTIVE_CONTACT_TIME_CACHE.clear()
            COORDINATE_CACHE.clear()
            RETARGETING_DELAY_CACHE.clear()

            run_table.mark_running(experiment_name, scheduler_name)
            run_data = experiment_driver(
                experiment_name,
                scheduler_name,
                reporter,
                time_expanded_graph,
                run_table,
            )
            run_table.update_result(experiment_name, scheduler_name, run_data)

    report_id = reporter.write_report()
    print(f"\nreport_id={report_id}")


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
    with RunTablePrinter(experiment_names, scheduler_names) as run_table:
        multi_experiment_driver(experiment_names, scheduler_names, run_table)


if __name__ == "__main__":
    app()
