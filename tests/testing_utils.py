from laser_link_scheduler.time_expanded_graph import TimeExpandedGraph
from laser_link_scheduler.schedulers import LaserLinkScheduler
from laser_link_scheduler.topology.contact_plan import IONContactPlanParser


def scheduler_test_driver(experiment_name):
    contact_plan_parser = IONContactPlanParser()
    contact_plan = contact_plan_parser.read(experiment_name)

    time_expanded_graph = TimeExpandedGraph.from_contact_plan(
        contact_plan, should_fractionate=False, should_reduce=False
    )

    split_time_expanded_graph = time_expanded_graph.fractionate_graph()

    return time_expanded_graph, LaserLinkScheduler().schedule(
        split_time_expanded_graph
    )
