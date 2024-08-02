from contact_plan import IONContactPlanParser
from scheduler import LaserLinkScheduler
from time_expanded_graph import build_time_expanded_graph, time_expanded_graph_splitter


def scheduler_test_driver(experiment_name):
    contact_plan_parser = IONContactPlanParser()
    contact_plan = contact_plan_parser.read(experiment_name)

    time_expanded_graph = build_time_expanded_graph(contact_plan)

    split_time_expanded_graph = time_expanded_graph_splitter(time_expanded_graph)

    return time_expanded_graph, LaserLinkScheduler().schedule(split_time_expanded_graph)
