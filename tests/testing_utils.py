from contact_plan import IONContactPlanParser
from scheduler import LaserLinkScheduler
from time_expanded_graph import convert_contact_plan_to_time_expanded_graph, fractionate_graph


def scheduler_test_driver(experiment_name):
    contact_plan_parser = IONContactPlanParser()
    contact_plan = contact_plan_parser.read(experiment_name)

    time_expanded_graph = convert_contact_plan_to_time_expanded_graph(contact_plan)

    split_time_expanded_graph = fractionate_graph(time_expanded_graph)

    return time_expanded_graph, LaserLinkScheduler().schedule(split_time_expanded_graph)
