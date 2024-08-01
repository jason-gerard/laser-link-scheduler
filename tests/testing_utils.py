import os

from contact_plan import IONContactPlanParser
from lls import lls
from time_expanded_graph import build_time_expanded_graph, time_expanded_graph_splitter
from utils import FileType

EXPERIMENT_NAMES = [
    "mars_earth_test_scenario",
    "mars_earth_simple_scenario",
]


def get_regression_experiment_file(experiment_name, file_type: FileType) -> str:
    file_suffix = file_type.value
    file_name = f"{experiment_name}_{file_suffix}_pickle.pkl"
    return str(os.path.join("tests", "regression_experiments", experiment_name, file_name))


def scheduler_test_driver(experiment_name):
    contact_plan_parser = IONContactPlanParser()
    contact_plan = contact_plan_parser.read(experiment_name)

    time_expanded_graph = build_time_expanded_graph(contact_plan)

    split_time_expanded_graph = time_expanded_graph_splitter(time_expanded_graph)

    return time_expanded_graph, lls(split_time_expanded_graph)
