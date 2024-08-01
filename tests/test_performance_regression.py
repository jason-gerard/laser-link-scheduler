import pickle

import numpy as np

from tests.testing_utils import scheduler_test_driver, get_regression_experiment_file, EXPERIMENT_NAMES
from time_expanded_graph import TimeExpandedGraph
from utils import FileType


class TestPerformanceRegression:
    
    def test_performance(self):
        """
        We want to verify that the topology has not changed and that the resulting capacities have also not changed.
        """
        for experiment_name in EXPERIMENT_NAMES:
            teg, scheduled_teg = scheduler_test_driver(experiment_name)

            teg_path = get_regression_experiment_file(experiment_name, FileType.TEG)
            with open(teg_path, "rb") as f:
                regression_teg = pickle.load(f)
                self.compare_teg(regression_teg, teg)

            scheduled_teg_path = get_regression_experiment_file(experiment_name, FileType.TEG_SCHEDULED)
            with open(scheduled_teg_path, "rb") as f:
                regression_scheduled_teg = pickle.load(f)
                self.compare_teg(regression_scheduled_teg, scheduled_teg)

    def compare_teg(self, teg1: TimeExpandedGraph, teg2: TimeExpandedGraph):
        assert np.array_equal(teg1.graphs, teg2.graphs)
        assert teg1.contacts == teg2.contacts
        assert np.array_equal(teg1.state_durations, teg2.state_durations)
        assert teg1.K == teg2.K
        assert teg1.N == teg2.N
        assert teg1.nodes == teg2.nodes
        assert teg1.node_map == teg2.node_map
        assert teg1.ipn_node_to_planet_map == teg2.ipn_node_to_planet_map

