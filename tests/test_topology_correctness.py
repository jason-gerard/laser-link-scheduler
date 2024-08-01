from tests.testing_utils import scheduler_test_driver


class TestTopologyCorrectness:
    
    EXPERIMENT_NAME = "mars_earth_test_scenario"
    teg, scheduled_teg = scheduler_test_driver(EXPERIMENT_NAME)
    
    def test_no_multiple_transmissions(self):
        """
        We want to verify that for each graph each node only transmits to a single other node. We can do this by
        making sure the sum of each row and column is equal to 0 or 1.
        """
        
        for k in range(self.scheduled_teg.K):
            for tx_idx in range(self.scheduled_teg.N):
                row_count = sum(self.scheduled_teg.graphs[k][tx_idx])
                assert row_count == 1 or row_count == 0

            for rx_idx in range(self.scheduled_teg.N):
                row_count = sum(self.scheduled_teg.graphs[k][:, rx_idx])
                assert row_count == 1 or row_count == 0
                
    def test_same_num_states(self):
        assert len(self.teg.graphs) == len(self.scheduled_teg.graphs)
