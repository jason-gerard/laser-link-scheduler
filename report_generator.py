import csv
import os
import time
import pickle

import constants
from time_expanded_graph import TimeExpandedGraph
from utils import FileType
from weights import compute_node_capacities, compute_capacity, compute_wasted_capacity


class Reporter:
    
    def __init__(self, debug: bool):
        self.reports = []
        self.time_expanded_graph_data = []
        self.debug = debug
    
    def generate_report(self, experiment_name: str, scheduler_name: str, duration: float, teg: TimeExpandedGraph):
        self.time_expanded_graph_data.append({
            "name": f"{scheduler_name}_{experiment_name}",
            "teg": teg,
        })
        
        node_capacities = compute_node_capacities(teg.graphs, teg.state_durations, teg.K, teg.ipn_node_to_planet_map)
        network_capacity = compute_capacity(node_capacities)
        network_wasted_capacity = compute_wasted_capacity(node_capacities)

        row = [scheduler_name, experiment_name, duration, network_capacity, network_wasted_capacity]
        self.reports.append(row)

        if self.debug:
            print(f"Scheduled network capacity: {network_capacity:,}")
            print(f"Scheduled network wasted capacity: {network_wasted_capacity:,}")
            print(f"Elapsed time: {duration:.4f} seconds")

    def write_report(self):
        # Create CSV of runtimes, capacity, and wasted capacity
        basic_report_headers = ["Algorithm", "Scenario", "Duration", "Capacity", "Wasted capacity"]

        report_id = int(time.time())

        report_dir = os.path.join(constants.REPORTS_ROOT, f"{report_id}")
        os.mkdir(report_dir)
        
        report_path = str(os.path.join(report_dir, f"{report_id}_{FileType.REPORT.value}.csv"))
        with open(report_path, "w") as f:
            writer = csv.writer(f, delimiter=",")
            writer.writerow(basic_report_headers)
            writer.writerows(self.reports)
        
        # Save each teg as a pkl file to the reports directory so that it can be used for analysis later
        for teg_data_dict in self.time_expanded_graph_data:
            pkl_path = str(os.path.join(report_dir, f"{teg_data_dict['name']}.pkl"))
            with open(pkl_path, "wb") as f:
                pickle.dump(teg_data_dict["teg"], f)
