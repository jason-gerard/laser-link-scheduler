import csv
import os
import time
import pickle

import constants
from time_expanded_graph import TimeExpandedGraph
from utils import FileType
import weights
from weights import (
    compute_node_capacities,
    compute_capacity,
    compute_wasted_capacity,
    compute_jains_fairness_index,
    compute_scheduled_delay,
    compute_wasted_buffer,
)


class Reporter:
    def __init__(self, write_pkl: bool):
        self.reports = []
        self.time_expanded_graph_data = []
        self.write_pkl = write_pkl

    def generate_report(
        self,
        experiment_name: str,
        scheduler_name: str,
        duration: float,
        teg: TimeExpandedGraph,
    ):
        print(f"Elapsed time: {duration:.4f} seconds")

        self.time_expanded_graph_data.append(
            {
                "name": f"{scheduler_name}_{experiment_name}",
                "teg": teg,
            }
        )

        weights.eval_eff_ct = {}

        node_capacities = compute_node_capacities(
            teg.graphs,
            teg.state_durations,
            teg.K,
            teg.nodes,
            teg.graphs,
            teg.pos,
            teg.optical_interfaces_to_node,
            teg.node_to_optical_interfaces,
        )

        # for cap in node_capacities:
        #     print(cap.id, min(cap.capacity_in, cap.capacity_out))
        #
        # for (k, tx, rx, d), t in weights.eval_eff_ct.items():
        #     print(k, tx, rx, t, d)

        network_capacity = int(compute_capacity(node_capacities))
        network_wasted_capacity = compute_wasted_capacity(node_capacities)

        wasted_buffer_capacity = int(compute_wasted_buffer(node_capacities))

        jains_fairness_index = compute_jains_fairness_index(
            teg.graphs,
            teg.state_durations,
            teg.nodes,
            teg.K,
            teg.N,
            teg.optical_interfaces_to_node,
        )

        scheduled_delay = compute_scheduled_delay(
            teg.graphs,
            teg.state_durations,
            teg.nodes,
            teg.K,
            teg.N,
            teg.optical_interfaces_to_node,
        )

        row = [
            scheduler_name,
            experiment_name,
            duration,
            network_capacity,
            network_wasted_capacity,
            wasted_buffer_capacity,
            jains_fairness_index,
            scheduled_delay,
        ]
        self.reports.append(row)

        print(f"Scheduled network capacity: {network_capacity:,}")
        print(f"Scheduled network wasted capacity: {network_wasted_capacity:,}")

        print(f"Wasted buffer capacity: {wasted_buffer_capacity:,}")

        print(f"Jain's fairness index: {jains_fairness_index}")
        print(f"Average delay: {scheduled_delay}")

    def write_report(self):
        # Create CSV of runtimes, capacity, and wasted capacity
        basic_report_headers = [
            "Algorithm",
            "Scenario",
            "Execution duration",
            "Capacity",
            "Wasted capacity",
            "Wasted buffer capacity",
            "Jain's fairness index",
            "Scheduled delay",
        ]

        report_id = int(time.time())
        print(f"Writing report ID {report_id} to disk")

        report_dir = os.path.join(constants.REPORTS_ROOT, f"{report_id}")
        os.mkdir(report_dir)

        report_path = str(
            os.path.join(report_dir, f"{report_id}_{FileType.REPORT.value}.csv")
        )
        with open(report_path, "w") as f:
            writer = csv.writer(f, delimiter=",", lineterminator="\n")
            writer.writerow(basic_report_headers)
            writer.writerows(self.reports)

        if not self.write_pkl:
            return

        # Save each teg as a pkl file to the reports directory so that it can be used for analysis later
        for teg_data_dict in self.time_expanded_graph_data:
            pkl_path = str(os.path.join(report_dir, f"{teg_data_dict['name']}.pkl"))
            with open(pkl_path, "wb") as f:
                pickle.dump(teg_data_dict["teg"], f)
