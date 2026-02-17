import networkx as nx
import numpy as np
from tqdm import tqdm

from laser_link_scheduler import constants
from laser_link_scheduler.time_expanded_graph.time_expanded_graph import (
    TimeExpandedGraph,
)
from .base_scheduler import BaseScheduler
from laser_link_scheduler.topology.contact_plan import Contact
from laser_link_scheduler.topology.weights import (
    compute_node_capacity_by_graph,
    delta_capacity,
    disabled_contact_time,
    merge_many_node_capacities,
)


class BruteForceScheduler(BaseScheduler):
    def schedule(self, teg: TimeExpandedGraph) -> TimeExpandedGraph:
        scheduled_graphs = np.empty((teg.K, teg.N, teg.N), dtype="int64")
        scheduled_contacts = []
        weights = np.empty((teg.K, teg.N, teg.N), dtype="float32")

        # See brute_force_matchings.py

        return TimeExpandedGraph(
            graphs=scheduled_graphs,
            contacts=scheduled_contacts,
            state_durations=teg.state_durations,
            K=teg.K,
            N=teg.N,
            nodes=teg.nodes,
            node_map=teg.node_map,
            ipn_node_to_planet_map=teg.ipn_node_to_planet_map,
            W=weights,
            pos=teg.pos,
            optical_interfaces_to_node=teg.optical_interfaces_to_node,
            node_to_optical_interfaces=teg.node_to_optical_interfaces,
            effective_contact_durations=teg.effective_contact_durations,
        )
