import pulp
import numpy as np

import constants
from constants import RELAY_NODES, SOURCE_NODES, DESTINATION_NODES
from contact_plan import IONContactPlanParser, IPNDContactPlanParser
from pointing_delay_model import pointing_delay
from link_acq_delay_model import link_acq_delay_ipn, link_acq_delay_leo
from report_generator import Reporter
from time_expanded_graph import convert_contact_plan_to_time_expanded_graph, TimeExpandedGraph, \
    write_time_expanded_graph, convert_time_expanded_graph_to_contact_plan
from utils import FileType

MAX_TIME = 2.5 * 60 * 60  # seconds
# MAX_TIME = 120  # seconds
MAX_EDGES_PER_LASER = 1


class PathSchedulerModel:
    def __init__(self, teg: TimeExpandedGraph):
        self.teg = teg
        self.edges = None
        self.edge_deviation_high = None
        self.edge_deviation_low = None
        self.edges_by_node = None
        self.edges_by_state = None
        self.edges_by_state_oi = None
        self.eff_contact_time = None
        self.edge_caps = None
        self.schedule_duration = sum(self.teg.state_durations)
        self.T = self.teg.state_durations
        
        self.flow_model = None
        
    def solve(self):
        # Compute all single hop paths for each k
        # I motif
        single_hop_paths = []
        tx_source = [tx_oi_idx for tx_oi_idx in range(self.teg.N) if self.teg.nodes[self.teg.optical_interfaces_to_node[tx_oi_idx]] in SOURCE_NODES]
        rx_dest = [rx_oi_idx for rx_oi_idx in range(self.teg.N) if self.teg.nodes[self.teg.optical_interfaces_to_node[rx_oi_idx]] in DESTINATION_NODES]
        for k in range(self.teg.K):
            for tx_oi_idx in tx_source:
                for rx_oi_idx in rx_dest:
                    if self.teg.graphs[k][tx_oi_idx][rx_oi_idx] >= 1:
                        single_hop_paths.append((k, tx_oi_idx, rx_oi_idx))

        # Compute all two hop paths for each k
        # L motif
        two_hop_paths = []
        tx_source = [tx_oi_idx for tx_oi_idx in range(self.teg.N) if self.teg.nodes[self.teg.optical_interfaces_to_node[tx_oi_idx]] in SOURCE_NODES]
        relay = [relay_oi_idx for relay_oi_idx in range(self.teg.N) if self.teg.nodes[self.teg.optical_interfaces_to_node[relay_oi_idx]] in RELAY_NODES]
        rx_dest = [rx_oi_idx for rx_oi_idx in range(self.teg.N) if self.teg.nodes[self.teg.optical_interfaces_to_node[rx_oi_idx]] in DESTINATION_NODES]
        for k in range(self.teg.K):
            for tx_oi_idx in tx_source:
                for relay_oi_idx in relay:
                    for rx_oi_idx in rx_dest:
                        if self.teg.graphs[k][tx_oi_idx][relay_oi_idx] >= 1 and self.teg.graphs[k][relay_oi_idx][rx_oi_idx] >= 1:
                            two_hop_paths.append((k, tx_oi_idx, relay_oi_idx, rx_oi_idx))

        # Compute all V paths for each k
        # V motif
        v_paths = []
        tx_source = [tx_oi_idx for tx_oi_idx in range(self.teg.N) if self.teg.nodes[self.teg.optical_interfaces_to_node[tx_oi_idx]] in SOURCE_NODES]
        relay = [relay_oi_idx for relay_oi_idx in range(self.teg.N) if self.teg.nodes[self.teg.optical_interfaces_to_node[relay_oi_idx]] in RELAY_NODES]
        for k in range(self.teg.K):
            for tx_oi_idx1 in tx_source:
                for tx_oi_idx2 in tx_source:
                    for relay_oi_idx in relay:
                        if tx_oi_idx1 != tx_oi_idx2 and self.teg.graphs[k][tx_oi_idx1][relay_oi_idx] >= 1 and self.teg.graphs[k][tx_oi_idx2][relay_oi_idx] >= 1:
                            v_paths.append((k, tx_oi_idx1, tx_oi_idx2, relay_oi_idx))

        print(len(single_hop_paths))
        print(len(two_hop_paths))
        print(len(v_paths))
        print(two_hop_paths[10])
        
        # The contact plan topology here should be in the form of a list of tuples (state idx, i, j)
        contact_topology = []
        for k in range(self.teg.K):
            for tx_oi_idx in range(self.teg.N):
                for rx_oi_idx in range(self.teg.N):
                    if self.teg.graphs[k][tx_oi_idx][rx_oi_idx] >= 1:
                        contact_topology.append((k, tx_oi_idx, rx_oi_idx))
        
        print("num edges", len(contact_topology))
