import copy
from dataclasses import dataclass

import numpy as np

import constants
from contact_plan import ContactPlan, Contact
from utils import get_experiment_file, FileType


@dataclass
class Graph:
    contacts: list[Contact]
    adj_matrix: np.ndarray
    k: int
    state_duration: int  # seconds
    state_start_time: int  # seconds

    def __repr__(self):
        rep = ""
        for row_idx in range(len(self.adj_matrix)):
            for col_idx in range(len(self.adj_matrix[row_idx])):
                if row_idx == col_idx:
                    rep += "*,"
                else:
                    rep += f"{self.adj_matrix[row_idx][col_idx]},"

            rep += "\n"

        rep += f"k={self.k}\n"
        # rep += f"state_start_time={self.state_start_time}\n"
        rep += f"t={self.state_duration}\n"
        # rep += f"{self.contacts}"

        return rep


@dataclass
class TimeExpandedGraph:
    graphs: list[Graph]
    nodes: list[str]
    node_map: dict[str, int]
    ipn_node_to_planet_map: dict[int, str]
    start_time: int
    end_time: int

    def __repr__(self):
        rep = ""
        rep += f"num_k={len(self.graphs)}\n"
        rep += f"num_nodes={len(self.nodes)}\n"
        rep += f"duration={(self.end_time - self.start_time) / 60 / 60} hours\n"
        rep += f"ipn_node_to_planet_map={self.ipn_node_to_planet_map}\n"
        rep += "\n"
        for graph in self.graphs:
            rep += f"{graph}\n"

        return rep


def build_time_expanded_graph(contact_plan: ContactPlan) -> TimeExpandedGraph:
    # Define the list of interplanetary nodes i.e. the nodes who can establish interplanetary links.
    # We are defining this as any contact with a range greater than 100,000 km.
    # A non-ipn node can receive across interplanetary distances, but cannot transmit across interplanetary
    # distances. We are defining interplanetary nodes here as nodes that can transmit and receive across interplanetary
    # distances. This depends on the scenario. Some definitions say that a non-ipn node cannot receive or transmit
    # across interplanetary distances, the code below supports both use cases.
    interplanetary_contacts = [contact for contact in contact_plan.contacts if contact.context["range"] > 100_000]
    interplanetary_tx_nodes = [contact.tx_node for contact in interplanetary_contacts]
    interplanetary_rx_nodes = [contact.rx_node for contact in interplanetary_contacts]
    interplanetary_nodes = [node for node in interplanetary_tx_nodes + interplanetary_rx_nodes
                            if node in interplanetary_tx_nodes and node in interplanetary_rx_nodes]

    # Create a sets of all start times and end times, the combined length will equal numK
    start_times = list(set([contact.start_time for contact in contact_plan.contacts]))
    end_times = list(set([contact.end_time for contact in contact_plan.contacts]))
    # After combining the discrete start and end times, convert to a set to only keep unique values. If there is a
    # contact start and contact end time that are at the same time, without doing this, it will create a state of
    # duration 0. This won't affect the results at all because no capacity can come from a state of duration = 0, but
    # with how the logic is set up that state should not exist.
    time_steps = sorted(list(set(start_times + end_times)))

    # Create a unique list of node ids and map them to array index for the adjacency matrix graph
    unique_nodes = sorted(set(
        [contact.rx_node for contact in contact_plan.contacts]
        + [contact.tx_node for contact in contact_plan.contacts]))
    node_map = {node: idx for idx, node in enumerate(unique_nodes)}

    # We want to split the list of interplanetary nodes into different sets of nodes for each planet. We can do this
    # by using the first digit of each node id to identify its constellation. We make the assumption that each planet
    # has at most a single interplanetary constellation
    ipn_node_to_planet_map = {}  # ipn_node_idx -> planet_id
    for idx, node in enumerate(unique_nodes):
        if node in interplanetary_nodes:
            ipn_node_to_planet_map[idx] = node[0]

    contact_topology_graphs = []
    for index, time_step in enumerate(time_steps[:-1]):
        state_start_time = time_step
        state_duration = time_steps[index + 1] - state_start_time

        # For each time step get a list of contacts that exist within that time step
        included_contacts = [contact for contact in contact_plan.contacts
                             if include_contact(contact, state_start_time, state_duration)]

        # The index here will map the node name to its index in the adjacency matrix, by default the values are set to 0
        # which indicates there is no contact between the two nodes
        num_nodes = len(unique_nodes)
        adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
        for tx_idx, tx_node in enumerate(unique_nodes):
            # List of rx_nodes that have a contact with the tx_node in this time step
            rx_nodes = [contact.rx_node for contact in included_contacts if contact.tx_node == tx_node]
            rx_idxs = [node_map[rx_node] for rx_node in rx_nodes]

            for rx_idx in rx_idxs:
                # For now, we assume all satellites only have a single default interface.
                adj_matrix[tx_idx][rx_idx] = constants.default_a

        contact_topology_graphs.append(Graph(
            contacts=included_contacts,
            adj_matrix=adj_matrix,
            k=index + 1,
            state_duration=state_duration,
            state_start_time=state_start_time,
        ))

    return TimeExpandedGraph(
        graphs=contact_topology_graphs,
        nodes=unique_nodes,
        ipn_node_to_planet_map=ipn_node_to_planet_map,
        node_map=node_map,
        start_time=min(start_times),
        end_time=max(end_times))


def include_contact(contact: Contact, state_start_time: int, state_duration: int) -> bool:
    state_end_time = state_start_time + state_duration
    # The current state should include contacts that start before the state start time, inclusive, and end after the
    # start end time, inclusive
    return contact.start_time <= state_start_time and contact.end_time >= state_end_time


def convert_time_expanded_graph_to_contact_plan(time_expanded_graph: TimeExpandedGraph) -> ContactPlan:
    contacts: list[Contact] = []
    in_progress_contacts: list[list[Contact]] = \
        [[None for _ in range(len(time_expanded_graph.nodes))] for _ in range(len(time_expanded_graph.nodes))]

    rolling_start_time = 0

    for graph in time_expanded_graph.graphs:
        for tx_idx in range(len(graph.adj_matrix)):
            for rx_idx in range(len(graph.adj_matrix[tx_idx])):
                # We have 4 different cases depending on the state of the in_progress_contacts and graph at index
                # tx_idx and rx_idx
                if graph.adj_matrix[tx_idx][rx_idx] == 0 and not in_progress_contacts[tx_idx][rx_idx]:
                    continue
                elif graph.adj_matrix[tx_idx][rx_idx] == 0 and in_progress_contacts[tx_idx][rx_idx]:
                    # End the contact
                    contacts.append(copy.deepcopy(in_progress_contacts[tx_idx][rx_idx]))
                    in_progress_contacts[tx_idx][rx_idx] = None
                elif graph.adj_matrix[tx_idx][rx_idx] == 1 and not in_progress_contacts[tx_idx][rx_idx]:
                    # Start the contact
                    in_progress_contacts[tx_idx][rx_idx] = Contact(
                        tx_node=time_expanded_graph.nodes[tx_idx],
                        rx_node=time_expanded_graph.nodes[rx_idx],
                        start_time=rolling_start_time,
                        # We can only assume the contact will be there for a single state, so set the end_time to the
                        # duration of the current state, this will be updated as it appears in later graphs
                        end_time=rolling_start_time + graph.state_duration,
                        context={},
                    )
                elif graph.adj_matrix[tx_idx][rx_idx] == 1 and in_progress_contacts[tx_idx][rx_idx]:
                    # Update the in progress contact but extending its end time to the end of the current state
                    in_progress_contacts[tx_idx][rx_idx].end_time += graph.state_duration

        # For each graph we traverse add the duration of that state to the rolling start time that new contacts use
        rolling_start_time += graph.state_duration

    # Cleanup the in progress contacts that start in the last state or last until the last state
    for contact_row_idx in range(len(in_progress_contacts)):
        for contact_col_idx in range(len(in_progress_contacts[contact_row_idx])):
            if in_progress_contacts[contact_row_idx][contact_col_idx]:
                contacts.append(copy.deepcopy(in_progress_contacts[contact_row_idx][contact_col_idx]))
                in_progress_contacts[contact_row_idx][contact_col_idx] = None

    return ContactPlan(contacts)


def time_expanded_graph_splitter(time_expanded_graph: TimeExpandedGraph) -> TimeExpandedGraph:
    return time_expanded_graph


def write_time_expanded_graph(experiment_name: str, time_expanded_graph: TimeExpandedGraph, file_type: FileType):
    path = get_experiment_file(experiment_name, file_type)
    with open(path, "w") as f:
        f.write(str(time_expanded_graph))
