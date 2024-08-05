import copy
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

import constants
from contact_plan import ContactPlan, Contact, IONContactPlanParser
from utils import get_experiment_file, FileType


@dataclass
class TimeExpandedGraph:
    graphs: np.ndarray
    contacts: list[list[Contact]]
    state_durations: np.ndarray
    K: int  # number of states
    N: int  # number of nodes
    nodes: list[str]
    node_map: dict[str, int]
    ipn_node_to_planet_map: dict[int, str]

    W: np.ndarray  # 3D Weights matrix [k][tx_idx][rx_idx]

    def __repr__(self):
        end_time = sum(self.state_durations)

        rep = ""
        rep += f"num_k={len(self.graphs)}\n"
        rep += f"num_nodes={len(self.nodes)}\n"
        rep += f"duration={end_time / 60 / 60} hours\n"
        rep += f"ipn_node_to_planet_map={self.ipn_node_to_planet_map}\n"
        rep += "\n"
        for k in range(self.K):
            for row_idx in range(self.N):
                for col_idx in range(self.N):
                    if row_idx == col_idx:
                        rep += "*,"
                    else:
                        rep += f"{self.graphs[k][row_idx][col_idx]},"

                rep += "\n"
            rep += f"k={k + 1}\n"
            rep += f"t={self.state_durations[k]}\n"
            rep += "\n"

        return rep


def convert_contact_plan_to_time_expanded_graph(contact_plan: ContactPlan) -> TimeExpandedGraph:
    # Define the list of interplanetary nodes i.e. the nodes who can establish interplanetary links.
    # We are defining this as any contact with a range greater than 100,000 km.
    # A non-ipn node can receive across interplanetary distances, but cannot transmit across interplanetary
    # distances. We are defining interplanetary nodes here as nodes that can transmit and receive across interplanetary
    # distances. This depends on the scenario. Some definitions say that a non-ipn node cannot receive or transmit
    # across interplanetary distances, the code below supports both use cases.
    interplanetary_contacts = [contact for contact in contact_plan.contacts
                               if contact.context[IONContactPlanParser.RANGE_CONTEXT] > constants.INTERPLANETARY_RANGE]
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

    N = len(unique_nodes)
    K = len(time_steps) - 1

    contact_topology_graphs = np.zeros((K, N, N), dtype='int64')
    contacts_by_state = []
    state_durations = np.empty(K, dtype='int64')

    for k, time_step in enumerate(tqdm(time_steps[:-1])):
        state_start_time = time_step
        state_duration = time_steps[k + 1] - state_start_time
        state_durations[k] = state_duration

        # For each time step get a list of contacts that exist within that time step
        included_contacts = [contact for contact in contact_plan.contacts
                             if include_contact(contact, state_start_time, state_duration)]
        contacts_by_state.append(included_contacts)

        # The index here will map the node name to its index in the adjacency matrix, by default the values are set to 0
        # which indicates there is no contact between the two nodes
        for tx_idx, tx_node in enumerate(unique_nodes):
            # List of rx_nodes that have a contact with the tx_node in this time step
            rx_nodes = [contact.rx_node for contact in included_contacts if contact.tx_node == tx_node]
            rx_idxs = [node_map[rx_node] for rx_node in rx_nodes]

            for rx_idx in rx_idxs:
                # For now, we assume all satellites only have a single default interface.
                contact_topology_graphs[k][tx_idx][rx_idx] = constants.default_a

    return TimeExpandedGraph(
        graphs=contact_topology_graphs,
        contacts=contacts_by_state,
        state_durations=state_durations,
        K=K,
        N=N,
        nodes=unique_nodes,
        node_map=node_map,
        ipn_node_to_planet_map=ipn_node_to_planet_map,
        W=np.array([]))


def include_contact(contact: Contact, state_start_time: int, state_duration: int) -> bool:
    state_end_time = state_start_time + state_duration
    # The current state should include contacts that start before the state start time, inclusive, and end after the
    # start end time, inclusive
    return contact.start_time <= state_start_time and contact.end_time >= state_end_time


def convert_time_expanded_graph_to_contact_plan(teg: TimeExpandedGraph) -> ContactPlan:
    contacts: list[Contact] = []
    in_progress_contacts: list[list[Contact]] = [[None for _ in range(teg.N)] for _ in range(teg.N)]

    rolling_start_time = 0

    for k in tqdm(range(teg.K)):
        for tx_idx in range(teg.N):
            for rx_idx in range(teg.N):
                # We have 4 different cases depending on the state of the in_progress_contacts and graph at index
                # tx_idx and rx_idx
                if teg.graphs[k][tx_idx][rx_idx] == 0 and not in_progress_contacts[tx_idx][rx_idx]:
                    continue
                elif teg.graphs[k][tx_idx][rx_idx] == 0 and in_progress_contacts[tx_idx][rx_idx]:
                    # End the contact
                    contacts.append(copy.deepcopy(in_progress_contacts[tx_idx][rx_idx]))
                    in_progress_contacts[tx_idx][rx_idx] = None
                elif teg.graphs[k][tx_idx][rx_idx] == 1 and not in_progress_contacts[tx_idx][rx_idx]:
                    tx_node = teg.nodes[tx_idx]
                    rx_node = teg.nodes[rx_idx]

                    associated_contact = [contact for contact in teg.contacts[k]
                                          if contact.tx_node == tx_node and contact.rx_node == rx_node]
                    
                    assert len(associated_contact) == 1

                    # Start the contact
                    in_progress_contacts[tx_idx][rx_idx] = Contact(
                        tx_node=tx_node,
                        rx_node=rx_node,
                        start_time=rolling_start_time,
                        # We can only assume the contact will be there for a single state, so set the end_time to the
                        # duration of the current state, this will be updated as it appears in later graphs
                        end_time=rolling_start_time + teg.state_durations[k],
                        context=associated_contact[0].context,
                    )
                elif teg.graphs[k][tx_idx][rx_idx] == 1 and in_progress_contacts[tx_idx][rx_idx]:
                    # Update the in progress contact but extending its end time to the end of the current state
                    in_progress_contacts[tx_idx][rx_idx].end_time += teg.state_durations[k]

        # For each graph we traverse add the duration of that state to the rolling start time that new contacts use
        rolling_start_time += teg.state_durations[k]

    # Cleanup the in progress contacts that start in the last state or last until the last state
    for contact_row_idx in range(len(in_progress_contacts)):
        for contact_col_idx in range(len(in_progress_contacts[contact_row_idx])):
            if in_progress_contacts[contact_row_idx][contact_col_idx]:
                contacts.append(copy.deepcopy(in_progress_contacts[contact_row_idx][contact_col_idx]))
                in_progress_contacts[contact_row_idx][contact_col_idx] = None
            
    # Merge contacts to minimize the size of the resulting contact plan
    merged_contacts = merge_neighboring_contacts(contacts)

    return ContactPlan(merged_contacts)


def merge_neighboring_contacts(contacts: list[Contact]) -> list[Contact]:
    """
    If there is a contact A->B during time t1 and a contact from A->B during time t2, these contacts should be merged
    together to reduce the length of the contact plan, they are only split due to the k states of the contact plan.
    This should also improve the performance of the simulators that consume the contact plan.
    """
    
    # I went and verified from the different experiments to see how often this happens, and it seems to never happen for
    # any of the scheduling algorithms. I think due to how the weight matrices are constructed with their fairness
    # properties, this case won't happen. After we implement fractionation we can test again to see if this happens.
    return contacts


def graph_fractionation(time_expanded_graph: TimeExpandedGraph) -> TimeExpandedGraph:
    return time_expanded_graph


def write_time_expanded_graph(experiment_name: str, time_expanded_graph: TimeExpandedGraph, file_type: FileType):
    path = get_experiment_file(experiment_name, file_type)
    with open(path, "w") as f:
        f.write(str(time_expanded_graph))
