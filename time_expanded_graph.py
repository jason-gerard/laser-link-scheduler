from dataclasses import dataclass, replace

import numpy as np
from tqdm import tqdm

import constants
from contact_plan import ContactPlan, Contact
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
    
    pos: np.ndarray

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


def convert_contact_plan_to_time_expanded_graph(contact_plan: ContactPlan, should_fractionate: bool) -> TimeExpandedGraph:
    # Define the list of interplanetary nodes i.e. the nodes who can establish interplanetary links.
    # We are defining this as any contact with a range greater than 100,000 km.
    # A non-ipn node can receive across interplanetary distances, but cannot transmit across interplanetary
    # distances. We are defining interplanetary nodes here as nodes that can transmit and receive across interplanetary
    # distances. This depends on the scenario. Some definitions say that a non-ipn node cannot receive or transmit
    # across interplanetary distances, the code below supports both use cases.
    interplanetary_contacts = [contact for contact in contact_plan.contacts
                               if contact.range > constants.INTERPLANETARY_RANGE]
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
    
    positions = np.empty((K, N, 3), dtype="float64")

    print("Starting contact plan to time expanded graph conversion")
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
            tx_included_contacts = [contact for contact in included_contacts if contact.tx_node == tx_node]
            rx_nodes = [contact.rx_node for contact in tx_included_contacts]
            rx_idxs = [node_map[rx_node] for rx_node in rx_nodes]

            for rx_idx in rx_idxs:
                # For now, we assume all satellites only have a single default interface.
                contact_topology_graphs[k][tx_idx][rx_idx] = constants.default_a
                
            # Add position data for the node
            tx_x = tx_included_contacts[0].tx_x
            tx_y = tx_included_contacts[0].tx_y
            tx_z = tx_included_contacts[0].tx_z
            positions[k][tx_idx] = [tx_x, tx_y, tx_z]

    time_expanded_graph = TimeExpandedGraph(
        graphs=contact_topology_graphs,
        contacts=contacts_by_state,
        state_durations=state_durations,
        K=K,
        N=N,
        nodes=unique_nodes,
        node_map=node_map,
        ipn_node_to_planet_map=ipn_node_to_planet_map,
        W=np.array([]),
        pos=positions)

    # This process of fractionation splits long contacts in the TEG into multiple smaller contacts, this will result in
    # each k state having a maximum duration of d_max. Since there are more states and more decision points some
    # algorithms will have better performance.
    return fractionate_graph(time_expanded_graph) if should_fractionate else time_expanded_graph


def include_contact(contact: Contact, state_start_time: int, state_duration: int) -> bool:
    state_end_time = state_start_time + state_duration
    # The current state should include contacts that start before the state start time, inclusive, and end after the
    # start end time, inclusive
    return contact.start_time <= state_start_time and contact.end_time >= state_end_time


def fractionate_graph(time_expanded_graph: TimeExpandedGraph) -> TimeExpandedGraph:
    new_k = time_expanded_graph.K
    for k in range(time_expanded_graph.K):
        if time_expanded_graph.state_durations[k] > constants.d_max:

            large_duration = time_expanded_graph.state_durations[k]
            new_k -= 1

            while large_duration > constants.d_max:
                large_duration -= constants.d_max
                new_k += 1

            if large_duration > 0:
                new_k += 1

    new_teg_graph = np.zeros((new_k, time_expanded_graph.N, time_expanded_graph.N), dtype="int64")
    new_teg_durations = np.empty(new_k, dtype='int64')
    new_contacts = [[] for _ in range(new_k)]

    k_offset = 0

    for k in range(time_expanded_graph.K):
        if time_expanded_graph.state_durations[k] > constants.d_max:

            large_duration = time_expanded_graph.state_durations[k]
            kth_graph = time_expanded_graph.graphs[k, :, :]
            kth_contacts = time_expanded_graph.contacts[k]

            while large_duration > constants.d_max:
                new_teg_durations[k + k_offset] = constants.d_max
                new_teg_graph[k + k_offset] = kth_graph
                new_contacts[k + k_offset] = kth_contacts

                large_duration -= constants.d_max
                k_offset += 1

            if large_duration > 0:
                new_teg_durations[k + k_offset] = large_duration
                new_teg_graph[k + k_offset] = kth_graph
                new_contacts[k + k_offset] = kth_contacts
                k_offset += 1
            
            k_offset -= 1
        else:
            new_teg_graph[k + k_offset] = time_expanded_graph.graphs[k]
            new_teg_durations[k + k_offset] = time_expanded_graph.state_durations[k]
            new_contacts[k + k_offset] = time_expanded_graph.contacts[k]

    time_expanded_graph.graphs = new_teg_graph
    time_expanded_graph.state_durations = new_teg_durations
    time_expanded_graph.contacts = new_contacts
    time_expanded_graph.K = new_k

    print(f"Finished graph fractionation, the number of states is {new_k}")

    return time_expanded_graph


def convert_time_expanded_graph_to_contact_plan(teg: TimeExpandedGraph) -> ContactPlan:
    contacts: list[Contact] = []
    # This matrix keeps track of active contacts where the value of the matrix = -1 if there is no active contact for
    # that edge, or a value >= 0 when there is an active contact for that edge. The value of the edge is equal to the
    # start time of that contact. This allows us to merge contacts that are active across multiple of the k states.
    active_contacts = np.full((teg.N, teg.N), fill_value=-1, dtype='int64')

    for k in tqdm(range(teg.K)):
        for tx_idx in range(teg.N):
            for rx_idx in range(teg.N):
                should_start_contact = teg.graphs[k][tx_idx][rx_idx] >= 1 and active_contacts[tx_idx][rx_idx] == -1
                should_end_contact = teg.graphs[k][tx_idx][rx_idx] == 0 and active_contacts[tx_idx][rx_idx] >= 0

                if should_start_contact:
                    active_contacts[tx_idx][rx_idx] = np.sum(teg.state_durations[0:k]) if k > 0 else 0
                elif should_end_contact:
                    # Find the associated contact in the previous state since now the contact is over
                    possible_contact = [contact for contact in teg.contacts[k - 1]
                                        if contact.tx_node == teg.nodes[tx_idx] and contact.rx_node == teg.nodes[rx_idx]]
                    
                    if possible_contact:
                        interface_id = teg.graphs[k - 1][tx_idx][rx_idx]
                        bit_rate = constants.R[interface_id]

                        contacts.append(replace(
                            possible_contact[0],
                            start_time=active_contacts[tx_idx][rx_idx],
                            end_time=np.sum(teg.state_durations[0:k]),
                            bit_rate=bit_rate))

                    active_contacts[tx_idx][rx_idx] = -1

                # If we are at the last state of the time expanded graph and there are still active contacts then
                # "clean" them up and append them to the list of contacts. Without this the loop will terminate and
                # not add these
                is_contact_in_progress = teg.graphs[k][tx_idx][rx_idx] >= 1 and active_contacts[tx_idx][rx_idx] >= 0
                should_cleanup_contact = k == teg.K - 1 and (should_start_contact or is_contact_in_progress)
                if should_cleanup_contact:
                    possible_contact = [contact for contact in teg.contacts[k]
                                        if contact.tx_node == teg.nodes[tx_idx] and contact.rx_node == teg.nodes[rx_idx]]

                    if possible_contact:
                        interface_id = teg.graphs[k][tx_idx][rx_idx]
                        bit_rate = constants.R[interface_id]

                        contacts.append(replace(
                            possible_contact[0],
                            start_time=active_contacts[tx_idx][rx_idx],
                            end_time=np.sum(teg.state_durations[0:k + 1]),
                            bit_rate=bit_rate))

    return ContactPlan(sorted(contacts, key=lambda c: c.end_time))


def write_time_expanded_graph(experiment_name: str, time_expanded_graph: TimeExpandedGraph, file_type: FileType):
    path = get_experiment_file(experiment_name, file_type)
    with open(path, "w") as f:
        f.write(str(time_expanded_graph))
