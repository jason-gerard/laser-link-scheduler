from dataclasses import dataclass
from enum import Enum

from contact_plan import ContactPlan, Contact
from utils import get_experiment_file, FileType


class StepType(Enum):
    START = 1
    END = 2


# TODO remove this, not needed anymore with simpler calculation
@dataclass
class TimeStep:
    t: int
    step_type: StepType

    def __lt__(self, other):
        return self.t < other.t


@dataclass
class Graph:
    contacts: list[Contact]
    adj_matrix: list[list[int]]
    k: int
    state_duration: int  # seconds
    state_start_time: int

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
    interplanetary_nodes: list[int]
    ipn_node_to_planet_map: dict[int, str]
    start_time: int
    end_time: int

    def __repr__(self):
        rep = ""
        rep += f"num_k={len(self.graphs)}\n"
        rep += f"num_nodes={len(self.nodes)}\n"
        rep += f"duration={(self.end_time - self.start_time) / 60 / 60} hours\n"
        rep += f"src_interplanetary_nodes={self.interplanetary_nodes}\n"
        rep += f"ipn_node_to_planet_map={self.ipn_node_to_planet_map}\n"
        rep += "\n"
        for graph in self.graphs:
            rep += f"{graph}\n"

        return rep

    class Builder:
        graphs: list[Graph] = []
        nodes: list[str]
        node_map: dict[str, int]
        interplanetary_nodes: list[int]
        ipn_node_to_planet_map: dict[int, str]
        start_time: int
        end_time: int

        def with_graph(self, graph: Graph):
            self.graphs.append(graph)
            return self

        def with_nodes(self, nodes: list[str]):
            self.nodes = nodes
            return self

        def with_node_map(self, node_map: dict[str, int]):
            self.node_map = node_map
            return self

        def with_interplanetary_nodes(self, interplanetary_nodes: list[int]):
            self.interplanetary_nodes = interplanetary_nodes
            return self

        def with_ipn_node_to_planet_map(self, ipn_node_to_planet_map: dict[int, str]):
            self.ipn_node_to_planet_map = ipn_node_to_planet_map
            return self

        def with_start_time(self, start_time: int):
            self.start_time = start_time
            return self

        def with_end_time(self, end_time: int):
            self.end_time = end_time
            return self

        def build(self):
            return TimeExpandedGraph(
                graphs=self.graphs,
                nodes=self.nodes,
                interplanetary_nodes=self.interplanetary_nodes,
                ipn_node_to_planet_map=self.ipn_node_to_planet_map,
                node_map=self.node_map,
                start_time=self.start_time,
                end_time=self.end_time,
            )


def build_time_expanded_graph(contact_plan: ContactPlan) -> TimeExpandedGraph:
    builder = TimeExpandedGraph.Builder()

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
    start_times = set([contact.start_time for contact in contact_plan.contacts])
    start_time_steps = [TimeStep(time, StepType.START) for time in start_times]

    end_times = set([contact.end_time for contact in contact_plan.contacts])
    end_time_steps = [TimeStep(time, StepType.END) for time in end_times]

    builder \
        .with_start_time(min(start_times)) \
        .with_end_time(max(end_times))

    time_steps = sorted(start_time_steps + end_time_steps)

    # Create a unique list of node ids and map them to array index for the adjacency matrix graph
    unique_nodes = sorted(set(
        [contact.rx_node for contact in contact_plan.contacts]
        + [contact.tx_node for contact in contact_plan.contacts]))
    node_map = {node: idx for idx, node in enumerate(unique_nodes)}

    # We want to split the list of interplanetary nodes into different sets of nodes for each planet. We can do this
    # by using the first digit of each node id to identify its constellation. We make the assumption that each planet
    # has at most a single interplanetary constellation
    ipn_node_to_planet_map = {}  # ipn_node_id -> planet_id
    for idx, node in enumerate(unique_nodes):
        if node in interplanetary_nodes:
            ipn_node_to_planet_map[idx] = node[0]

    interplanetary_node_ids = [idx for idx, node in enumerate(unique_nodes) if node in interplanetary_nodes]

    builder \
        .with_nodes(unique_nodes) \
        .with_node_map(node_map) \
        .with_interplanetary_nodes(interplanetary_node_ids) \
        .with_ipn_node_to_planet_map(ipn_node_to_planet_map)

    for index, time_step in enumerate(time_steps[:-1]):
        state_start_time = time_step.t
        state_duration = time_steps[index + 1].t - state_start_time

        # There is a bug if start and end time is equal for a new contact
        included_contacts = [contact for contact in contact_plan.contacts
                             if include_contact(contact, state_start_time, state_duration)]

        # The index here will map the node name to its index in the adjacency matrix
        adj_matrix = [[-1 for _ in range(len(unique_nodes))] for _ in range(len(unique_nodes))]
        for tx_idx, tx_node in enumerate(unique_nodes):
            # List of rx_nodes that have a contact with the tx_node in this time step
            rx_nodes = [contact.rx_node for contact in included_contacts if contact.tx_node == tx_node]

            for rx_idx, rx_node in enumerate(unique_nodes):
                if tx_node == rx_node:
                    adj_matrix[tx_idx][rx_idx] = 0
                elif rx_node in rx_nodes:  # Check if there exists a contact between these two nodes
                    adj_matrix[tx_idx][rx_idx] = 1
                else:  # There is no contact between the two nodes in this time step
                    adj_matrix[tx_idx][rx_idx] = 0

        builder.with_graph(Graph(
            contacts=included_contacts,
            adj_matrix=adj_matrix,
            k=index + 1,
            state_duration=state_duration,
            state_start_time=state_start_time,
        ))

    return builder.build()


def include_contact(contact: Contact, state_start_time: int, state_duration: int) -> bool:
    state_end_time = state_start_time + state_duration
    # The current state should include contacts that start before the state start time, inclusive, and end after the
    # start end time, inclusive
    return contact.start_time <= state_start_time and contact.end_time >= state_end_time


def write_time_expanded_graph(experiment_name: str, time_expanded_graph: TimeExpandedGraph):
    path = get_experiment_file(experiment_name, FileType.TEG)
    with open(path, "w") as f:
        f.write(str(time_expanded_graph))
