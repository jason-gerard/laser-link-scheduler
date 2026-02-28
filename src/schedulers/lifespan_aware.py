import numpy as np
from tqdm import tqdm

from src import constants
from src.time_expanded_graph.time_expanded_graph import (
    TimeExpandedGraph,
    Node,
)
from .base_scheduler import BaseScheduler
from src.topology.weights import (
    disabled_contact_time,
    compute_effective_contact_time,
)

from functools import total_ordering


@total_ordering
class NodeLifespan:
    def __init__(self, id: str, mission_lifespan: float):
        self.id = id
        self.mission_lifespan = mission_lifespan

    def __eq__(self, value):
        if not isinstance(value, NodeLifespan):
            return self.mission_lifespan == value
        return self.mission_lifespan == value.mission_lifespan

    def __lt__(self, value):
        if not isinstance(value, NodeLifespan):
            return self.mission_lifespan < value
        return self.mission_lifespan < value.mission_lifespan


class LifespanAware(BaseScheduler):
    # def _initialize_node_lifespans(
    #     self, nodes: list[Node], **args
    # ) -> list[NodeLifespan]:
    #     lifespans: list[NodeLifespan] = []

    #     return lifespans

    # def _weight_node_lifespans(
    #     self,
    #     state: int,
    #     previous_schedule_contact_topology: np.ndarray,
    #     number_oi: int,
    #     teg: TimeExpandedGraph,
    # ) -> list[NodeLifespan]:

    #     for tx_oi_idx in range(number_oi):
    #         for rx_oi_idx in range(number_oi):
    #             if not teg.graphs[state][tx_oi_idx][rx_oi_idx]:
    #                 continue

    #             bit_rate = min(
    #                 constants.BIT_RATES[
    #                     teg.nodes[teg.optical_interfaces_to_node[tx_oi_idx]]
    #                 ],
    #                 constants.BIT_RATES[
    #                     teg.nodes[teg.optical_interfaces_to_node[rx_oi_idx]]
    #                 ],
    #             )
    #             # In this first approach we'll bypass the retargeting time
    #             # effective_contact_duration = state_duration
    #             effective_contact_duration = compute_effective_contact_time(
    #                 tx_oi_idx,
    #                 rx_oi_idx,
    #                 previous_schedule_contact_topology,
    #                 teg.state_durations[state],
    #                 teg.pos,
    #                 teg.optical_interfaces_to_node,
    #                 teg.nodes,
    #                 True,
    #             )

    #             lifespans.append()
    #             ...

    #     ...

    # def _compute_node_lifespans(
    #     self,
    #     state: int,
    #     L_k: np.ndarray,
    #     state_duration: int,
    #     nodes: list[Node],
    #     scheduled_graphs: np.ndarray,
    #     weights_delta_lfspn: np.ndarray,
    # ) -> list[NodeLifespan]:
    #     new_node_lifespan = []
    #     for node in nodes:
    #         new_node_lifespan.append(
    #             NodeLifespan(
    #                 id=node,
    #                 mission_lifespan=(
    #                     weights_delta_lfspn[node][node] - state_duration
    #                 ),
    #             )
    #         )
    #     return new_node_lifespan

    def schedule(self, teg: TimeExpandedGraph) -> TimeExpandedGraph:
        """
        Placeholder for lifespan schedule algorithm
        """
        scheduled_graphs = np.empty((teg.K, teg.N, teg.N), dtype="int64")
        scheduled_contacts = []
        weights = np.empty((teg.K, teg.N, teg.N), dtype="float32")

        # node_lifespans = _initialize_node_lifespans()
        # weights_dct = np.zeros((teg.N, teg.N), dtype="int64")

        for state in tqdm(range(teg.K)):
            #
            # Description here
            #
            # weights_delta_lifespan = self._weight_node_lifespans(teg)

            # Compute the weight of each edge by doing a weighted sum of the lifespan and fairness metrics
            # weights[state] = (
            #     (1 - constants.alpha) * weights_delta_lifespan
            # ) + (constants.alpha * weights_dct)

            # # Compute max weight maximal matching using the blossom algorithm
            # matched_edges = self._blossom(teg.graphs[state], weights[state])

            # # Compute L_k from the matched edges
            # L_k, contacts = self._build_graph(
            #     matched_edges,
            #     teg.graphs[state],
            #     teg.contacts[state],
            # )
            # scheduled_graphs[state] = L_k
            # scheduled_contacts.append(contacts)

            # Update node_lifespan list with node lifespans from current state contact plan and merge them together
            # scheduled_node_lifespans = self._compute_node_lifespans(
            #     L_k,
            #     teg.state_durations[state],
            #     teg.nodes,
            #     scheduled_graphs[:state],
            #     weights_delta_lifespan,
            # )

            # Update the matrix containing the disabled contact time for current state
            # weights_dct += disabled_contact_time(
            #     teg.graphs[state], L_k, teg.state_durations[state]
            # )
            ...
        return TimeExpandedGraph(
            graphs=scheduled_graphs,
            contacts=scheduled_contacts,
            state_durations=teg.state_durations,
            K=teg.K,
            N=teg.N,
            nodes=teg.nodes,
            node_map=teg.node_map,
            W=weights,
            pos=teg.pos,
            optical_interfaces_to_node=teg.optical_interfaces_to_node,
            node_to_optical_interfaces=teg.node_to_optical_interfaces,
            effective_contact_durations=teg.effective_contact_durations,
        )
