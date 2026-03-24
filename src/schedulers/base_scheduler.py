from abc import abstractmethod
import networkx as nx
import numpy as np

from src.time_expanded_graph.time_expanded_graph import (
    TimeExpandedGraph,
)
from src.topology.contact_plan import Contact
from src.utils import ProgressCallback


class BaseScheduler:
    def _blossom(self, P_k: np.ndarray, W_k: np.ndarray) -> set:
        """
        The blossom algorithm assumes undirected edges meaning that we cannot have A -> B without B -> A. Logically this
        makes sense for laser communications because both laser transceivers must be physically pointing towards each other.
        This algorithm does support individually defining the properties of the laser in each direction i.e. A -> B with
        laser ID 1 but B -> A with laser ID 3.
        """

        # Create list of edges, represented by three-tuples of (tx_idx, rx_idx, weight),
        # based on the contact topology P_k and computed weights based on
        # delta_capacity + alpha * delta_time.
        #
        # Because the contact topology is symmetric, we can omit the bottom triangle.
        # When we compute the weight matrix it is not symmetric, because we compute the
        # capacity on an edge basis, but since NetworkX uses an undirected graph it
        # will consider both directions of the edge. To account for this, we sum the
        # weights in either direction to obtain the total weight for that undirected edge.

        sym_weights = W_k + W_k.T
        valid_mask = np.triu(P_k >= 1, k=1) & (sym_weights >= 0)

        tx_idx, rx_idx = np.where(valid_mask)
        edges = list(zip(tx_idx, rx_idx, sym_weights[tx_idx, rx_idx]))

        # Create graph containing edges from P_k
        G = nx.Graph()
        G.add_weighted_edges_from(edges)

        # Perform max weight matching using the blossom algorithm. We leverage the networkx library to do this
        return nx.max_weight_matching(G)

    def _build_graph(
        self,
        matched_edges: set,
        contact_topology_k: np.ndarray,
        contacts_k: list[Contact],
        node_map: dict[str, int],
    ) -> tuple[np.ndarray, list[Contact]]:
        num_nodes = len(contact_topology_k)
        # Build adj_matrix from matched edges list. nx.max_weight_matching works on an undirected graph so when we see
        # an edge add it in both directions i.e. (i,j) and (j,i)
        contact_plan_k = np.zeros((num_nodes, num_nodes), dtype="int64")
        for tx_idx, rx_idx in matched_edges:
            # Make sure to map the value of the graph i.e. the communication interface id back to the correct edge. This
            # allows us to support different lasers in each direction while using an undirected graph algorithm (blossom)
            contact_plan_k[tx_idx][rx_idx] = contact_topology_k[tx_idx][rx_idx]
            contact_plan_k[rx_idx][tx_idx] = contact_topology_k[rx_idx][tx_idx]

        contacts = [
            contact
            for contact in contacts_k
            if self._should_keep_contact(matched_edges, node_map, contact)
        ]

        return contact_plan_k, contacts

    def _should_keep_contact(
        self, matched_edges: set, node_map: dict[str, int], contact: Contact
    ) -> bool:
        node1_idx = node_map[contact.tx_node]
        node2_idx = node_map[contact.rx_node]

        return (node1_idx, node2_idx) in matched_edges or (
            node2_idx,
            node1_idx,
        ) in matched_edges

    # TODO: Move `teg` to the scheduler instance (store it on `BaseScheduler`) and remove it from the `schedule()` API.
    #       Target: `BaseScheduler.schedule(self) -> TimeExpandedGraph`, with `BaseScheduler` owning/initializing `self.teg`.
    #       Then update all scheduler implementations and call sites to use the new signature.
    @abstractmethod
    def schedule(
        self,
        teg: TimeExpandedGraph,
        progress_callback: ProgressCallback | None = None,
    ) -> TimeExpandedGraph: ...
