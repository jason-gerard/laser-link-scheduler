import pulp
import numpy as np

import constants
from constants import RELAY_NODES, SOURCE_NODES, DESTINATION_NODES
from contact_plan import IONContactPlanParser, IPNDContactPlanParser
from dag_topology_reduction import dag_reduction
from report_generator import Reporter
from time_expanded_graph import convert_contact_plan_to_time_expanded_graph, TimeExpandedGraph, \
    write_time_expanded_graph, convert_time_expanded_graph_to_contact_plan
from utils import FileType

src_node_bit_rate = 1000
relay_node_bit_rate = 2000
gs_node_bit_rate = 2000


def get_bit_rate(node_id):
    if node_id in SOURCE_NODES:
        return src_node_bit_rate
    elif node_id in RELAY_NODES:
        return relay_node_bit_rate
    elif node_id in DESTINATION_NODES:
        return gs_node_bit_rate


def get_num_lasers(node_id):
    if node_id in SOURCE_NODES:
        return 1
    elif node_id in RELAY_NODES:
        return 2
    elif node_id in DESTINATION_NODES:
        return 2


class LLSModel:
    def __init__(self, teg: TimeExpandedGraph):
        self.teg = teg
        self.edges = None
        self.edges_by_node = None
        self.edges_by_state = None
        self.schedule_duration = sum(self.teg.state_durations)
        self.T = self.teg.state_durations
        self.bit_rates = {node_id: get_bit_rate(node_id) for node_id in teg.nodes}

    def solve(self):
        # The contact plan topology here should be in the form of a list of tuples (state idx, i, j)
        contact_topology = []
        for k in range(self.teg.K):
            for tx_idx in range(self.teg.N):
                for rx_idx in range(self.teg.N):
                    if self.teg.graphs[k][tx_idx][rx_idx] >= 1:
                        tx_node = self.teg.nodes[tx_idx]
                        rx_node = self.teg.nodes[rx_idx]

                        contact_topology.append((k, tx_node, rx_node))
                        
        print(f"Creating binary variables for {len(contact_topology)} number of edges")
        # This represents the constraint that a selected edge must be a part of the initial contact plan
        self.edges = pulp.LpVariable.dicts(
            "edges", contact_topology, lowBound=0, upBound=1, cat=pulp.LpInteger
        )
        
        # In order to create these constraints faster we will first pre-process the data into a dict such that we can
        # isolate the edges by state k by node
        # the dict will have the key be the node name, which contains a list of length k, where each element is a list
        # of edges
        print(f"Pre-computing edge dictionary")
        self.edges_by_state = {node: [[] for _ in range(self.teg.K)] for node in self.teg.nodes}
        self.edges_by_node = {node: [] for node in self.teg.nodes}
        for edge in self.edges:
            k, tx_node, rx_node = edge

            self.edges_by_state[tx_node][k].append(edge)
            self.edges_by_state[rx_node][k].append(edge)

            self.edges_by_node[tx_node].append(edge)
            self.edges_by_node[rx_node].append(edge)

        print(f"Initializing the capacity variables")
        capacities = {relay_node: pulp.LpVariable(f"Capacity_{relay_node}", lowBound=0)
                      for relay_node in self.teg.nodes if relay_node in RELAY_NODES}

        # Create new single hop capacity variables
        print(f"Initializing single hop capacity variables")
        single_hop_capacities = {gs_node: pulp.LpVariable(f"Capacity_{gs_node}", lowBound=0)
                                 for gs_node in self.teg.nodes if gs_node in DESTINATION_NODES}

        flow_model = pulp.LpProblem("Network_flow_model", pulp.LpMaximize)
        
        print(f"Initializing the objective function")
        # objective function should maximize the summation of the capacity for each relay satellite and dst ground
        # stations
        flow_model += pulp.lpSum(capacities.values()) + pulp.lpSum(single_hop_capacities.values())

        for relay_node, capacity in capacities.items():
            print(f"Setting up inflow and outflow capacity constraints for node {relay_node}")
            # inflow
            flow_model += capacity <= pulp.lpSum([self.flow(i, relay_node)
                                                  for i in self.teg.nodes if i in SOURCE_NODES])
            # outflow
            flow_model += capacity <= pulp.lpSum([self.flow(relay_node, x)
                                                  for x in self.teg.nodes if x in DESTINATION_NODES])
        
        # Create new inflow and outflow capacity constraints for single hop
        for gs_node, capacity in single_hop_capacities.items():
            print(f"Setting up inflow and outflow capacity constraints for ground station node {gs_node}")
            
            # We only have inflow here, outflow is basically infinite since its the dst node. We don't look at inflow
            # from the relay nodes either since that capacity is already accounted for in the two hop capacity
            # constraint above.
            flow_model += capacity <= pulp.lpSum([self.flow(i, gs_node)
                                                 for i in self.teg.nodes if i in SOURCE_NODES])

        # Constraint for fairness of source nodes
        source_nodes = [source_node for source_node in self.teg.nodes if source_node in SOURCE_NODES]
        avg_ect = pulp.lpSum([self.ect(node) for node in source_nodes]) / len(source_nodes) - 1000
        for source_node in source_nodes:
            flow_model += self.ect(source_node) >= avg_ect

        # Constraint that each node can only be a part of a single selected edge per state
        print(f"Setting up 1 to 1 relationship between nodes per state k")
        for node in self.teg.nodes:
            for k in range(self.teg.K):
                flow_model += (
                    pulp.lpSum([self.edges[edge] for edge in self.edges_by_state[node][k]]) <= get_num_lasers(node),
                    f"Max_edge_{node}_{k}",
                )

        print("Starting solve...")
        flow_model.solve(pulp.PULP_CBC_CMD(timeLimit=15))

        print(f"Generating adjacency matrix form of the scheduled contact plan")
        contact_plan = np.zeros((self.teg.K, self.teg.N, self.teg.N), dtype="int64")
        scheduled_contacts = []
        matched_edges_by_k = [[] for _ in range(self.teg.K)]
        for edge in self.edges.keys():
            if self.edges[edge].value() == 1.0:
                k, tx_node, rx_node = edge
                tx_idx = self.teg.node_map[tx_node]
                rx_idx = self.teg.node_map[rx_node]
                
                contact_plan[k][tx_idx][rx_idx] = 1
                contact_plan[k][rx_idx][tx_idx] = 1
                
                matched_edges_by_k[k].append((tx_node, rx_node))
        
        for k in range(len(matched_edges_by_k)):
            matched_edges = set(matched_edges_by_k[k])
            contacts = [contact for contact in self.teg.contacts[k] if (contact.tx_node, contact.rx_node) in matched_edges or (contact.rx_node, contact.tx_node) in matched_edges]
            scheduled_contacts.append(contacts)
        
        return TimeExpandedGraph(
            graphs=contact_plan,
            contacts=scheduled_contacts,
            state_durations=self.teg.state_durations,
            K=self.teg.K,
            N=self.teg.N,
            nodes=self.teg.nodes,
            node_map=self.teg.node_map,
            ipn_node_to_planet_map=self.teg.ipn_node_to_planet_map,
            W=self.teg.W,
            pos=self.teg.pos)
    
    def retargeting_delay(self, edge):
        # For each edge, take the previous k, and for both i and j, see if they were selected for an edge in the
        # previous k state. If not then assume retargeting delay is 0. If it was then use the position data to compute
        # the retargeting delay.
        # TODO finish this impl and probably create another pre-computation for getting seeing the edges in the last state
        # so we don't have to search through everything
        return 0

    def flow(self, i, j):
        edges = list(set(self.edges_by_node[i]) & set(self.edges_by_node[j]))
        bit_rate = min(self.bit_rates[i], self.bit_rates[j])
        
        return sum([self.edges[edge] * (self.T[edge[0]] - self.retargeting_delay(edge)) * bit_rate for edge in edges])

    def ect(self, i):
        """
        In order to make the schedule fair to all the nodes we can use the enabled contact time (ECT) for each inflow
        edge.
        """
        selected_edges = [edge for edge in self.edges.keys() if i in edge]
        # T[edge[0]] gives the duration of the edge
        return sum([self.edges[edge] * self.T[edge[0]] for edge in selected_edges])


if __name__ == "__main__":
    use_reduction = True
    EXPERIMENT_NAME = "gs_mars_earth_s_scenario"

    contact_plan_parser = IONContactPlanParser()
    contact_plan = contact_plan_parser.read(EXPERIMENT_NAME)

    initial_teg = convert_contact_plan_to_time_expanded_graph(
        contact_plan,
        should_fractionate=True)
    teg = dag_reduction(initial_teg) if use_reduction else initial_teg

    solver = LLSModel(teg)
    scheduled_teg = solver.solve()

    write_time_expanded_graph(EXPERIMENT_NAME, scheduled_teg, FileType.TEG_SCHEDULED)
    print("Finished contact scheduling")

    # Convert the TEG back to a contact plan
    scheduled_contact_plan = convert_time_expanded_graph_to_contact_plan(scheduled_teg)
    contact_plan_parser.write(EXPERIMENT_NAME, scheduled_contact_plan, FileType.SCHEDULED)
    print("Finished converting time expanded graph to contact plan")

    # Write contact plan to disk as IPN-D contact plan, so we can visualize the output
    ipnd_contact_plan_parser = IPNDContactPlanParser()
    ipnd_contact_plan_parser.write(EXPERIMENT_NAME, scheduled_contact_plan)

    reporter = Reporter(write_pkl=False)
    reporter.generate_report(
        EXPERIMENT_NAME,
        "LLSModel",
        10,
        scheduled_teg)
    reporter.write_report()
