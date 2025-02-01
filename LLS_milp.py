import pulp
import numpy as np

import constants
from constants import RELAY_NODES, SOURCE_NODES, DESTINATION_NODES
from contact_plan import IONContactPlanParser, IPNDContactPlanParser
from dag_topology_reduction import dag_reduction
from pat_delay_model import pat_delay
from report_generator import Reporter
from time_expanded_graph import convert_contact_plan_to_time_expanded_graph, TimeExpandedGraph, \
    write_time_expanded_graph, convert_time_expanded_graph_to_contact_plan
from utils import FileType

IS_MIP = False
MAX_TIME = 8 * 60 * 60  # seconds
MAX_EDGES_PER_LASER = 1

class LLSModel:
    def __init__(self, teg: TimeExpandedGraph):
        self.teg = teg
        self.edges = None
        self.edges_by_node = None
        self.edges_by_state = None
        self.edges_by_state_oi = None
        self.retargeting_delays = None
        self.schedule_duration = sum(self.teg.state_durations)
        self.T = self.teg.state_durations
        
        self.model = None

    def solve(self):
        # The contact plan topology here should be in the form of a list of tuples (state idx, i, j)
        contact_topology = []
        for k in range(self.teg.K):
            for tx_oi_idx in range(self.teg.N):
                for rx_oi_idx in range(self.teg.N):
                    if self.teg.graphs[k][tx_oi_idx][rx_oi_idx] >= 1:
                        contact_topology.append((k, tx_oi_idx, rx_oi_idx))
                        
        print(f"Creating binary variables for {len(contact_topology)} number of edges")
        # This represents the constraint that a selected edge must be a part of the initial contact plan
        if IS_MIP:
            self.edges = pulp.LpVariable.dicts(
                "edges", contact_topology, lowBound=0, upBound=1, cat=pulp.LpInteger
            )
        else:
            self.edges = pulp.LpVariable.dicts(
                # Pure LP problem, by relaxing the MIP into a pure LP problem we can remove the
                # integrality constraint of the decision variable, then use a threshold and a validation to assert that the
                # solution remains feasible.
                "edges", contact_topology, lowBound=0, upBound=1, cat=pulp.LpContinuous
            )

        print(f"Creating variables for retargeting delays")
        self.retargeting_delays = pulp.LpVariable.dicts(
            "retargeting_delays", contact_topology, lowBound=0, cat=pulp.LpContinuous
        )
        
        # In order to create these constraints faster we will first pre-process the data into a dict such that we can
        # isolate the edges by state k by node
        # the dict will have the key be the node name, which contains a list of length k, where each element is a list
        # of edges
        print(f"Pre-computing edge dictionary")
        self.edges_by_state = {node: [[] for _ in range(self.teg.K)] for node in self.teg.nodes}
        self.edges_by_node = {node: [] for node in self.teg.nodes}
        self.edges_by_state_oi = {oi: [[] for _ in range(self.teg.K)] for oi in self.teg.optical_interfaces_to_node}
        for edge in self.edges:
            k, tx_oi_idx, rx_oi_idx = edge
            
            self.edges_by_state_oi[tx_oi_idx][k].append(edge)
            self.edges_by_state_oi[rx_oi_idx][k].append(edge)

            tx_node = self.teg.nodes[self.teg.optical_interfaces_to_node[tx_oi_idx]]
            rx_node = self.teg.nodes[self.teg.optical_interfaces_to_node[rx_oi_idx]]
            
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

        self.flow_model = pulp.LpProblem("Network_flow_model", pulp.LpMaximize)
        
        print(f"Initializing the objective function")
        # objective function should maximize the summation of the capacity for each relay satellite and dst ground
        # stations
        # The negative term applies downward pressure on the number of edges selected, so we don't take unnecessary edges
        # due to the upward pressure from the Big-M retargeting constraint
        self.flow_model += pulp.lpSum(capacities.values()) + pulp.lpSum(single_hop_capacities.values()) - pulp.lpSum(self.edges.values())

        for relay_node, capacity in capacities.items():
            print(f"Setting up inflow and outflow capacity constraints for node {relay_node}")
            # inflow
            self.flow_model += capacity <= pulp.lpSum([self.flow(i, relay_node)
                                                  for i in self.teg.nodes if i in SOURCE_NODES])
            # outflow
            self.flow_model += capacity <= pulp.lpSum([self.flow(relay_node, x)
                                                  for x in self.teg.nodes if x in DESTINATION_NODES])
        
        # Create new inflow and outflow capacity constraints for single hop
        for gs_node, capacity in single_hop_capacities.items():
            print(f"Setting up inflow and outflow capacity constraints for ground station node {gs_node}")
            
            # We only have inflow here, outflow is basically infinite since its the dst node. We don't look at inflow
            # from the relay nodes either since that capacity is already accounted for in the two hop capacity
            # constraint above.
            self.flow_model += capacity <= pulp.lpSum([self.flow(i, gs_node)
                                                 for i in self.teg.nodes if i in SOURCE_NODES])

        # Constraint for fairness of source nodes
        print(f"Setting up fairness constraints based on ECT for source nodes")
        source_node_ect_dict = {source_node: self.ect(source_node) for source_node in self.teg.nodes if source_node in SOURCE_NODES}
        avg_ect = pulp.lpSum([source_node_ect_dict.values()]) / len(source_node_ect_dict) - 1000
        for ect in source_node_ect_dict.values():
            self.flow_model += ect >= avg_ect

        # Constraint that each optical interface can only be a part of a single selected edge per state
        print(f"Setting up 1 to 1 relationship between nodes per state k")
        for oi in self.teg.optical_interfaces_to_node:
            for k in range(self.teg.K):
                self.flow_model += pulp.lpSum([self.edges[edge] for edge in self.edges_by_state_oi[oi][k]]) <= MAX_EDGES_PER_LASER
        
        # Constraint for retargeting delay being greater than or equal to the tx and rx retargeting delays. These will
        # have downward pressure since as the retargeting delay decreases there is a higher effective contact time.
        print("Create retargeting delay constraints for tx and rx...")
        for edge in self.retargeting_delays:
            tx_delay, rx_delay = self.compute_retargeting_delay(edge)
            self.flow_model += self.retargeting_delays[edge] >= tx_delay
            self.flow_model += self.retargeting_delays[edge] >= rx_delay

        print("Starting solve...")
        self.flow_model.solve(pulp.PULP_CBC_CMD(timeLimit=MAX_TIME))

        print(f"Generating adjacency matrix from the scheduled contact plan")
        contact_plan = np.zeros((self.teg.K, self.teg.N, self.teg.N), dtype="int64")
        scheduled_contacts = []
        matched_edges_by_k = [[] for _ in range(self.teg.K)]
        # y = (dict(sorted(self.edges.items(), key=lambda x: x[1].value(), reverse=True)))
        # for k, v in y.items():
        #     print(k, v.value())
        for edge in dict(sorted(self.edges.items(), key=lambda x: x[1].value(), reverse=True)):
            # if self.edges[edge].value() != 0.0 and self.edges[edge].value() != 1.0:
            #     print(self.edges[edge].value())
            if self.is_edge_selected(edge, contact_plan):
                k, tx_oi_idx, rx_oi_idx = edge
                
                contact_plan[k][tx_oi_idx][rx_oi_idx] = 1
                contact_plan[k][rx_oi_idx][tx_oi_idx] = 1

                tx_node = self.teg.nodes[self.teg.optical_interfaces_to_node[tx_oi_idx]]
                rx_node = self.teg.nodes[self.teg.optical_interfaces_to_node[rx_oi_idx]]
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
            pos=self.teg.pos,
            optical_interfaces_to_node=self.teg.optical_interfaces_to_node,
            node_to_optical_interfaces=self.teg.node_to_optical_interfaces,
        )
    
    def compute_retargeting_delay(self, edge):
        # For each edge, take the previous k, and for both i and j, see if they were selected for an edge in the
        # previous k state. If not then assume retargeting delay is 0. If it was then use the position data to compute
        # the retargeting delay.
        k, tx_oi_idx, rx_oi_idx = edge
        k = min(k, len(self.teg.pos) - 1)
        
        if k == 0:
            return 0, 0
        
        def get_delay(node, new_node, oi_idx):  # retargeting_delay for prev edge
            # Create a dict for each previous edge get retargeting delay
            prev_edge_delays = {prev_edge: pat_delay(
                np.array([
                    np.array(self.teg.pos[k][self.teg.node_map[node]]),
                    np.array(self.teg.pos[k][self.teg.optical_interfaces_to_node[prev_edge[1] if prev_edge[2] == oi_idx else prev_edge[2]]]),
                    np.array(self.teg.pos[k][self.teg.node_map[new_node]])
                ]),
                np.array([
                    np.array(self.teg.pos[k][self.teg.node_map[node]]),
                    np.array(self.teg.pos[k][self.teg.optical_interfaces_to_node[prev_edge[1] if prev_edge[2] == oi_idx else prev_edge[2]]]),
                    np.array(self.teg.pos[k][self.teg.node_map[new_node]])
                ]),
            ) for prev_edge in self.edges_by_state_oi[oi_idx][k-1]}
            return sum([self.edges[prev_edge] * prev_edge_delays[prev_edge] for prev_edge in self.edges_by_state_oi[oi_idx][k-1]])
        
        tx_node = self.teg.nodes[self.teg.optical_interfaces_to_node[tx_oi_idx]]
        rx_node = self.teg.nodes[self.teg.optical_interfaces_to_node[rx_oi_idx]]
        return get_delay(tx_node, rx_node, tx_oi_idx), get_delay(rx_node, tx_node, rx_oi_idx)
                
    def flow(self, i, j):
        edges = list(set(self.edges_by_node[i]) & set(self.edges_by_node[j]))
        bit_rate = min(constants.BIT_RATES[i], constants.BIT_RATES[j])
        
        # Apply our retargeting delay as a Big-M constraint here so that we maintain linearity in our model.
        for edge in edges:
            # Upward pressure on this constraint if the edge is selected, otherwise 0 will be less than the resulting value
            self.flow_model += self.edges[edge] * self.T[edge[0]] * bit_rate <= (self.T[edge[0]] - self.retargeting_delays[edge]) * bit_rate
            
        return sum([self.edges[edge] * self.T[edge[0]] * bit_rate for edge in edges])

    def ect(self, i):
        """
        In order to make the schedule fair to all the nodes we can use the enabled contact time (ECT) for each inflow
        edge.
        """
        # T[edge[0]] gives the duration of the edge
        return sum([self.edges[edge] * self.T[edge[0]] for edge in self.edges_by_node[i]])

    def is_edge_selected(self, edge, contact_plan):
        if IS_MIP:
            return self.edges[edge].value() == 1.0
        else:
            if self.edges[edge].value() <= 0.1:
                return False

            # if self.edges[edge].value() > 0.5:
            #     return True

            # Here we want to check that the solution is still feasible if we add the edge
            k, tx_oi_idx, rx_oi_idx = edge
            is_tx_good = sum(contact_plan[k][tx_oi_idx]) < 1
            is_rx_good = sum(contact_plan[k][:, rx_oi_idx]) < 1
            is_rx2_good = sum(contact_plan[k][rx_oi_idx]) < 1
            is_tx2_good = sum(contact_plan[k][:, tx_oi_idx]) < 1

            return is_tx_good and is_rx_good and is_tx2_good and is_rx2_good

if __name__ == "__main__":
    use_reduction = True
    EXPERIMENT_NAME = "gs_mars_earth_l_scenario"

    contact_plan_parser = IONContactPlanParser()
    contact_plan = contact_plan_parser.read(EXPERIMENT_NAME)

    initial_teg = convert_contact_plan_to_time_expanded_graph(
        contact_plan,
        should_fractionate=True)
    teg = dag_reduction(initial_teg) if use_reduction else initial_teg

    solver = LLSModel(teg)
    scheduled_teg = solver.solve()

    for k in range(scheduled_teg.K):
        for tx_idx in range(scheduled_teg.N):
            row_count = sum(scheduled_teg.graphs[k][tx_idx])
            assert row_count <= 1, f"{scheduled_teg.graphs[k]}"

        for rx_idx in range(scheduled_teg.N):
            row_count = sum(scheduled_teg.graphs[k][:, rx_idx])
            assert row_count <= 1, f"{scheduled_teg.graphs[k]}"

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
