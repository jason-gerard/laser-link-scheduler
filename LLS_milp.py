import pulp
import numpy as np

from constants import RELAY_NODES, SOURCE_NODES, DESTINATION_NODES
from contact_plan import IONContactPlanParser
from report_generator import Reporter
from time_expanded_graph import convert_contact_plan_to_time_expanded_graph, TimeExpandedGraph

bit_rate = 1000

max_matching = 2


class LLSModel:
    def __init__(self, experiment_name):
        contact_plan_parser = IONContactPlanParser()
        contact_plan = contact_plan_parser.read(experiment_name)

        self.teg = convert_contact_plan_to_time_expanded_graph(
            contact_plan,
            should_fractionate=True)
        
        self.edges = None
        self.schedule_duration = sum(self.teg.state_durations)
        self.T = self.teg.state_durations

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

        # This represents the constraint that a selected edge must be a part of the initial contact plan
        self.edges = pulp.LpVariable.dicts(
            "edges", contact_topology, lowBound=0, upBound=1, cat=pulp.LpInteger
        )

        flow_model = pulp.LpProblem("Network_flow_model", pulp.LpMaximize)

        capacities = {relay_node: pulp.LpVariable(f"Capacity_{relay_node}", lowBound=0)
                      for relay_node in self.teg.nodes if relay_node in RELAY_NODES}

        # objective function should maximize the summation of the capacity for each relay satellite
        flow_model += pulp.lpSum(capacities.values())

        for relay_node, capacity in capacities.items():
            # inflow
            flow_model += capacity <= pulp.lpSum([self.flow(i, relay_node) for i in self.teg.nodes if i in SOURCE_NODES])
            # outflow
            flow_model += capacity <= pulp.lpSum([self.flow(relay_node, x) for x in self.teg.nodes if x in DESTINATION_NODES])

        # constraint that each node can only be a part of a single selected edge per state
        for node in self.teg.nodes:
            for k in range(self.teg.K):
                flow_model += (
                    pulp.lpSum([self.edges[edge] for edge in self.edges.keys() if node in edge and k in edge]) <= 1,
                    f"Max_edge_{node}_{k}",
                )

        print("Starting solve...")
        flow_model.solve()

        contact_plan = np.zeros((self.teg.K, self.teg.N, self.teg.N), dtype="int64")
        for edge in self.edges.keys():
            if self.edges[edge].value() == 1.0:
                k, tx_node, rx_node = edge
                tx_idx = self.teg.node_map[tx_node]
                rx_idx = self.teg.node_map[rx_node]
                
                contact_plan[k][tx_idx][rx_idx] = 1
                contact_plan[k][rx_idx][tx_idx] = 1

        return TimeExpandedGraph(
            graphs=contact_plan,
            contacts=[],
            state_durations=self.teg.state_durations,
            K=self.teg.K,
            N=self.teg.N,
            nodes=self.teg.nodes,
            node_map=self.teg.node_map,
            ipn_node_to_planet_map=self.teg.ipn_node_to_planet_map,
            W=self.teg.W)

    def flow(self, i, j):
        selected_edges = [edge for edge in self.edges.keys() if i in edge and j in edge]
        return sum([self.edges[edge] * self.T[edge[0]] * bit_rate for edge in selected_edges])

    def dct(self, i):
        """
        In order to make the schedule fair to all the nodes we can use the disabled contact time (DCT) for each inflow edge
        by taking the total duration of the schedule minus the enabled contact time
        """
        selected_edges = [edge for edge in self.edges.keys() if i in edge]
        # T[edge[0]] gives the duration of the edge
        return sum([self.schedule_duration - self.edges[edge] * self.T[edge[0]] for edge in selected_edges])


if __name__ == "__main__":
    EXPERIMENT_NAME = "mars_earth_m_scenario"
    solver = LLSModel(EXPERIMENT_NAME)
    teg = solver.solve()

    reporter = Reporter(write_pkl=False)
    reporter.generate_report(
        EXPERIMENT_NAME,
        "LLSModel",
        10,
        teg)
    reporter.write_report()
