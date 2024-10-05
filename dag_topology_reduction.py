import matplotlib.pyplot as plt
import networkx as nx

from contact_plan import IONContactPlanParser
from time_expanded_graph import convert_contact_plan_to_time_expanded_graph

SHOW_FIGS = False

experiment_name = "mars_earth_xs_scenario"

contact_plan_parser = IONContactPlanParser()
contact_plan = contact_plan_parser.read(experiment_name)

teg = convert_contact_plan_to_time_expanded_graph(
    contact_plan,
    should_fractionate=True)

for k in range(10):
    num_nodes = teg.N

    edges = []
    for tx_idx in range(num_nodes):
        for rx_idx in range(num_nodes):
            if teg.graphs[k][tx_idx][rx_idx] >= 1:
                edges.append((teg.nodes[tx_idx], teg.nodes[rx_idx]))

    G = nx.Graph()
    for node in teg.nodes:
        G.add_node(node)
    G.add_edges_from(edges)

    # print(teg.graphs[k])
    plt.figure(k)
    nx.draw(G, with_labels=True)

if SHOW_FIGS:
    plt.show()
