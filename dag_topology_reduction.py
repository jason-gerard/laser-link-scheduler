import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from contact_plan import IONContactPlanParser
from time_expanded_graph import convert_contact_plan_to_time_expanded_graph, dag_reduction, count_edges

SHOW_FIGS = True

EXPERIMENT_NAME = "mars_earth_xs_scenario"


def main():
    contact_plan_parser = IONContactPlanParser()
    contact_plan = contact_plan_parser.read(EXPERIMENT_NAME)

    teg = convert_contact_plan_to_time_expanded_graph(
        contact_plan,
        should_fractionate=True)
    
    reduced_teg = dag_reduction(teg)
    
    print(count_edges(teg), count_edges(reduced_teg))
    print(f"Percent of edges removed = {100 * (1 - count_edges(reduced_teg) / count_edges(teg)):.3f}%")

    if SHOW_FIGS:
        visualize(teg, name="teg")
        visualize(reduced_teg, name="reduced_teg")
        # plt.show()


def visualize(teg, name):
    rand = np.random.randint(1, 10)
    for k in range(1):
        num_nodes = teg.N

        edges = []
        for tx_idx in range(num_nodes):
            for rx_idx in range(num_nodes):
                if teg.graphs[k][tx_idx][rx_idx] >= 1:
                    edges.append((teg.nodes[tx_idx], teg.nodes[rx_idx]))

        G = nx.DiGraph()
        for node in teg.nodes:
            G.add_node(node)
        G.add_edges_from(edges)

        # print(teg.graphs[k])
        plt.figure(k + rand)
        # nx.draw(G, nx.spring_layout(G), node_size=1500, with_labels=False)
        A = nx.nx_agraph.to_agraph(G)
        A.layout(prog="dot")
        A.draw(f'{name}.png', args='-Gnodesep=0.01 -Gfont_size=1', prog='dot')


if __name__ == "__main__":
    main()
