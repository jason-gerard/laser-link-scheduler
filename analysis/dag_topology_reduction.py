import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import networkx as nx
import numpy as np

import os
import sys

sys.path.append(os.path.join(os.path.dirname(sys.path[0])))
from contact_plan import IONContactPlanParser
from time_expanded_graph import convert_contact_plan_to_time_expanded_graph, dag_reduction, fractionate_graph, count_edges

SHOW_FIGS = False

EXPERIMENT_NAME = "mars_earth_xs_scenario"

plt.rcParams.update({'font.size': 18})
plt.rc('legend', fontsize=14)
plt.rcParams.update({'font.family': 'Times New Roman'})

def count_reduced_edges(experiment_name):
    contact_plan_parser = IONContactPlanParser()
    contact_plan = contact_plan_parser.read(experiment_name)

    teg = convert_contact_plan_to_time_expanded_graph(
        contact_plan,
        should_fractionate=False,
        should_reduce=False)
    teg_count = count_edges(teg)
    
    frac_teg = fractionate_graph(teg)
    frac_teg_count = count_edges(frac_teg)
    
    reduced_teg = dag_reduction(frac_teg)
    reduced_teg_count = count_edges(reduced_teg)
    
    print(teg_count, frac_teg_count, reduced_teg_count)
    print(f"Percent of edges removed = {100 * (1 - count_edges(reduced_teg) / count_edges(frac_teg)):.3f}%")

    return teg_count, frac_teg_count, reduced_teg_count
    # if SHOW_FIGS:
    #     visualize(teg, name="teg")
    #     visualize(reduced_teg, name="reduced_teg")
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
    EXPERIMENT_NAMES = [
        "gs_mars_earth_xs_scenario",
        "gs_mars_earth_s_scenario",
        "gs_mars_earth_m_scenario",
        "gs_mars_earth_l_scenario",
        "gs_mars_earth_xl_scenario",
    ]

    x = [
        "X-Small",
        "Small",
        "Medium",
        "Large",
        "X-Large",
    ]
    
    teg_counts = []
    frac_teg_counts = []
    reduced_teg_counts = []
    for name in EXPERIMENT_NAMES:
        teg_count, frac_teg_count, reduced_teg_count = count_reduced_edges(name)
        teg_counts.append(teg_count)
        frac_teg_counts.append(frac_teg_count)
        reduced_teg_counts.append(reduced_teg_count)

    fig, ax1 = plt.subplots()

    ax1.plot(x, teg_counts, label="Standard TEG", linewidth=2.5)
    ax1.plot(x, frac_teg_counts, label="Fractionated TEG", linewidth=2.5)
    ax1.plot(x, reduced_teg_counts, label="Reduced TEG", linewidth=2.5)

    label = "Number of decision variables"
    ax1.set_ylabel(label)
    ax1.set_xlabel("Network Size")

    ax1.set_yscale('log')

    plt.grid(linestyle='-', color='0.95')

    
    ax2 = ax1.twinx()

    y2 = [float(f"{100 * (1 - reduced_teg_counts[i] / frac_teg_counts[i]): .3f}") for i in range(len(teg_counts))]
    ax2.plot(x, y2, "tab:cyan", linestyle="dashed", label="DAG Reduction")
    ax2.set_ylabel("Percent reduction")
    ax2.set_ylim(0, 100)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter())

    # Get the handles and labels from both axes
    handles, labels = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    # Combine the handles and labels
    all_handles = handles + handles2
    all_labels = labels + labels2

    # Create a single legend
    plt.legend(all_handles, all_labels)

    file_name = label.replace(" ", "_").replace("/", "_")
    plt.savefig(
        os.path.join("analysis", f"{file_name}.pdf"),
        format="pdf",
        bbox_inches="tight"
    )
    plt.savefig(
        os.path.join("analysis", f"{file_name}.png"),
        format="png",
        bbox_inches="tight",
        dpi=300,
    )
