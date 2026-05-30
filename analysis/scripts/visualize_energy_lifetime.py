"""
visualize_energy_lifetime.py

Loads scheduled TEGs from a report directory, replays the battery simulation
(using OCTConfig.AVG_TRANSMISSION_POWER, consistent with the scheduler),
computes energy-consumption and mission-lifetime metrics per
(algorithm, scenario), and saves one line-plot per metric following the same
style as visualize_metrics_incremental_runs.py.

Energy model assumptions (mirrors the scheduler exactly)
---------------------------------------------------------
- Power source  : RTG only — exponential decay P(t) = P0 · exp(−λ·t)
- Recharge      : none  (solar panel model is not yet implemented)
- Battery       : starts full; only a *deficit* from (baseline + tx − RTG)
                  is drawn from the battery; surpluses are discarded
- Tx power      : OCTConfig.AVG_TRANSMISSION_POWER (4 W) × effective contact
                  duration (pointing+acquisition delay already subtracted)

Plots produced
--------------
  total_tx_energy_wh     — Network-wide transmission energy overhead [Wh]
  tx_energy_per_node_wh  — Same, normalised per spacecraft node
  mean_final_battery_pct — Average remaining battery at end of mission [%]
  min_final_battery_pct  — Minimum (worst-case node) remaining battery [%]
  depleted_node_count    — Number of spacecraft nodes that hit 0 % battery

Usage
-----
    python analysis/scripts/visualize_energy_lifetime.py --report-id <ID>
    python analysis/scripts/visualize_energy_lifetime.py --report-id <ID> --debug
"""

import csv
import math
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import typer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.scripts.utils import (  # noqa: E402
    AnalysisRunTable,
    load_report_tegs,
    normalize_algorithm_name,
)
from src.constants import (
    DESTINATION_NODES,
    MLConfig,
    OCTConfig,
    PLOTS_ROOT,
    RELAY_NODES,
    SOURCE_NODES,
)  # noqa: E402
from src.models.energy_comsumption import (
    transmission_energy as compute_tx_energy,
)  # noqa: E402
from src.models.mission_lifetime import generated_energy  # noqa: E402
from src.time_expanded_graph.time_expanded_graph import TimeExpandedGraph  # noqa: E402
from src.topology.weights import compute_effective_contact_time  # noqa: E402

# ---------------------------------------------------------------------------
# Plot style — identical to visualize_metrics_incremental_runs.py
# ---------------------------------------------------------------------------
plt.rcParams.update({"font.size": 18})
plt.rc("legend", fontsize=14)
plt.rcParams.update({"font.family": "Times New Roman"})

_J_TO_WH = 1.0 / 3600.0
_DESTINATION_IDS = set(DESTINATION_NODES)
_SOURCE_IDS = set(SOURCE_NODES)
_RELAY_IDS = set(RELAY_NODES)

app = typer.Typer()

# ---------------------------------------------------------------------------
# Algorithm display config — mirrors visualize_metrics_incremental_runs.py
# (algorithm_key, display_name, linestyle, linewidth, color)
# ---------------------------------------------------------------------------
algorithms = [
    # Baselines
    ("lls", "LLS_Greedy", "solid", 2.5, None),
    ("fcp", "FCP", "solid", 2.5, None),
    # Contributions
    ("energy_aware", "Energy-Aware", "solid", 2.5, None),
    ("battery_energy", "Battery-Aware", "dashed", 2.5, None),
    ("lifespan_aware", "Lifespan-Aware", "dotted", 2.5, None),
]

# (metric_key, label, unit, y_min, y_max, y_step)
# y_max = None  →  matplotlib auto-scale with floor at y_min
metrics = [
    # --- Energy ---
    ("total_tx_energy_wh", "Total transmission energy", "Wh", 0, None, None),
    (
        "tx_energy_per_node_wh",
        "Transmission energy per node",
        "Wh/node",
        0,
        None,
        None,
    ),
    # --- Battery / lifetime ---
    ("mean_final_battery_pct", "Mean final battery", "%", None, None, 10),
    ("min_final_battery_pct", "Min final battery", "%", None, None, 10),
    (
        "depleted_node_count",
        "Nodes with depleted battery",
        "nodes",
        0,
        None,
        None,
    ),
]


# ---------------------------------------------------------------------------
# Battery simulation
# ---------------------------------------------------------------------------


def _simulate_battery(
    teg: TimeExpandedGraph,
    should_bypass_retargeting_time: bool,
) -> dict:
    """
    Replay a *scheduled* TEG through the same energy model used by the
    scheduler and return a summary dict with energy and battery metrics.

    The logic intentionally mirrors LifespanAware._update_battery_states():
      - For every active TX optical-interface in each state k, compute the
        effective contact duration (after retargeting delay), multiply by
        AVG_TRANSMISSION_POWER, and apply the resulting delta to the battery.
      - Surplus RTG energy is NOT stored (solar recharge = 0).
      - Battery is clamped to [0, max_capacity] after each state.

    Ground-station nodes (DESTINATION_NODES) have infinite power and are
    excluded from all energy and battery tracking.
    """
    node_ids = np.array([node.id for node in teg.nodes])
    oi_to_node_idx = np.array(
        [teg.optical_interfaces_to_node[i] for i in range(teg.N)], dtype=int
    )
    is_spacecraft = np.array(
        [node.id not in _DESTINATION_IDS for node in teg.nodes]
    )

    initial_powers = MLConfig.get_initial_powers(node_ids)
    baseline_powers = MLConfig.get_baseline_powers(node_ids)
    battery_max = MLConfig.get_initial_batteries(node_ids) * 3600.0  # Wh → J
    battery = battery_max.copy()

    # Accumulated transmission energy per node (J)
    total_tx_energy_j = np.zeros(len(teg.nodes))
    accumulated_time = 0.0

    for state in range(teg.K):
        duration = int(teg.state_durations[state])
        from_time = accumulated_time
        to_time = accumulated_time + duration

        baseline_energy_j = baseline_powers * duration
        rtg_generated_j = generated_energy(
            from_time=from_time,
            to_time=to_time,
            initial_power=initial_powers,
            decay_constant=MLConfig.DECAY_RATE,
        )

        next_battery = battery.copy()

        active_tx, active_rx = np.where(teg.graphs[state] >= 1)
        for tx_oi, rx_oi in zip(active_tx.tolist(), active_rx.tolist()):
            tx_node = oi_to_node_idx[tx_oi]
            if not is_spacecraft[tx_node]:
                continue  # Ground stations have infinite power — skip

            eff_dur = compute_effective_contact_time(
                oi_idx1=tx_oi,
                oi_idx2=rx_oi,
                scheduled_contact_topology=teg.graphs[:state],
                state_duration=duration,
                positions=teg.pos,
                optical_interfaces_to_node=teg.optical_interfaces_to_node,
                nodes=teg.nodes,
                should_bypass_retargeting_time=should_bypass_retargeting_time,
            )
            tx_cost = compute_tx_energy(
                power=OCTConfig.AVG_TRANSMISSION_POWER, duration=eff_dur
            )
            total_tx_energy_j[tx_node] += tx_cost

            # delta = RTG_generated − (tx_cost + baseline)
            # Mirror the scheduler: only deficits drain the battery;
            # surpluses are discarded because solar recharge = 0.
            delta = rtg_generated_j[tx_node] - (
                tx_cost + baseline_energy_j[tx_node]
            )
            if delta < 0:
                next_battery[tx_node] += delta

        # Clamp battery to [0, max_capacity]
        finite = np.isfinite(next_battery)
        next_battery[finite] = np.clip(
            next_battery[finite], 0.0, battery_max[finite]
        )
        battery = next_battery
        accumulated_time += duration

    # --- Summary ---
    sc_count = int(is_spacecraft.sum())
    # Only spacecraft nodes with a finite (non-inf) battery capacity
    finite_sc = is_spacecraft & np.isfinite(battery_max) & (battery_max > 0)

    total_tx_wh = float(total_tx_energy_j[is_spacecraft].sum() * _J_TO_WH)
    tx_per_node_wh = total_tx_wh / max(sc_count, 1)

    if finite_sc.any():
        final_bat = battery[finite_sc]
        max_bat = battery_max[finite_sc]
        pct = final_bat / max_bat * 100.0
        mean_pct = float(pct.mean())
        min_pct = float(pct.min())
        n_depleted = int((final_bat <= 0).sum())
    else:
        mean_pct = 100.0
        min_pct = 100.0
        n_depleted = 0

    return {
        "spacecraft_count": sc_count,
        "total_tx_energy_wh": total_tx_wh,
        "tx_energy_per_node_wh": tx_per_node_wh,
        "mean_final_battery_pct": mean_pct,
        "min_final_battery_pct": min_pct,
        "depleted_node_count": n_depleted,
    }


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------


def run_analysis(report_id: int, plain_progress: bool = False) -> None:
    tegs = load_report_tegs(report_id)

    rows_init = [
        {
            "algorithm": alg,
            "scenario": scn,
            "nodes": str(len(teg.node_map)),
            "states": str(teg.K),
            "progress": "-",
        }
        for alg, scn, teg in tegs
    ]
    columns = [
        ("algorithm", {"no_wrap": True}),
        ("scenario", {"no_wrap": True}),
        ("nodes", {"justify": "right", "no_wrap": True}),
        ("states", {"justify": "right", "no_wrap": True}),
        ("progress", {"no_wrap": True}),
    ]

    report: list[dict] = []

    with AnalysisRunTable(
        columns, rows_init, enable_live=not plain_progress
    ) as run_table:
        for idx, (algorithm, scenario, teg) in enumerate(tegs):
            run_table.mark_progress(idx, 0)

            num_nodes = int(scenario.split("_")[-1])
            should_bypass = (
                normalize_algorithm_name(algorithm) == "lls_pat_unaware"
            )
            source_count = sum(1 for n in teg.nodes if n.id in _SOURCE_IDS)
            relay_count = sum(1 for n in teg.nodes if n.id in _RELAY_IDS)

            sim = _simulate_battery(teg, should_bypass)
            report.append(
                {
                    "Algorithm": normalize_algorithm_name(algorithm),
                    "Scenario": scenario,
                    "num_nodes": num_nodes,
                    "source_count": source_count,
                    "relay_count": relay_count,
                    **sim,
                }
            )
            run_table.mark_progress(idx, 100)

    if not report:
        print("No TEGs found in report directory.")
        return

    plot_dir = os.path.join(PLOTS_ROOT, str(report_id))
    os.makedirs(plot_dir, exist_ok=True)

    # Save aggregated CSV inside the report folder
    csv_path = os.path.join(plot_dir, "energy_lifetime.csv")
    fieldnames = list(report[0].keys())
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report)

    x = sorted(set(run["num_nodes"] for run in report))

    # Map total-node-count → "source/relay" label using actual TEG node counts
    node_count_to_label: dict[int, str] = {}
    for r in report:
        nk = r["num_nodes"]
        if nk not in node_count_to_label:
            node_count_to_label[nk] = f"{r['source_count']}/{r['relay_count']}"

    # -----------------------------------------------------------------------
    # Plots — one figure per metric, same loop structure as the template
    # -----------------------------------------------------------------------
    for metric_key, label, unit, y_min, y_max, y_step in metrics:
        fig = plt.figure()
        ax = fig.add_subplot(111)

        for alg_key, display_name, linestyle, linewidth, color in algorithms:
            y = [r[metric_key] for r in report if r["Algorithm"] == alg_key]
            if not y:
                continue
            kwargs = dict(
                label=display_name, linewidth=linewidth, linestyle=linestyle
            )
            if color:
                kwargs["color"] = color
            plt.plot(x[: len(y)], y, **kwargs)

        full_label = f"{label} [{unit}]" if unit else label
        plt.ylabel(full_label)
        plt.xlabel("Source/relay node counts")
        plt.legend()
        plt.grid(linestyle="-", color="0.95")

        if y_max is not None and y_step is not None and y_min is not None:
            plt.ylim(max(y_min - y_step, 0), y_max)
            ax.set_yticks(
                [y_min] + np.arange(y_step, y_max + 0.01, y_step).tolist()
            )
        else:
            ax.autoscale(axis="y")
            plt.ylim(bottom=y_min)

        # Thin ticks when there are too many x values (keep ≤ 12)
        x_ticks = x if len(x) <= 12 else x[:: math.ceil(len(x) / 12)]
        x_labels = [node_count_to_label.get(v, str(v)) for v in x_ticks]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)

        file_name = metric_key
        plt.savefig(
            os.path.join(plot_dir, f"{file_name}.pdf"),
            format="pdf",
            bbox_inches="tight",
        )
        plt.savefig(
            os.path.join(plot_dir, f"{file_name}.png"),
            format="png",
            bbox_inches="tight",
            dpi=300,
        )
        plt.close(fig)


@app.command()
def main(
    report_id: int = typer.Option(
        ...,
        "--report-id",
        "-r",
        help="Report identifier used to read scheduled TEGs from reports/.",
    ),
    plain_progress: bool = typer.Option(
        False,
        "--debug",
        help="Disable the live run table. Useful when debugging with pdb/ipdb.",
    ),
) -> None:
    run_analysis(report_id, plain_progress)


if __name__ == "__main__":
    app()
