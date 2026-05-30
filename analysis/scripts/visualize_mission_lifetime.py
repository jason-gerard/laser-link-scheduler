"""
mission_lifetime_analysis.py

Loads scheduled TEGs from a report directory, replays the battery simulation
(using OCTConfig.AVG_TRANSMISSION_POWER, consistent with the scheduler),
then estimates how long the relay fleet can keep transmitting after the
scheduled mission ends.

Why relay nodes only
--------------------
  Source nodes feed data into the relay network; relay nodes forward it to
  Earth ground stations.  When ALL relays are dead no data can reach Earth
  regardless of how many source nodes are still alive.  Therefore:

      system_lifetime = min over all RELAY nodes of (estimated node lifetime)

  (i.e. the system keeps working until the weakest relay dies.)

Lifetime model — realistic future TX load
------------------------------------------
  After the last scheduled state each relay node's remaining lifetime is
  estimated assuming it continues to transmit at the *same average rate* it
  had during the mission:

      avg_tx_power   = total_tx_energy_J / mission_duration_s   [W]
      required_power = P_baseline + avg_tx_power                [W]

      rtg_time_left  = max(mission_lifetime(P0, λ, required_power) − T, 0)
      battery_buffer = final_battery_J / required_power

      t_relay = T_mission + rtg_time_left + battery_buffer

  Using the actual TX rate (instead of zero) gives a realistic answer to
  "how many more days can the relay keep forwarding data at this schedule?"

Plots produced
--------------
  min_relay_lifetime_days  — Lifetime of the weakest relay [days]
  mean_relay_lifetime_days — Average relay lifetime [days]
  Both include a dashed reference line at the scenario duration.

Usage
-----
    python analysis/scripts/mission_lifetime_analysis.py --report-id <ID>
    python analysis/scripts/mission_lifetime_analysis.py --report-id <ID> --debug
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
from src.models.mission_lifetime import generated_energy, mission_lifetime  # noqa: E402
from src.time_expanded_graph.time_expanded_graph import TimeExpandedGraph  # noqa: E402
from src.topology.weights import compute_effective_contact_time  # noqa: E402

# ---------------------------------------------------------------------------
# Plot style — identical to visualize_metrics_incremental_runs.py
# ---------------------------------------------------------------------------
plt.rcParams.update({"font.size": 18})
plt.rc("legend", fontsize=14)
plt.rcParams.update({"font.family": "Times New Roman"})

_S_TO_DAYS = 1.0 / 86400.0
_DESTINATION_IDS = set(DESTINATION_NODES)
_SOURCE_IDS = set(SOURCE_NODES)
_RELAY_IDS = set(RELAY_NODES)

app = typer.Typer()

# ---------------------------------------------------------------------------
# Algorithm display config
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

# (metric_key, label, unit)  — y-axis is always auto-scaled
metrics = [
    ("min_relay_lifetime_days", "Min relay lifetime", "days"),
    ("mean_relay_lifetime_days", "Mean relay lifetime", "days"),
]


# ---------------------------------------------------------------------------
# Battery simulation + lifetime estimation
# ---------------------------------------------------------------------------


def _simulate_system_lifetime(
    teg: TimeExpandedGraph,
    should_bypass_retargeting_time: bool,
) -> dict:
    """
    1. Battery simulation (mirrors LifespanAware._update_battery_states):
       Replay all K states using AVG_TRANSMISSION_POWER; track both the
       final battery level and the total TX energy consumed per node.

    2. Per-relay lifetime estimate using realistic future TX load:
           avg_tx_power   = total_tx_energy_J / mission_duration_s
           required_power = P_baseline + avg_tx_power

           rtg_time_left  = max(mission_lifetime(P0, λ, required_power) − T, 0)
           battery_buffer = final_battery_J / required_power
           t_relay        = T + rtg_time_left + battery_buffer

       Using avg_tx_power answers "how long can this relay keep forwarding
       data at the same rate it maintained during the mission?"

    3. System-level aggregation over RELAY nodes only:
           system dies when the last relay dies
           → min_relay_lifetime = min(t_relay)  [weakest relay]
           → mean_relay_lifetime = mean(t_relay)

    Returns scalar metrics in seconds and days.
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

    # Accumulated TX energy per node (J) — needed for avg_tx_power
    total_tx_energy_j = np.zeros(len(teg.nodes))
    accumulated_time = 0.0

    # --- Battery simulation ---
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
                continue  # GS nodes have infinite power — skip

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

            # Deficit draws from battery; surplus is discarded (no solar)
            delta = rtg_generated_j[tx_node] - (
                tx_cost + baseline_energy_j[tx_node]
            )
            if delta < 0:
                next_battery[tx_node] += delta

        finite = np.isfinite(next_battery)
        next_battery[finite] = np.clip(
            next_battery[finite], 0.0, battery_max[finite]
        )
        battery = next_battery
        accumulated_time += duration

    mission_duration = accumulated_time  # seconds

    # Average TX power each node sustained during the mission
    avg_tx_power = total_tx_energy_j / max(mission_duration, 1.0)  # W

    # --- Relay node lifetime with realistic future TX load ---
    relay_lifetimes: list[float] = []

    for node_idx, node in enumerate(teg.nodes):
        if node.id not in _RELAY_IDS:
            continue  # only relay nodes count for system lifetime

        p_baseline = float(baseline_powers[node_idx])
        p_required = p_baseline + float(avg_tx_power[node_idx])

        if p_required <= 0:
            relay_lifetimes.append(np.inf)
            continue

        # Time until RTG drops below the required power level
        rtg_total = mission_lifetime(
            P0=float(initial_powers[node_idx]),
            decay_constant=MLConfig.DECAY_RATE,
            P_min=p_required,
        )
        rtg_remaining = max(rtg_total - mission_duration, 0.0)

        # How long the remaining battery sustains the required power
        battery_buffer = float(battery[node_idx]) / p_required  # seconds

        relay_lifetimes.append(
            mission_duration + rtg_remaining + battery_buffer
        )

    finite_lt = [lt for lt in relay_lifetimes if np.isfinite(lt)]
    min_lt = min(finite_lt) if finite_lt else 0.0
    mean_lt = sum(finite_lt) / len(finite_lt) if finite_lt else 0.0

    return {
        "mission_duration_days": mission_duration * _S_TO_DAYS,
        "min_relay_lifetime_days": min_lt * _S_TO_DAYS,
        "mean_relay_lifetime_days": mean_lt * _S_TO_DAYS,
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

            sim = _simulate_system_lifetime(teg, should_bypass)
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
    pdf_dir = os.path.join(plot_dir, "pdf")
    png_dir = os.path.join(plot_dir, "png")
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)

    # Save aggregated CSV
    csv_path = os.path.join(plot_dir, "relay_lifetime.csv")
    fieldnames = list(report[0].keys())
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report)

    x = sorted(set(run["num_nodes"] for run in report))
    # All scenarios in one report share the same mission duration
    mission_duration_days = report[0]["mission_duration_days"]

    # Map total-node-count → "source/relay" label using actual TEG node counts
    node_count_to_label: dict[int, str] = {}
    for r in report:
        nk = r["num_nodes"]
        if nk not in node_count_to_label:
            node_count_to_label[nk] = f"{r['source_count']}/{r['relay_count']}"

    # -----------------------------------------------------------------------
    # Plots — one figure per metric, same loop structure as the template
    # -----------------------------------------------------------------------
    for metric_key, label, unit in metrics:
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

        # Reference line: scenario duration
        plt.axhline(
            y=mission_duration_days,
            color="tab:red",
            linestyle="dashed",
            linewidth=2.0,
            label="Scenario duration",
        )

        full_label = f"{label} [{unit}]" if unit else label
        plt.ylabel(full_label)
        plt.xlabel("Source/relay node counts")
        plt.legend()
        plt.grid(linestyle="-", color="0.95")
        ax.relim()
        ax.autoscale_view()
        y_bot, y_top = ax.get_ylim()
        ax.set_ylim(bottom=y_bot, top=y_top * 1.05)

        # Thin ticks when there are too many x values (keep ≤ 12)
        x_ticks = x if len(x) <= 12 else x[:: math.ceil(len(x) / 12)]
        x_labels = [node_count_to_label.get(v, str(v)) for v in x_ticks]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)

        file_name = metric_key
        plt.savefig(
            os.path.join(pdf_dir, f"{file_name}.pdf"),
            format="pdf",
            bbox_inches="tight",
        )
        plt.savefig(
            os.path.join(png_dir, f"{file_name}.png"),
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
