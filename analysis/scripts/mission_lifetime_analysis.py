import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.topology.weights import EFFECTIVE_CONTACT_TIME_CACHE, COORDINATE_CACHE  # noqa: E402
from analysis.scripts.utils import (  # noqa: E402
    AnalysisRunTable,
    compute_lifetime_metrics,
    load_report_tegs,
    normalize_algorithm_name,
    summarize_lifetime_metrics,
)
from src.constants import PLOTS_ROOT, MLConfig  # noqa: E402


plt.rcParams.update({"font.size": 18})
plt.rc("legend", fontsize=14)
plt.rcParams.update({"font.family": "Times New Roman"})

PILOT_SCENARIOS = [
    "gs_mars_earth_scenario_inc_16",
    "gs_mars_earth_scenario_inc_32",
    "gs_mars_earth_scenario_inc_48",
]
VARIANT_ORDER = [
    "lifespan_aware_v0",
    "lifespan_aware_v1",
    "lifespan_aware_v2",
]
VARIANT_LABELS = {
    "lifespan_aware": "lifespan_aware_v0",
    "lifespan_aware_v0": "lifespan_aware_v0",
    "lifespan_aware_v1": "lifespan_aware_v1",
    "lifespan_aware_v2": "lifespan_aware_v2",
}

app = typer.Typer()


def _safe_float_str(value: float) -> str:
    if np.isinf(value):
        return "inf"
    return f"{value:.4f}"


def _save_node_lifetime_plot(
    metrics_df: pd.DataFrame,
    scenario: str,
    algorithm: str,
) -> None:
    plot_dir = os.path.join(PLOTS_ROOT, scenario, algorithm)
    os.makedirs(plot_dir, exist_ok=True)

    sorted_df = metrics_df.sort_values("node_id").reset_index(drop=True)
    finite_lifetimes = sorted_df.loc[
        np.isfinite(sorted_df["estimated_lifetime_years"]),
        "estimated_lifetime_years",
    ]
    mission_duration_years = float(
        sorted_df["mission_duration"].iloc[0] / (356.25 * 24 * 60 * 60)
    )
    max_finite_lifetime = (
        float(finite_lifetimes.max()) if not finite_lifetimes.empty else 0.0
    )
    clip_value = max(max_finite_lifetime, mission_duration_years) * 1.05
    if clip_value <= 0:
        clip_value = 1.0

    plot_values = sorted_df["estimated_lifetime_years"].where(
        np.isfinite(sorted_df["estimated_lifetime_years"]),
        clip_value,
    )
    colors = np.where(
        np.isfinite(sorted_df["estimated_lifetime_years"]),
        "tab:blue",
        "tab:green",
    )
    average_power_load = (
        sorted_df["average_power_load"]
        + MLConfig.BASELINE_POWER_FOR_BASIC_OPERATION
    )
    final_generated_power = sorted_df["final_generated_power"]
    x_positions = np.arange(len(sorted_df))

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111)
    ax.bar(x_positions, plot_values, color=colors, width=0.75)
    ax.axhline(
        y=mission_duration_years,
        color="tab:red",
        linestyle="--",
        linewidth=2,
        label="Mission duration",
    )
    ax.set_ylabel("Estimated lifetime [y]")
    ax.set_xlabel("Spacecraft node ID")
    ax.set_title(f"Estimated lifetimes for {algorithm} on {scenario}")
    ax.grid(linestyle="-", color="0.95", axis="y")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(sorted_df["node_id"])
    ax.tick_params(axis="x", rotation=90)

    ax2 = ax.twinx()
    ax2.plot(
        x_positions,
        average_power_load,
        color="tab:orange",
        marker="o",
        linewidth=2,
        label="Average power load",
    )
    ax2.plot(
        x_positions,
        final_generated_power,
        color="tab:purple",
        marker="s",
        linewidth=2,
        linestyle="--",
        label="Final generated power",
    )
    ax2.set_ylabel("Power [W]")

    if (~np.isfinite(sorted_df["estimated_lifetime_years"])).any():
        ax.text(
            0.99,
            0.98,
            "Green bars indicate non-depleting spacecraft",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=12,
            bbox=dict(boxstyle="round", fc="0.95"),
        )
    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles + handles2, labels + labels2, loc="upper left")
    plt.tight_layout()
    plt.savefig(
        os.path.join(plot_dir, "mission_lifetime_distribution.pdf"),
        format="pdf",
        bbox_inches="tight",
    )
    plt.savefig(
        os.path.join(plot_dir, "mission_lifetime_distribution.png"),
        format="png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)


def _save_variant_comparison_plots(summary_df: pd.DataFrame) -> None:
    if summary_df.empty:
        return

    for scenario in PILOT_SCENARIOS:
        scenario_df = summary_df[summary_df["scenario"] == scenario].copy()
        if scenario_df.empty:
            continue

        scenario_df["comparison_variant"] = scenario_df["algorithm"].map(
            lambda value: VARIANT_LABELS.get(
                normalize_algorithm_name(value),
                normalize_algorithm_name(value),
            )
        )
        scenario_df = scenario_df[
            scenario_df["comparison_variant"].isin(VARIANT_ORDER)
        ]
        if scenario_df.empty:
            continue

        scenario_df = (
            scenario_df.sort_values(
                "comparison_variant",
                key=lambda series: series.map(
                    {variant: idx for idx, variant in enumerate(VARIANT_ORDER)}
                ),
            )
            .drop_duplicates(subset=["comparison_variant"], keep="last")
            .reset_index(drop=True)
        )

        comparison_dir = os.path.join(
            PLOTS_ROOT, scenario, "lifespan_comparison"
        )
        os.makedirs(comparison_dir, exist_ok=True)
        scenario_df.to_csv(
            os.path.join(comparison_dir, "mission_lifetime_summary.csv"),
            index=False,
        )

        metrics = [
            ("min_estimated_lifetime_years", "Min estimated lifetime [y]"),
            ("mean_estimated_lifetime_years", "Mean estimated lifetime [y]"),
            (
                "depleted_spacecraft_within_horizon",
                "Depleted spacecraft within horizon",
            ),
        ]
        for metric_name, ylabel in metrics:
            fig = plt.figure(figsize=(8, 4))
            ax = fig.add_subplot(111)
            ax.bar(
                scenario_df["comparison_variant"],
                scenario_df[metric_name],
                color="tab:blue",
            )
            ax.set_ylabel(ylabel)
            ax.set_xlabel("Variant")
            ax.set_title(f"{ylabel} for {scenario}")
            ax.grid(linestyle="-", color="0.95", axis="y")
            plt.tight_layout()
            plt.savefig(
                os.path.join(comparison_dir, f"{metric_name}.pdf"),
                format="pdf",
                bbox_inches="tight",
            )
            plt.savefig(
                os.path.join(comparison_dir, f"{metric_name}.png"),
                format="png",
                bbox_inches="tight",
                dpi=300,
            )
            plt.close(fig)


def run_analysis(report_id: int, plain_progress: bool = False) -> None:
    tegs = load_report_tegs(report_id)
    rows = [
        {
            "algorithm": algorithm,
            "scenario": scenario,
            "nodes": str(len(teg.node_map)),
            "states": str(teg.K),
            "min_life": "-",
            "progress": "-",
        }
        for algorithm, scenario, teg in tegs
    ]
    columns = [
        ("algorithm", {"no_wrap": True}),
        ("scenario", {"no_wrap": True}),
        ("nodes", {"justify": "right", "no_wrap": True}),
        ("states", {"justify": "right", "no_wrap": True}),
        ("min_life", {"justify": "right", "no_wrap": True}),
        ("progress", {"no_wrap": True}),
    ]
    summary_rows: list[dict[str, object]] = []

    with AnalysisRunTable(
        columns,
        rows,
        enable_live=not plain_progress,
    ) as run_table:
        for idx, (algorithm, scenario, teg) in enumerate(tegs):
            run_table.mark_progress(idx, 0)
            should_bypass_retargeting_time = (
                normalize_algorithm_name(algorithm) == "lls_pat_unaware"
            )
            EFFECTIVE_CONTACT_TIME_CACHE.clear()
            COORDINATE_CACHE.clear()
            metrics_df = compute_lifetime_metrics(
                teg=teg,
                should_bypass_retargeting_time=should_bypass_retargeting_time,
                progress_callback=lambda current,
                total,
                row_idx=idx: run_table.mark_progress(
                    row_idx,
                    int((current / max(total, 1)) * 100),
                ),
            )
            summary = summarize_lifetime_metrics(metrics_df)
            plot_dir = os.path.join(PLOTS_ROOT, scenario, algorithm)

            os.makedirs(plot_dir, exist_ok=True)
            metrics_df.to_csv(
                os.path.join(plot_dir, "mission_lifetime_node_metrics.csv"),
                index=False,
            )
            pd.DataFrame(
                [
                    {
                        "algorithm": algorithm,
                        "scenario": scenario,
                        **summary,
                    }
                ]
            ).to_csv(
                os.path.join(plot_dir, "mission_lifetime_summary.csv"),
                index=False,
            )
            _save_node_lifetime_plot(metrics_df, scenario, algorithm)

            summary_rows.append(
                {
                    "algorithm": algorithm,
                    "scenario": scenario,
                    **summary,
                }
            )
            run_table.update(
                idx,
                "min_life",
                _safe_float_str(
                    float(summary["min_estimated_lifetime_years"])
                ),
            )
            run_table.mark_progress(idx, 100)

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        report_plot_dir = os.path.join(PLOTS_ROOT, str(report_id))
        os.makedirs(report_plot_dir, exist_ok=True)
        summary_df.to_csv(
            os.path.join(report_plot_dir, "mission_lifetime_summary.csv"),
            index=False,
        )
        _save_variant_comparison_plots(summary_df)


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
        "--plain-progress",
        help="Disable the live run table. Useful when debugging with pdb/ipdb.",
    ),
) -> None:
    run_analysis(report_id, plain_progress)


if __name__ == "__main__":
    app()
