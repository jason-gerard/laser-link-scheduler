from collections.abc import Callable
from enum import StrEnum, auto
import os

from rich.console import Console
from rich.live import Live
from rich.table import Table

from src.constants import SOURCES_ROOT

ProgressCallback = Callable[[str, int, int], None]


class FileType(StrEnum):
    CONTACT_PLAN = auto()
    CONTACT_PLAN_SCHEDULED = auto()
    TEG = auto()
    TEG_SCHEDULED = auto()
    REPORT = auto()


def get_experiment_file(experiment_name, file_type: FileType) -> str:
    return os.path.join(
        SOURCES_ROOT, experiment_name, f"{experiment_name}_{file_type}"
    )


def make_fanout_progress_callback(
    callbacks: list[ProgressCallback],
) -> ProgressCallback:
    def callback(stage: str, current: int, total: int) -> None:
        for progress_callback in callbacks:
            progress_callback(stage, current, total)

    return callback


class RunTablePrinter:
    STAGE_RANGES: dict[str, tuple[int, int]] = {
        "build_teg": (0, 30),
        "fractionate": (30, 40),
        "dag_reduction": (40, 50),
        "schedule": (50, 85),
        "export": (85, 95),
        "report": (95, 100),
    }

    def __init__(
        self,
        experiment_names: list[str],
        scheduler_names: list[str],
        enable_live: bool = True,
    ) -> None:
        self.experiment_names = experiment_names
        self.scheduler_names = scheduler_names
        self.enable_live = enable_live
        self.rows: list[dict[str, str | float | int]] = []
        self.row_index: dict[tuple[str, str], int] = {}
        self.console = Console()
        self.live: Live | None = None

        for experiment_name in experiment_names:
            for scheduler_name in scheduler_names:
                self.row_index[(experiment_name, scheduler_name)] = len(
                    self.rows
                )
                self.rows.append(
                    {
                        "experiment_name": experiment_name,
                        "scheduler_name": scheduler_name,
                        "progress": "-",
                        "duration": 0,
                        "network_capacity": 0,
                        "network_wasted_capacity": 0,
                        "wasted_buffer_capacity": 0,
                        "jains_fairness_index": 0,
                        "scheduled_delay": 0,
                    }
                )

        if self.enable_live:
            self.live = Live(
                self._build_table(),
                refresh_per_second=10,
                transient=False,
                redirect_stdout=True,
                redirect_stderr=True,
            )

    def __enter__(self) -> "RunTablePrinter":
        if self.live is not None:
            self.live.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.live is not None:
            self.live.__exit__(exc_type, exc_value, traceback)
        else:
            self.console.print(self._build_table())

    def mark_running(self, experiment_name: str, scheduler_name: str) -> None:
        self._update_row(experiment_name, scheduler_name, {})

    def make_progress_callback(
        self, experiment_name: str, scheduler_name: str
    ) -> ProgressCallback:
        last_label: str | None = None

        def callback(stage: str, current: int, total: int) -> None:
            nonlocal last_label
            label = self._format_progress_label(stage, current, total)
            if label == last_label:
                return

            last_label = label
            self.update_progress(experiment_name, scheduler_name, label)

        return callback

    def update_progress(
        self, experiment_name: str, scheduler_name: str, progress: str
    ) -> None:
        self._update_row(
            experiment_name,
            scheduler_name,
            {
                "progress": progress,
            },
        )

    def update_result(
        self,
        experiment_name: str,
        scheduler_name: str,
        run_data: dict[str, str | float | int],
    ) -> None:
        self._update_row(experiment_name, scheduler_name, run_data)

    def _update_row(
        self,
        experiment_name: str,
        scheduler_name: str,
        row_data: dict[str, str | float | int],
    ) -> None:
        row_idx = self.row_index[(experiment_name, scheduler_name)]
        self.rows[row_idx] = {**self.rows[row_idx], **row_data}
        if self.live is not None:
            self.live.update(self._build_table(), refresh=True)

    def _build_table(self) -> Table:
        table = Table()
        table.add_column("experiment")
        table.add_column("scheduler")
        table.add_column("progress")
        table.add_column("seconds", justify="right")
        table.add_column("capacity", justify="right")
        table.add_column("wasted", justify="right")
        table.add_column("wasted_buf", justify="right")
        table.add_column("jain", justify="right")
        table.add_column("delay", justify="right")

        for row in self.rows:
            table.add_row(
                str(row["experiment_name"]),
                str(row["scheduler_name"]),
                str(row["progress"]),
                self._format_float(row["duration"]),
                self._format_int(row["network_capacity"]),
                self._format_float(row["network_wasted_capacity"]),
                self._format_int(row["wasted_buffer_capacity"]),
                self._format_float(row["jains_fairness_index"]),
                self._format_float(row["scheduled_delay"]),
            )

        return table

    def _format_progress_label(
        self, stage: str, current: int, total: int
    ) -> str:
        start, end = self.STAGE_RANGES.get(stage, (0, 100))
        fraction = 1.0 if total <= 0 else min(max(current / total, 0.0), 1.0)
        percent = int(start + ((end - start) * fraction))
        return f"{percent}%"

    @staticmethod
    def _format_int(value: str | float | int | None) -> str:
        if value is None:
            return "-"
        return f"{int(value):,}"

    @staticmethod
    def _format_float(value: str | float | int | None) -> str:
        if value is None:
            return "-"
        return f"{float(value):.4f}"
