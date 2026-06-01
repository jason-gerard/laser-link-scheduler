#!/usr/bin/env python3

import argparse
from pathlib import Path


def parse_teg_file(path: Path) -> tuple[int, list[int], list[int]]:
    lines = path.read_text().splitlines()
    if not lines or not lines[0].startswith("num_k="):
        raise ValueError(f"{path} does not look like a TEG text file")

    num_k = int(lines[0].split("=", 1)[1])
    state_durations: list[int] = []
    edge_counts: list[int] = []

    i = 0
    while i < len(lines):
        if not lines[i].startswith("  0,"):
            i += 1
            continue

        i += 1
        edge_count = 0
        while i < len(lines) and not lines[i].startswith("k="):
            line = lines[i]
            if line.strip():
                _, adjacency = line.split(" ", 1)
                values = adjacency.split(",")[:-1]
                edge_count += sum(1 for value in values if value == "1")
            i += 1

        if i >= len(lines) or not lines[i].startswith("k="):
            raise ValueError(f"{path} is missing a k= line for one of the states")
        i += 1

        if i >= len(lines) or not lines[i].startswith("t="):
            raise ValueError(f"{path} is missing a t= line for one of the states")
        state_durations.append(int(lines[i].split("=", 1)[1]))
        edge_counts.append(edge_count)
        i += 1

    if len(edge_counts) != num_k:
        raise ValueError(
            f"{path} declares num_k={num_k}, but {len(edge_counts)} states were parsed"
        )

    return num_k, state_durations, edge_counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report state and edge counts for one or more saved TEG text files."
    )
    parser.add_argument("paths", nargs="+", help="Path(s) to TEG text file(s).")
    parser.add_argument(
        "--show-per-state",
        action="store_true",
        help="Print each state's duration and edge count.",
    )
    args = parser.parse_args()

    for raw_path in args.paths:
        path = Path(raw_path)
        num_k, state_durations, edge_counts = parse_teg_file(path)
        total_edges = sum(edge_counts)
        avg_edges = total_edges / num_k if num_k else 0.0
        total_duration = sum(state_durations)

        print(path)
        print(f"  num_states: {num_k}")
        print(f"  total_duration_seconds: {total_duration}")
        print(f"  total_edges: {total_edges}")
        print(f"  avg_edges_per_state: {avg_edges:.3f}")
        print(f"  min_edges_in_state: {min(edge_counts)}")
        print(f"  max_edges_in_state: {max(edge_counts)}")

        if args.show_per_state:
            for idx, (duration, edge_count) in enumerate(zip(state_durations, edge_counts), start=1):
                print(f"  state {idx}: duration={duration}, edges={edge_count}")


if __name__ == "__main__":
    main()
