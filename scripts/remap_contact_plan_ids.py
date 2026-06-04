#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from constants import BIT_RATES


CONTACT_PREFIX = "a contact "
RANGE_PREFIX = "a range "
POSITION_PREFIX = "a position "
RELAY_MIN = 1000
RELAY_MAX = 1999
SENDER_MIN = 2000
SENDER_MAX = 2999
SINK_MIN = 9000
SINK_MAX = 9999
MBPS_TO_BPS = 125_000
DEFAULT_EARTH_HUB_ID = "9999"
DEFAULT_EARTH_HUB_BIT_RATE_BPS = 100_000_000_000
DEFAULT_EARTH_HUB_RANGE_LIGHT_SECONDS = 0.0


def is_relay(node: str) -> bool:
    return node.isdigit() and RELAY_MIN <= int(node) <= RELAY_MAX


def is_sender(node: str) -> bool:
    return node.isdigit() and SENDER_MIN <= int(node) <= SENDER_MAX


def is_sink(node: str) -> bool:
    return node.isdigit() and SINK_MIN <= int(node) <= SINK_MAX


def get_contact_bit_rate_bps(tx_node: str, rx_node: str) -> int:
    return min(BIT_RATES[tx_node], BIT_RATES[rx_node])


def collect_node_ids(
    lines: list[str],
    extra_sink_ids: list[str] | None = None,
) -> tuple[list[str], list[str], list[str]]:
    relay_ids = set()
    sender_ids = set()
    sink_ids = set()

    for line in lines:
        if not (line.startswith(CONTACT_PREFIX) or line.startswith(RANGE_PREFIX)):
            continue

        parts = line.split()
        tx_node = parts[4]
        rx_node = parts[5]

        for node in (tx_node, rx_node):
            if is_relay(node):
                relay_ids.add(node)
            elif is_sender(node):
                sender_ids.add(node)
            elif is_sink(node):
                sink_ids.add(node)

    for node in extra_sink_ids or []:
        sink_ids.add(node)

    return (
        sorted(relay_ids, key=int),
        sorted(sender_ids, key=int),
        sorted(sink_ids, key=int),
    )


def build_id_map(
    lines: list[str],
    extra_sink_ids: list[str] | None = None,
) -> tuple[
    dict[str, str],
    tuple[int, int] | None,
    tuple[int, int] | None,
    tuple[int, int] | None,
    list[str],
]:
    relay_ids, sender_ids, sink_ids = collect_node_ids(lines, extra_sink_ids=extra_sink_ids)

    id_map: dict[str, str] = {}
    next_id = 1

    for node in relay_ids:
        id_map[node] = str(next_id)
        next_id += 1

    for node in sender_ids:
        id_map[node] = str(next_id)
        next_id += 1

    for node in sink_ids:
        id_map[node] = str(next_id)
        next_id += 1

    relay_range = None if not relay_ids else (1, len(relay_ids))
    sender_start = 1 + len(relay_ids)
    sender_range = None if not sender_ids else (1, len(sender_ids))
    if sender_range is not None:
        sender_range = (sender_start, sender_start + len(sender_ids) - 1)
    sink_start = 1 + len(relay_ids) + len(sender_ids)
    sink_range = None if not sink_ids else (sink_start, sink_start + len(sink_ids) - 1)

    return id_map, relay_range, sender_range, sink_range, sink_ids


def remap_contact_line(line: str, id_map: dict[str, str], bit_rate: int | None = None) -> str:
    parts = line.split()
    tx_node = parts[4]
    rx_node = parts[5]
    parts[4] = id_map.get(parts[4], parts[4])
    parts[5] = id_map.get(parts[5], parts[5])
    if bit_rate is not None:
        parts[6] = str(bit_rate)
    else:
        parts[6] = str(get_contact_bit_rate_bps(tx_node, rx_node) * MBPS_TO_BPS)
    return " ".join(parts)


def remap_range_line(line: str, id_map: dict[str, str]) -> str:
    parts = line.split()
    parts[4] = id_map.get(parts[4], parts[4])
    parts[5] = id_map.get(parts[5], parts[5])
    return " ".join(parts)


def remap_position_line(line: str, id_map: dict[str, str]) -> str:
    parts = line.split()
    parts[4] = id_map.get(parts[4], parts[4])
    parts[5] = id_map.get(parts[5], parts[5])
    return " ".join(parts)


def split_sections(lines: list[str]) -> tuple[list[str], list[str], list[str], list[str]]:
    contact_lines = []
    range_lines = []
    position_lines = []
    other_lines = []
    for line in lines:
        if line.startswith(CONTACT_PREFIX):
            contact_lines.append(line)
        elif line.startswith(RANGE_PREFIX):
            range_lines.append(line)
        elif line.startswith(POSITION_PREFIX):
            position_lines.append(line)
        elif line:
            other_lines.append(line)
    return contact_lines, range_lines, position_lines, other_lines


def create_earth_hub_lines(
    contact_lines: list[str],
    sink_ids: list[str],
    id_map: dict[str, str],
    earth_hub_id: str,
    earth_hub_bit_rate: int,
    earth_hub_range: float,
) -> tuple[list[str], list[str]]:
    if not contact_lines:
        return [], []

    start_time = min(int(line.split()[2][1:]) for line in contact_lines)
    end_time = max(int(line.split()[3][1:]) for line in contact_lines)
    hub_node = id_map[earth_hub_id]
    remapped_sink_ids = [id_map[node] for node in sink_ids if node != earth_hub_id]

    synthetic_contacts = []
    synthetic_ranges = []
    for sink_node in remapped_sink_ids:
        synthetic_contacts.append(
            f"{CONTACT_PREFIX}+{start_time} +{end_time} {sink_node} {hub_node} {earth_hub_bit_rate}"
        )
        synthetic_contacts.append(
            f"{CONTACT_PREFIX}+{start_time} +{end_time} {hub_node} {sink_node} {earth_hub_bit_rate}"
        )
        synthetic_ranges.append(
            f"{RANGE_PREFIX}+{start_time} +{end_time} {sink_node} {hub_node} {earth_hub_range}"
        )
        synthetic_ranges.append(
            f"{RANGE_PREFIX}+{start_time} +{end_time} {hub_node} {sink_node} {earth_hub_range}"
        )

    return synthetic_contacts, synthetic_ranges


def remap_lines(
    lines: list[str],
    id_map: dict[str, str],
    bit_rate: int | None = None,
    add_earth_hub: bool = False,
    earth_hub_id: str = DEFAULT_EARTH_HUB_ID,
    earth_hub_bit_rate: int = DEFAULT_EARTH_HUB_BIT_RATE_BPS,
    earth_hub_range: float = DEFAULT_EARTH_HUB_RANGE_LIGHT_SECONDS,
    sink_ids: list[str] | None = None,
) -> list[str]:
    contact_lines, range_lines, position_lines, other_lines = split_sections(lines)
    remapped_contact_lines = [remap_contact_line(line, id_map, bit_rate=bit_rate) for line in contact_lines]
    remapped_range_lines = [remap_range_line(line, id_map) for line in range_lines]
    remapped_position_lines = [remap_position_line(line, id_map) for line in position_lines]

    if add_earth_hub:
        synthetic_contacts, synthetic_ranges = create_earth_hub_lines(
            contact_lines=contact_lines,
            sink_ids=sink_ids or [],
            id_map=id_map,
            earth_hub_id=earth_hub_id,
            earth_hub_bit_rate=earth_hub_bit_rate,
            earth_hub_range=earth_hub_range,
        )
        remapped_contact_lines.extend(synthetic_contacts)
        remapped_range_lines.extend(synthetic_ranges)

    remapped_lines = remapped_contact_lines + [""] + remapped_range_lines
    if remapped_position_lines:
        remapped_lines += [""] + remapped_position_lines
    if other_lines:
        remapped_lines += [""] + other_lines
    return remapped_lines


def format_range(label: str, node_range: tuple[int, int] | None) -> str:
    if node_range is None:
        return f"{label}: none"
    return f"{label}: {node_range[0]}-{node_range[1]}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remap relay, sender, and sink IDs in a contact plan to contiguous DTNSim node IDs."
    )
    parser.add_argument("-i", "--input", required=True, help="Path to the source contact plan.")
    parser.add_argument("-o", "--output", required=True, help="Path to write the remapped contact plan.")
    parser.add_argument(
        "-r",
        "--reference",
        default=None,
        help="Optional reference contact plan whose node set should define the contiguous DTNSim ID mapping.",
    )
    parser.add_argument(
        "--bit-rate",
        type=int,
        default=None,
        help="Optional fixed bitrate in bps to apply to all remapped contact lines.",
    )
    parser.add_argument(
        "--add-earth-hub",
        action="store_true",
        help="Append a synthetic always-on Earth hub node connected to every sink node.",
    )
    parser.add_argument(
        "--earth-hub-id",
        default=DEFAULT_EARTH_HUB_ID,
        help="Original node ID to assign to the synthetic Earth hub before remapping.",
    )
    parser.add_argument(
        "--earth-hub-bit-rate",
        type=int,
        default=DEFAULT_EARTH_HUB_BIT_RATE_BPS,
        help="Bit rate in bps for synthetic Earth hub contacts.",
    )
    parser.add_argument(
        "--earth-hub-range",
        type=float,
        default=DEFAULT_EARTH_HUB_RANGE_LIGHT_SECONDS,
        help="Range in light-seconds for synthetic Earth hub contacts.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    lines = input_path.read_text().splitlines()
    if args.reference is not None:
        reference_lines = Path(args.reference).read_text().splitlines()
    else:
        reference_lines = lines

    extra_sink_ids = [args.earth_hub_id] if args.add_earth_hub else None
    id_map, relay_range, sender_range, sink_range, sink_ids = build_id_map(
        reference_lines,
        extra_sink_ids=extra_sink_ids,
    )
    remapped_lines = remap_lines(
        lines,
        id_map,
        bit_rate=args.bit_rate,
        add_earth_hub=args.add_earth_hub,
        earth_hub_id=args.earth_hub_id,
        earth_hub_bit_rate=args.earth_hub_bit_rate,
        earth_hub_range=args.earth_hub_range,
        sink_ids=sink_ids,
    )

    output_path.write_text("\n".join(remapped_lines) + "\n")

    print(f"Wrote remapped contact plan to {output_path}")
    if args.bit_rate is not None:
        print(f"contact_bit_rate: {args.bit_rate}")
    else:
        print(f"contact_bit_rate: source Mbps values converted to Bps using x{MBPS_TO_BPS}")
    if args.add_earth_hub:
        print(
            "earth_hub: "
            f"original_id={args.earth_hub_id}, remapped_id={id_map[args.earth_hub_id]}, "
            f"bit_rate={args.earth_hub_bit_rate}, range={args.earth_hub_range}"
        )
    print(format_range("relay_range", relay_range))
    print(format_range("sender_range", sender_range))
    print(format_range("sink_range", sink_range))


if __name__ == "__main__":
    main()
