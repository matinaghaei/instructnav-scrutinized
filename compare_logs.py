import argparse
import os
import sys


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Compare log files by per-scene success rate. "
            "Print scenes sorted by how much better the focus log does "
            "compared to the other log file(s)."
        )
    )
    parser.add_argument(
        "log_files",
        nargs="+",
        help="Paths to log files (at least two).",
    )
    parser.add_argument(
        "-f", "--focus",
        type=int,
        default=0,
        help=(
            "Index (0-based) of the log file to treat as the focus/baseline. "
            "Default: 0 (the first log file)."
        ),
    )
    parser.add_argument(
        "-n", "--names",
        nargs="*",
        help=(
            "Optional human-readable names for the log files, in the same "
            "order as the paths. If omitted, basenames of the paths are used."
        ),
    )
    return parser.parse_args()


def parse_log_file(path):
    """
    Parse a log file and return a dict:
        scene_id -> average success (float)
    """
    scene_success_sum = {}
    scene_counts = {}

    try:
        with open(path, "r") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                data = {}
                for part in parts:
                    if "=" in part:
                        key, value = part.split("=", 1)
                        data[key] = value

                scene_id = data.get("scene_id")
                success_str = data.get("success")

                if scene_id is None or success_str is None:
                    # No scene or success: skip
                    continue

                try:
                    success = float(success_str)
                except ValueError:
                    print(
                        f"Warning: could not parse success on line {line_number} "
                        f"of {path!r}: {success_str!r}",
                        file=sys.stderr,
                    )
                    continue

                scene_success_sum[scene_id] = scene_success_sum.get(scene_id, 0.0) + success
                scene_counts[scene_id] = scene_counts.get(scene_id, 0) + 1

    except FileNotFoundError:
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error while reading {path}: {e}", file=sys.stderr)
        sys.exit(1)

    # Average success per scene
    return {
        scene_id: scene_success_sum[scene_id] / scene_counts[scene_id]
        for scene_id in scene_success_sum
    }


def main():
    args = parse_arguments()

    if len(args.log_files) < 2:
        print("Error: please provide at least two log files.", file=sys.stderr)
        sys.exit(1)

    num_logs = len(args.log_files)

    if args.focus < 0 or args.focus >= num_logs:
        print(
            f"Error: focus index {args.focus} is out of range for {num_logs} log files.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Determine names
    if args.names is not None:
        if len(args.names) != num_logs:
            print(
                "Error: number of names must match number of log files.",
                file=sys.stderr,
            )
            sys.exit(1)
        log_names = args.names
    else:
        log_names = [os.path.basename(p) or p for p in args.log_files]

    # Parse all logs -> list of dicts: [ {scene -> avg_success}, ... ]
    log_scene_success = []
    for path in args.log_files:
        log_scene_success.append(parse_log_file(path))

    # Collect per-scene success rates across all logs
    # scene_id -> [success_rate in log0, success_rate in log1, ...] (None if not present)
    scene_to_sr_list = {}
    for idx, sr_dict in enumerate(log_scene_success):
        for scene_id, sr in sr_dict.items():
            if scene_id not in scene_to_sr_list:
                scene_to_sr_list[scene_id] = [None] * num_logs
            scene_to_sr_list[scene_id][idx] = sr

    # Build comparison rows: only scenes where focus log has data and at least one other log has data
    rows = []
    focus_idx = args.focus

    for scene_id, srs in scene_to_sr_list.items():
        focus_sr = srs[focus_idx]
        if focus_sr is None:
            continue

        others = [sr for i, sr in enumerate(srs) if i != focus_idx and sr is not None]
        if not others:
            # No other logs for comparison
            continue

        avg_others = sum(others) / len(others)
        delta = focus_sr - avg_others
        rows.append((scene_id, focus_sr, avg_others, delta, srs))

    # Sort by how much better (or worse) the focus log did vs the others
    rows.sort(key=lambda x: x[3], reverse=True)  # sort by delta descending

    # Print results
    print(f"Comparing per-scene success rate across {num_logs} log files.")
    print(f"Focus log index: {focus_idx}  ({log_names[focus_idx]})")
    print()

    header_cols = ["Scene", "Focus_SR", "Delta_SR"] + [
        f"SR[{name.split('.')[0]}]" for name in log_names
    ]
    print("\t".join(header_cols).expandtabs(20))

    for scene_id, focus_sr, avg_others, delta, srs in rows:
        per_log_strs = [
            "" if sr is None else f"{sr:.3f}"
            for sr in srs
        ]
        cols = [
            scene_id.split(".")[0],
            f"{focus_sr:.3f}",
            # f"{avg_others:.3f}",
            f"{delta:+.3f}",
        ] + per_log_strs
        print("\t".join(cols).expandtabs(20))


if __name__ == "__main__":
    main()
