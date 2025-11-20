import sys
import os              # NEW
import argparse
from selected_scene_episode import selected_episodes
from hard_scene_episode import hard_episodes


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Calculate average metrics from a log file based on selected, hard, or all episodes."
    )
    parser.add_argument(
        "log_file_path",
        type=str,
        help="Path to the log file."
    )
    parser.add_argument(
        "-t", "--type",
        type=str,
        choices=["all", "selected", "hard"],
        default="all",
        help="Type of episodes to use: 'all' (default), 'selected', or 'hard'."
    )
    return parser.parse_args()


# NEW: load optimal steps from optimal.txt (same directory as log file)
def load_optimal_steps(log_file_path):
    optimal_steps = {}
    optimal_path = os.path.join(os.path.dirname(log_file_path), "optimal.txt")

    try:
        with open(optimal_path, "r") as opt_file:
            for line in opt_file:
                parts = line.strip().split()
                data = {}
                for part in parts:
                    if '=' in part:
                        key, value = part.split('=', 1)
                        data[key] = value

                scene_id = data.get("scene_id")
                episode_id = data.get("episode_id")
                steps_str = data.get("steps")

                if not scene_id or not episode_id or not steps_str:
                    continue

                # Treat "None" as +inf so min(optimal, steps) works as intended
                if steps_str == "None":
                    opt = float("inf")
                else:
                    try:
                        opt = float(steps_str)
                    except ValueError:
                        opt = float("inf")

                optimal_steps[(scene_id, episode_id)] = opt
    except FileNotFoundError:
        print(f"Warning: optimal.txt not found next to {log_file_path}. "
              "New metric will assume optimal=inf for all episodes.")

    return optimal_steps


def main(log_file_path, episode_type):
    # Choose episode subset
    if episode_type == "selected":
        target_set = set(selected_episodes)
    elif episode_type == "hard":
        target_set = set(hard_episodes)
    else:  # "all"
        target_set = None

    # NEW: load optimal steps
    optimal_steps = load_optimal_steps(log_file_path)

    # Initialize accumulators
    total_success = 0.0
    total_spl = 0.0
    total_steps = 0
    total_action_spl = 0.0    # NEW: accumulator for the new metric
    match_count = 0

    try:
        # Open and read the provided log file
        with open(log_file_path, 'r') as log_file:
            for line_number, line in enumerate(log_file, start=1):
                parts = line.strip().split()
                if not parts:
                    continue

                data = {}
                for part in parts:
                    if '=' in part:
                        key, value = part.split('=', 1)
                        data[key] = value

                # Extract scene_id and episode_id as strings
                scene_id = data.get('scene_id')
                episode_id = data.get('episode_id')

                # Filter by selected / hard episodes if requested
                if episode_type != "all":
                    if not (scene_id and episode_id):
                        print(f"Warning: Missing scene_id or episode_id on line {line_number}. Skipping.")
                        continue
                    if (scene_id, episode_id) not in target_set:
                        continue  # Skip lines not in the target set

                # Parse metrics
                try:
                    success = float(data.get('success', 0.0))
                    spl = float(data.get('spl', 0.0))
                    steps = int(data.get('steps', 0))
                except ValueError as ve:
                    print(f"Warning: Skipping line {line_number} due to parsing error: {ve}")
                    continue

                # Accumulate standard metrics
                total_success += success
                total_spl += spl
                total_steps += steps

                # NEW: compute action-based SPL-like metric using optimal steps
                # action_SPL = success * min(optimal_steps, steps) / steps
                opt = optimal_steps.get((scene_id, episode_id), float("inf"))
                if steps > 0:
                    action_spl = success * (min(opt, steps) / steps)
                else:
                    action_spl = 0.0
                total_action_spl += action_spl

                match_count += 1

    except FileNotFoundError:
        print(f"Error: The file '{log_file_path}' was not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while processing the log file: {e}")
        sys.exit(1)

    # Compute averages if at least one matching line was found
    if match_count > 0:
        avg_success = total_success / match_count
        avg_spl = total_spl / match_count
        avg_steps = total_steps / match_count
        avg_action_spl = total_action_spl / match_count  # NEW

        if episode_type == "all":
            print(f"Processed {match_count} entries from 'all episodes'.")
        else:
            print(f"Processed {match_count} matching entries from '{episode_type}_episodes'.")

        print(f"Average Success: {avg_success:.4f}")
        print(f"Average SPL: {avg_spl:.4f}")
        print(f"Average Steps: {avg_steps:.2f}")
        print(f"Average action-based SPL: {avg_action_spl:.4f}")  # NEW
    else:
        print("No matching episodes found in the log file.")


if __name__ == "__main__":
    args = parse_arguments()
    main(args.log_file_path, args.type)
