import sys
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

def main(log_file_path, episode_type):
    # Determine the target set based on the episode_type
    if episode_type == "selected":
        episode_list = selected_episodes
        target_set = set(episode_list)
    elif episode_type == "hard":
        episode_list = hard_episodes
        target_set = set(episode_list)
    elif episode_type == "all":
        # In 'all' mode, we do not filter based on episode lists
        target_set = None
    else:
        print(f"Invalid episode type: {episode_type}")
        sys.exit(1)

    # Initialize accumulators for metrics and a counter for number of matching lines
    total_success = 0.0
    total_spl = 0.0
    total_steps = 0
    match_count = 0

    try:
        # Open and read the provided log file
        with open(log_file_path, 'r') as log_file:
            for line_number, line in enumerate(log_file, start=1):
                # Each line is expected to be in the format:
                # index=<int> scene_id=<id> episode_id=<id> success=<float> spl=<float> steps=<int>
                # Split line into parts
                parts = line.strip().split()
                # Create a dictionary for key-value pairs from the line
                data = {}
                for part in parts:
                    if '=' in part:
                        key, value = part.split('=', 1)
                        data[key] = value

                # Extract scene_id and episode_id as strings
                scene_id = data.get('scene_id')
                episode_id = data.get('episode_id')

                # Determine whether to process this line based on episode_type
                if episode_type != "all":
                    if not (scene_id and episode_id):
                        print(f"Warning: Missing scene_id or episode_id on line {line_number}. Skipping.")
                        continue
                    if (scene_id, episode_id) not in target_set:
                        continue  # Skip lines not in the target set

                # Parse and accumulate metrics
                try:
                    success = float(data.get('success', 0.0))
                    spl = float(data.get('spl', 0.0))
                    steps = int(data.get('steps', 0))
                except ValueError as ve:
                    print(f"Warning: Skipping line {line_number} due to parsing error: {ve}")
                    continue

                total_success += success
                total_spl += spl
                total_steps += steps
                match_count += 1
    except FileNotFoundError:
        print(f"Error: The file '{log_file_path}' was not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while processing the log file: {e}")
        sys.exit(1)

    # Compute and display averages if at least one matching line was found
    if match_count > 0:
        avg_success = total_success / match_count
        avg_spl = total_spl / match_count
        avg_steps = total_steps / match_count
        if episode_type == "all":
            print(f"Processed {match_count} entries from 'all episodes'.")
        else:
            print(f"Processed {match_count} matching entries from '{episode_type}_episodes'.")
        print(f"Average Success: {avg_success:.4f}")
        print(f"Average SPL: {avg_spl:.4f}")
        print(f"Average Steps: {avg_steps:.2f}")
    else:
        print("No matching episodes found in the log file.")

if __name__ == "__main__":
    args = parse_arguments()
    main(args.log_file_path, args.type)
