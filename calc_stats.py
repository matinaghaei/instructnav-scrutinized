import sys
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Calculate average metrics from a log file."
    )
    parser.add_argument(
        "log_file_path",
        type=str,
        help="Path to the log file."
    )
    return parser.parse_args()

def main(log_file_path):
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
        print(f"Processed {match_count} entries.")
        print(f"Average Success: {avg_success:.4f}")
        print(f"Average SPL: {avg_spl:.4f}")
        print(f"Average Steps: {avg_steps:.2f}")
    else:
        print("No matching episodes found in the log file.")

if __name__ == "__main__":
    args = parse_arguments()
    main(args.log_file_path)
