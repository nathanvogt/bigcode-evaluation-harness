import argparse
import os
import json
import sys
from pynput import keyboard


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steering_results_path", type=str, required=True)
    return parser.parse_args()


def print_result(idx, result):
    clear_console()
    content = (
        f"ID: {idx}\n"
        + f"Passed: {result['passed']}\n"
        + f"Prompt: \n{result['prompt'].strip()}\n"
        + f"Solution: \n{result['solution'].strip()}\n\n"
        + "\n\n".join(
            [
                f"Generation {i}: \n {gen.strip()}"
                for i, gen in enumerate(result["generations"])
            ]
        )
    )
    print(content)


def clear_console():
    os.system("clear")


def main():
    args = parse_args()
    results = {}

    for file_name in os.listdir(args.steering_results_path):
        id = int(file_name.split(".")[0])
        with open(os.path.join(args.steering_results_path, file_name)) as f:
            result = json.load(f)
        results[id] = result

    sorted_keys = sorted(results.keys())
    sorted_results = [results[k] for k in sorted_keys]
    current_index = [0]

    print_result(current_index[0], sorted_results[current_index[0]])

    def on_press(key):
        try:
            if key == keyboard.Key.right:
                current_index[0] = current_index[0] + 1 % len(sorted_results)
                print_result(current_index[0], sorted_results[current_index[0]])
            elif key == keyboard.Key.left:
                current_index[0] = (current_index[0] - 1 + len(sorted_results)) % len(
                    sorted_results
                )
                print_result(current_index[0], sorted_results[current_index[0]])
            elif key == keyboard.Key.esc:
                return False
        except Exception as e:
            print(str(e))

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()


if __name__ == "__main__":
    main()
