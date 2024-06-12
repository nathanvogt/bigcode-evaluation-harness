import argparse
import os
import json
import curses


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steering_results_path", type=str, required=True)
    return parser.parse_args()


def load_data(steering_results_path):
    """
    Example of the structure:
    results[0] = {"k": 4, "passed": True, "generations": ["from typing import List\n", "class Pair:\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y\n\ndef max_chain_length(pairs, k):\n    pairs.sort(key=lambda x: x.y)\n    dp = [1] * k\n    for i in range(1, k):\n        for j in range(i):\n            if pairs[j].y < pairs[i].x:\n                dp[i] = max(dp[i], dp[j] + 1)\n    return max(dp)\n\n# Test the function"], "prompt": "\"\"\"\nWrite a function to find the longest chain which can be formed from the given set of pairs.\nassert max_chain_length([Pair(5, 24), Pair(15, 25),Pair(27, 40), Pair(50, 60)], 4) == 3\n\"\"\"\n", "solution": "class Pair(object): \r\n\tdef __init__(self, a, b): \r\n\t\tself.a = a \r\n\t\tself.b = b \r\ndef max_chain_length(arr, n): \r\n\tmax = 0\r\n\tmcl = [1 for i in range(n)] \r\n\tfor i in range(1, n): \r\n\t\tfor j in range(0, i): \r\n\t\t\tif (arr[i].a > arr[j].b and\r\n\t\t\t\tmcl[i] < mcl[j] + 1): \r\n\t\t\t\tmcl[i] = mcl[j] + 1\r\n\tfor i in range(n): \r\n\t\tif (max < mcl[i]): \r\n\t\t\tmax = mcl[i] \r\n\treturn max"}
    """
    results = dict()
    for file_name in os.listdir(steering_results_path):
        try:
            id = int(file_name.split(".")[0])
            with open(os.path.join(steering_results_path, file_name), "r") as file:
                result = json.load(file)
                results[id] = result
        except Exception as e:
            print(f"Failed to read {file_name}: {e}")
    return results


def display_result(stdscr, results, idx):
    stdscr.clear()
    result = results[idx]
    result_str = json.dumps(result, indent=4)
    stdscr.addstr(result_str)
    stdscr.refresh()


def main(stdscr):
    args = parse_args()
    results = load_data(args.steering_results_path)
    ids = sorted(results.keys())
    idx = 0

    while True:
        display_result(stdscr, results, ids[idx])
        key = stdscr.getch()
        if key == curses.KEY_RIGHT:
            idx = (idx + 1) % len(ids)
        elif key == curses.KEY_LEFT:
            idx = (idx - 1) % len(ids)
        elif key == ord("q"):
            break


if __name__ == "__main__":
    curses.wrapper(main)
