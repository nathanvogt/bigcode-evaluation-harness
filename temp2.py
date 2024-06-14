import argparse
from steer_res import id_results
import os
import json
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steering_results_path", type=str, required=True)
    parser.add_argument("--no_steering_results_path", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    no_steer_passed, no_steer_failed = id_results(args.no_steering_results_path)
    steer_passed = []
    steer_failed = []
    for file_name in os.listdir(args.steering_results_path):
        try:
            id = int(file_name.split(".")[0])
        except:
            continue
        result = json.load(open(os.path.join(args.steering_results_path, file_name)))
        if result["passed"]:
            steer_passed.append(id)
        else:
            steer_failed.append(id)
    print(f"Originally: {len(no_steer_passed)} pass and {len(no_steer_failed)} fail")
    print(f"Steering: {len(steer_passed)} pass and {len(steer_failed)} fail")
    print(
        f"Failing now passing: {len(set(no_steer_failed).intersection(steer_passed))}"
    )
    print(
        f"Passing now failing: {len(set(no_steer_passed).intersection(steer_failed))}"
    )


if __name__ == "__main__":
    main()
