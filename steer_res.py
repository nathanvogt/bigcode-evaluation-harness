import json
import argparse


def id_results(path: str):
    failed_ids = []
    passed_ids = []
    with open(path, "r") as f:
        results = json.load(f)
    mbpp_results = results["mbpp"]
    for mbpp_result in mbpp_results.values():
        _, result = mbpp_result[0]
        task_id = result["task_id"]
        passed = result["passed"]
        if passed:
            passed_ids.append(task_id)
        else:
            failed_ids.append(task_id)
    return passed_ids, failed_ids


def compare_steering_results(no_steer_path, steer_path):
    no_steer_passed, no_steer_failed = id_results(no_steer_path)
    steer_passed, steer_failed = id_results(steer_path)
    print(
        f"Without steering: {len(no_steer_passed)} passed, {len(no_steer_failed)} failed, {len(no_steer_passed) / (len(no_steer_passed) + len(no_steer_failed)) * 100:.2f}% passed"
    )
    print(
        f"With steering: {len(steer_passed)} passed, {len(steer_failed)} failed, {len(steer_passed) / (len(steer_passed) + len(steer_failed)) * 100:.2f}% passed"
    )


def main():
    # create argument parser for the two result paths
    # parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_steer_path", type=str, required=True)
    parser.add_argument("--steer_path", type=str, required=True)
    args = parser.parse_args()
    compare_steering_results(args.no_steer_path, args.steer_path)


if __name__ == "__main__":
    main()
