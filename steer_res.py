import json
import argparse
import matplotlib.pyplot as plt
import numpy as np


def load_probabilities(file_path):
    probabilities = []
    with open(file_path, "r") as file:
        for line in file:
            prob = float(line.strip())
            probabilities.append(prob)
    return probabilities


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


def compare_steering_results(no_steer_path, steer_path, probs_path=None):
    no_steer_passed, no_steer_failed = id_results(no_steer_path)
    steer_passed, steer_failed = id_results(steer_path)

    print(
        f"Without steering: {len(no_steer_passed)} passed, {len(no_steer_failed)} failed"
    )
    print(f"With steering: {len(steer_passed)} passed, {len(steer_failed)} failed")

    failed_now_passing = set(no_steer_failed).intersection(steer_passed)
    passed_now_failing = set(no_steer_passed).intersection(steer_failed)

    print(f"Failed but now passing with steering: {len(failed_now_passing)}")
    print(f"Passed but now failing with steering: {len(passed_now_failing)}")

    if probs_path:
        probabilities = load_probabilities(probs_path)
        plot_task_probabilities(
            probabilities, no_steer_passed, no_steer_failed, steer_passed, steer_failed
        )


def plot_task_probabilities(
    probabilities, no_steer_passed, no_steer_failed, steer_passed, steer_failed
):
    colors = [
        "green" if task in no_steer_passed else "red"
        for task in range(len(probabilities))
    ]
    edge_colors = [
        "green" if task in steer_passed else "red" for task in range(len(probabilities))
    ]

    plt.scatter(
        range(len(probabilities)),
        probabilities,
        color=colors,
        edgecolors=edge_colors,
        linewidth=2,
    )
    plt.xlabel("Task Index")
    plt.ylabel("Probability")
    plt.title("Task Probability by Steering Results")
    plt.show()


def plot_comparison_matrix(
    still_passed, passed_now_failing, failed_now_passing, still_failed
):
    fig, ax = plt.subplots()
    categories = ["Passed", "Failed"]
    counts = [[still_passed, passed_now_failing], [failed_now_passing, still_failed]]
    cax = ax.matshow(counts, cmap="Blues")
    plt.title("Comparison of Test Results With and Without Steering")
    fig.colorbar(cax)
    ax.set_xticklabels([""] + categories)
    ax.set_yticklabels([""] + categories)
    ax.set_xlabel("With Steering")
    ax.set_ylabel("Without Steering")

    for (i, j), val in np.ndenumerate(counts):
        ax.text(j, i, f"{val}", ha="center", va="center", color="black")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_steer_path", type=str, required=True)
    parser.add_argument("--steer_path", type=str, required=True)
    parser.add_argument("--probs_path", type=str, required=False)
    args = parser.parse_args()
    compare_steering_results(args.no_steer_path, args.steer_path, args.probs_path)


if __name__ == "__main__":
    main()
