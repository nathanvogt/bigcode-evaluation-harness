import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch


def load_probabilities(file_path):
    return torch.load(file_path)


def id_results(path: str):
    failed_ids = []
    passed_ids = []
    with open(path, "r") as f:
        results = json.load(f)
    mbpp_results = (
        results["mbppplus"]
        if "mbppplus" in path
        else results["mbpp"] if "mbpp" in path else None
    )
    if mbpp_results is None:
        raise ValueError("No MBPP or MBPP+ results found in file")
    for mbpp_result in mbpp_results.values():
        _, result = mbpp_result[0]
        task_id = result["task_id"]
        passed = result["passed"]
        if passed:
            passed_ids.append(task_id)
        else:
            failed_ids.append(task_id)
    return passed_ids, failed_ids


def compare_steering_results(no_steer_path, steer_path, probs_path=None, plot=False):
    no_steer_passed, no_steer_failed = id_results(no_steer_path)
    steer_passed, steer_failed = id_results(steer_path)

    print(
        f"Without steering: {len(no_steer_passed)} passed, {len(no_steer_failed)} failed"
    )
    print(f"With steering:    {len(steer_passed)} passed, {len(steer_failed)} failed")

    failed_now_passing = set(no_steer_failed).intersection(steer_passed)
    passed_now_failing = set(no_steer_passed).intersection(steer_failed)

    print(f"Failed but now passing with steering: {len(failed_now_passing)}")
    print(f"Passed but now failing with steering: {len(passed_now_failing)}")

    def accuracy_at_threshold(threshold, probabilities):
        correct = 0
        failed = 0
        for i, p in enumerate(probabilities):
            if p >= threshold:
                if i in no_steer_passed:
                    correct += 1
                else:
                    failed += 1
            else:
                if i in steer_passed:
                    correct += 1
                else:
                    failed += 1
        return correct / (correct + failed)

    def find_max_accuracy(probabilities):
        max_accuracy = 0
        for thresh in probabilities:
            accuracy = accuracy_at_threshold(thresh, probabilities)
            if accuracy > max_accuracy:
                max_accuracy = accuracy
        return max_accuracy

    if probs_path:
        probabilities = load_probabilities(probs_path)
        max_accuracy = find_max_accuracy(probabilities)
        print(f"Max accuracy: {max_accuracy}")
        if plot:
            plot_task_probabilities(
                probabilities,
                no_steer_passed,
                no_steer_failed,
                steer_passed,
                steer_failed,
            )


def plot_task_probabilities(
    probabilities,
    no_steer_passed,
    no_steer_failed,
    steer_passed,
    steer_failed,
    filter=False,
):
    colors = [
        "green" if task in no_steer_passed else "red"
        for task in range(len(probabilities))
    ]
    edge_colors = [
        "green" if task in steer_passed else "red" for task in range(len(probabilities))
    ]
    if filter:
        probabilities = [
            probabilities[i]
            for i in range(len(probabilities))
            if (i in no_steer_passed and i in steer_failed)
            or (i in no_steer_failed and i in steer_passed)
        ]
        colors = [
            colors[i]
            for i in range(len(colors))
            if (i in no_steer_passed and i in steer_failed)
            or (i in no_steer_failed and i in steer_passed)
        ]
        edge_colors = [
            edge_colors[i]
            for i in range(len(edge_colors))
            if (i in no_steer_passed and i in steer_failed)
            or (i in no_steer_failed and i in steer_passed)
        ]

    plt.scatter(
        range(len(probabilities)),
        probabilities,
        s=50,
        color=colors,
        edgecolors=edge_colors,
        linewidth=2,
        alpha=0.6,
    )
    plt.xlabel("Task Index")
    plt.ylabel("Log Probability")
    plt.title("Task Probability by Result")

    plt.scatter([], [], color="green", label="Both Passed")
    plt.scatter([], [], color="red", label="Both Failed")
    plt.scatter(
        [],
        [],
        color="green",
        edgecolors="red",
        label="Passed but Steering Failed",
        linewidth=2,
    )
    plt.scatter(
        [],
        [],
        color="red",
        edgecolors="green",
        label="Failed but Steering Passed",
        linewidth=2,
    )
    plt.legend(title="Legend", loc="best")

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
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    compare_steering_results(
        args.no_steer_path, args.steer_path, args.probs_path, args.plot
    )


if __name__ == "__main__":
    main()
