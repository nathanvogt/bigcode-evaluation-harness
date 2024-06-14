import os
import json


def main():
    res = {}
    folder_path = "/Users/nathanvogt/Downloads/gens_train_k_8"
    destination_path = "/Users/nathanvogt/Downloads/res_det_train_k8.json"
    passed_w_steering = []
    for file_name in os.listdir(folder_path):
        try:
            id = int(file_name.split(".")[0])
        except:
            continue
        with open(os.path.join(folder_path, file_name)) as f:
            result = json.load(f)
            if result["passed"]:
                passed_w_steering.append(id)
        # print(passed_w_steering)
        res[str(id)] = [
            [
                0,
                {
                    "task_id": id,
                    "passed": result["passed"],
                    "result": "",
                    "completion_id": 0,
                },
            ]
        ]
    stuff = {"mbpp": res}
    with open(destination_path, "w") as f:
        json.dump(stuff, f)


def k_generations_to_classic_generations():
    folder_path = "/Users/nathanvogt/Downloads/gens_train_k_4"
    destination_path = "/Users/nathanvogt/Downloads/gens_train_solutions.json"
    generations = []
    for file_name in os.listdir(folder_path):
        try:
            id = int(file_name.split(".")[0])
        except:
            continue
        with open(os.path.join(folder_path, file_name)) as f:
            result = json.load(f)
            # gen = result["generations"][-1]
            gen = result["solution"]
            generations.append((id, gen))
    # sort by id
    generations.sort(key=lambda x: x[0])
    generations = [[x[1]] for x in generations]
    with open(destination_path, "w") as f:
        json.dump(generations, f)


if __name__ == "__main__":
    k_generations_to_classic_generations()
    # main()
