import os
import json


def main():
    res = {}
    folder_path = "/Users/nathanvogt/Downloads/gens_train_k_4"
    destination_path = "/Users/nathanvogt/Downloads/res_det_k4.json"
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
    print(passed_w_steering)
    #     res[str(id)] = [
    #         [
    #             0,
    #             {
    #                 "task_id": id,
    #                 "passed": result["passed"],
    #                 "result": "",
    #                 "completion_id": 0,
    #             },
    #         ]
    #     ]
    # stuff = {"mbpp": res}
    # with open(destination_path, "w") as f:
    #     json.dump(stuff, f)


main()
