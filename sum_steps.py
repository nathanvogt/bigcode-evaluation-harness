import steering
import os
import torch
import matplotlib.pyplot as plt


def main():
    folder_path = "/Users/nathanvogt/Downloads/vecs_k1_mbppplus"
    destination_path = "/Users/nathanvogt/Downloads/vecs_k1_summed_mbppplus"
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)
    for task_folder in os.listdir(folder_path):
        try:
            task_id = int(task_folder.split(".")[0])
        except:
            continue
        vecs = []
        for step_folder in os.listdir(os.path.join(folder_path, task_folder)):
            try:
                int(step_folder[0])
            except:
                continue
            vecs.append(
                steering.load_steering_vecs(
                    os.path.join(folder_path, task_folder, step_folder), device="cpu"
                )
            )
        summed_vecs = steering.normalize_steering_vectors(vecs[0])
        for vec in vecs[1:]:
            vec = steering.normalize_steering_vectors(vec)
            summed_vecs = steering.add_steering_vectors(summed_vecs, vec)
        steering.save_steering_vecs(
            os.path.join(destination_path, f"{task_id}"), summed_vecs
        )


main()
