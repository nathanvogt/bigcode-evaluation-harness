from typing import List, Dict
import os
import torch

default_layers = [14, 15, 16, 17, 18]


class WrappedModule(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.output = None
        self.steering_vec = None

    def forward(self, *args, **kwargs):
        self.output = self.module(*args, **kwargs)
        if self.steering_vec is not None:
            return (self.output[0] + self.steering_vec,) + self.output[1:]
        else:
            return self.output


def wrap_layers(model, layers: List[int]):
    for layer_index in layers:
        old_layer = model.model.layers[layer_index].self_attn
        wrapped = WrappedModule(old_layer)
        model.model.layers[layer_index].self_attn = wrapped
    return model


def create_steering_vectors(
    model, tokenizer, layers: List[int], prompt: str, include_steering=False
):
    vectors = dict()
    for layer_index in layers:
        layer = model.model.layers[layer_index].self_attn
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        _ = model(input_ids)
        vec = layer.output[0][:, -1:, :].detach()
        if include_steering and layer.steering_vec is not None:
            vec += layer.steering_vec
        vectors[layer_index] = vec
    return vectors


def subtract_steering_vectors(vectors1: Dict, vectors2: Dict):
    res = dict()
    for layer_index, steering_vec in vectors1.items():
        if not layer_index in vectors2:
            continue
        res[layer_index] = steering_vec - vectors2[layer_index]
    return res


def add_steering_vectors(vectors1: Dict, vectors2: Dict):
    res = dict()
    for layer_index, steering_vec in vectors1.items():
        if not layer_index in vectors2:
            continue
        res[layer_index] = steering_vec + vectors2[layer_index]
    return res


def normalize_steering_vectors(vectors: Dict):
    vectors = vectors.copy()
    for layer, vec in vectors.items():
        vectors[layer] = vec / torch.linalg.vector_norm(vec)
    return vectors


def scale_steering_vectors(alpha, vectors: Dict):
    res = dict()
    for layer_index, steering_vec in vectors.items():
        res[layer_index] = alpha * steering_vec
    return res


def apply_steering_vectors(model, steering_vecs: Dict):
    for layer_index, steering_vec in steering_vecs.items():
        model.model.layers[layer_index].self_attn.steering_vec = steering_vec


def clear_steering_vectors(model):
    for layer in model.model.layers:
        if not hasattr(layer, "self_attn"):
            continue
        attn = layer.self_attn
        if not hasattr(attn, "steering_vec") or attn.steering_vec is None:
            continue
        attn.steering_vec = torch.zeros_like(attn.steering_vec)


def generate_one_completion(model, tokenizer, prompt):
    messages = [
        {
            "role": "user",
            "content": f"Correctly implement the python function. No explanations. Only code. Don't put quotations.\n\n{prompt}",
        },
    ]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]
    outputs = model.generate(
        input_ids,
        max_new_tokens=128,
        eos_token_id=terminators,
        do_sample=False,
        use_cache=False,
    )
    response = outputs[0][input_ids.shape[-1] :]
    return tokenizer.decode(response, skip_special_tokens=True)


def save_steering_vecs(save_path: str, vecs: Dict):
    for layer_index, layer_vec in vecs.items():
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # torch.save(layer_vec, f"./vecs/{task_id}/{layer_index}.pt")
        torch.save(layer_vec, os.path.join(save_path, f"{layer_index}.pt"))


def load_steering_vecs(folder_path: str, layers=None):
    res = dict()
    for file_name in os.listdir(folder_path):
        layer_index = int(file_name.split(".")[0])
        if layers is not None and layer_index not in layers:
            continue
        vec = torch.load(
            f"{folder_path}/{file_name}", map_location=torch.device("cuda")
        )
        vec = vec.to(torch.float16)
        res[layer_index] = vec
    return res
