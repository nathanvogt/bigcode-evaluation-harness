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


def create_zero_steering_vectors(layers: List[int], dim: int, device="cpu"):
    res = dict()
    for layer_index in layers:
        res[layer_index] = torch.zeros((1, 1, dim), device=device)
    return res


def create_steering_vectors(
    model, tokenizer, layers: List[int], prompt: str, include_steering=False
):
    vectors = dict()
    for layer_index in layers:
        layer = model.model.layers[layer_index].self_attn
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        _ = model(input_ids)
        vec = layer.output[0][:, -1:, :].detach().cpu()
        if include_steering and layer.steering_vec is not None:
            vec += layer.steering_vec.detach().cpu()
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
        layer = model.model.layers[layer_index].self_attn
        if layer.steering_vec is not None:
            del layer.steering_vec
        steering_vec.to(model.device)
        layer.steering_vec = steering_vec


def clear_steering_vectors(model):
    for layer in model.model.layers:
        if not hasattr(layer, "self_attn"):
            continue
        attn = layer.self_attn
        if not hasattr(attn, "steering_vec") or attn.steering_vec is None:
            continue
        del attn.steering_vec
        attn.steering_vec = None


def generate_one_completion(
    model, tokenizer, prompt, max_new_tokens=512, return_prompt=False
):
    messages = [
        {
            "role": "user",
            "content": f"Correctly implement the python function. No explanations. Only code. Don't put quotations.\n\n{prompt}",
        },
    ]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    # input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]
    outputs = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=terminators,
        do_sample=False,
    )
    response = outputs[0][input_ids.shape[-1] :]
    response_text = tokenizer.decode(response, skip_special_tokens=True)
    input_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    if return_prompt:
        return response_text, input_text
    return response_text


def save_steering_vecs(save_path: str, vecs: Dict):
    for layer_index, layer_vec in vecs.items():
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # torch.save(layer_vec, f"./vecs/{task_id}/{layer_index}.pt")
        torch.save(layer_vec, os.path.join(save_path, f"{layer_index}.pt"))


def load_steering_vecs(folder_path: str, layers=None, device="cuda"):
    res = dict()
    for file_name in os.listdir(folder_path):
        layer_index = int(file_name.split(".")[0])
        if layers is not None and layer_index not in layers:
            continue
        vec = torch.load(
            f"{folder_path}/{file_name}", map_location=torch.device(device)
        )
        vec = vec.to(torch.float16)
        res[layer_index] = vec
    return res


def to_tokens_and_logprobs(model, tokenizer, input_text):
    input_texts = [input_text]
    input_ids = tokenizer(input_texts, return_tensors="pt").input_ids
    outputs = model(input_ids)
    probs = torch.log_softmax(outputs.logits, dim=-1).detach()
    probs = probs[:, :-1, :]
    input_ids = input_ids[:, 1:]
    gen_probs = torch.gather(probs, 2, input_ids[:, :, None]).squeeze(-1)
    return gen_probs


def seq_prob(model, tokenizer, input_text):
    logprobs = to_tokens_and_logprobs(model, tokenizer, input_text)
    sum = torch.sum(logprobs, dim=-1)
    return torch.exp(sum)


def seq_log_prob(model, tokenizer, input_text):
    logprobs = to_tokens_and_logprobs(model, tokenizer, input_text)
    sum = torch.sum(logprobs, dim=-1)
    return sum
