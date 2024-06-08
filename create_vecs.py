import os
import json
import warnings
from bigcode_eval.tasks.mbpp import MBPP
import datasets
import torch
import transformers

from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    HfArgumentParser,
)

from bigcode_eval.arguments import EvalArguments
from bigcode_eval.evaluator import Evaluator
from bigcode_eval.tasks import ALL_TASKS

import steering


def parse_args():
    parser = HfArgumentParser(EvalArguments)

    parser.add_argument(
        "--model",
        default="codeparrot/codeparrot-small",
        help="Model to evaluate, provide a repo name in Hugging Face hub or a local path",
    )
    parser.add_argument(
        "--modeltype",
        default="causal",
        help="AutoModel to use, it can be causal or seq2seq",
    )
    parser.add_argument(
        "--peft_model",
        type=str,
        default=None,
        help="Adapter to the PEFT base model. Can be utilized for loading PEFT adapters such as a LoRA trained model. The --model parameter needs to be the base model.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Model revision to use",
    )
    parser.add_argument(
        "--use_auth_token",
        action="store_true",
        help="Use the token generated when running `huggingface-cli login` (necessary for private model).",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Use a model with custom code, this requires executing code by the author of the model.",
    )
    parser.add_argument(
        "--instruction_tokens",
        default=None,
        help="A series of instruction tokens used for instruction-tuning benchamrks separated by comma e.g. <user_message>,<end_user_message>,<assistant_message>",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation on each worker, can be larger for HumanEval",
    )
    parser.add_argument(
        "--max_length_generation",
        type=int,
        default=512,
        help="Maximum length of generated sequence (prompt+generation)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        help="Model precision, from: fp32, fp16 or bf16",
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load model in 8bit",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load model in 4bit",
    )
    parser.add_argument(
        "--left_padding",
        action="store_true",
        help="Force left padding, needed for models like chatglm3-6b",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of samples to solve and evaluate from the benchmark",
    )
    parser.add_argument(
        "--limit_start",
        type=int,
        default=0,
        help="Optional offset to start from when limiting the number of samples",
    )
    parser.add_argument(
        "--save_every_k_tasks",
        type=int,
        default=-1,
        help="Optional saving after every k tasks",
    )
    parser.add_argument(
        "--max_memory_per_gpu",
        type=str,
        default=None,
        help="Max memroy to allocate per gpu, you can also use 'auto'",
    )
    parser.add_argument(
        "--generations_path",
        type=str,
        default=None,
        help="Path to save generations",
    )
    return parser.parse_args()


def get_gpus_max_memory(max_memory, num_gpus):
    max_memory = {i: max_memory for i in range(num_gpus)}
    print("Loading model via these GPUs & max memories: ", max_memory)
    return max_memory


def create_model(args):
    # here we generate code and save it (evaluation is optional but True by default)
    dict_precisions = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    if args.precision not in dict_precisions:
        raise ValueError(
            f"Non valid precision {args.precision}, choose from: fp16, fp32, bf16"
        )

    model_kwargs = {
        "revision": args.revision,
        "trust_remote_code": args.trust_remote_code,
        "token": args.use_auth_token,
    }
    if args.load_in_8bit:
        print("Loading model in 8bit")
        model_kwargs["load_in_8bit"] = args.load_in_8bit
        model_kwargs["device_map"] = {"": accelerator.process_index}
    elif args.load_in_4bit:
        print("Loading model in 4bit")
        model_kwargs["load_in_4bit"] = args.load_in_4bit
        model_kwargs["torch_dtype"] = torch.float16
        model_kwargs["bnb_4bit_compute_dtype"] = torch.float16
        model_kwargs["device_map"] = {"": accelerator.process_index}
    else:
        print(f"Loading model in {args.precision}")
        model_kwargs["torch_dtype"] = dict_precisions[args.precision]

        if args.max_memory_per_gpu:
            if args.max_memory_per_gpu != "auto":
                model_kwargs["max_memory"] = get_gpus_max_memory(
                    args.max_memory_per_gpu, accelerator.num_processes
                )
                model_kwargs["offload_folder"] = "offload"
            else:
                model_kwargs["device_map"] = "auto"
                print("Loading model in auto mode")

    if args.modeltype == "causal":
        layers = steering.default_layers
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            **model_kwargs,
        )
        model = steering.wrap_layers(model, layers)
        steering_path = args.steering_path
        if steering_path:
            vecs = steering.load_steering_vecs(steering_path, layers)
            if args.norm_steering:
                vecs = steering.normalize_steering_vectors(vecs)
            model = steering.apply_steering_vectors(model, vecs)
    if not model:
        raise ValueError("Model not found")
    return model


def main():
    args = parse_args()
    # model = create_model(args)
    # get each generation
    # get corresponding solution
    # create steering vector

    if not args.generations_path:
        raise ValueError("Please provide generations path")
    with open(args.generations_path, "r") as f:
        generations = json.load(f)

    dataset = MBPP().get_dataset()

    for idx, gens in enumerate(generations):
        gen = gens[0]
        print(f"gen raw: {gen}")
        gen = dataset.postprocess_generation(gen, idx)
        print(f"post gen: {gen}")
        sol = dataset.get_solution(idx)
        print(f"sol: {sol}")


if __name__ == "__main__":
    main()
