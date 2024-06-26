import os
import json
from bigcode_eval.generation import parallel_generations
from bigcode_eval.tasks.mbpp import MBPP
import torch

from accelerate import Accelerator

from bigcode_eval.tasks.mbppplus import MBPPPlus
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
)

from bigcode_eval.arguments import EvalArguments

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
        "--task",
        type=str,
        default="mbpp",
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
        "--postprocess",
        action="store_false",
        help="Postprocess model outputs before execution, always on except during generation tests",
    )
    parser.add_argument(
        "--allow_code_execution",
        action="store_true",
        help="Allow code evaluation to execute external/untrusted Python code on your machine",
    )
    parser.add_argument(
        "--generation_only",
        action="store_true",
        help="Do code generation but no evaluation",
    )
    parser.add_argument(
        "--load_generations_path",
        type=str,
        default=None,
        help="Path of file with previously generated solutions, if provided generation is skipped and only evaluation is done",
    )
    parser.add_argument(
        "--load_data_path",
        type=str,
        default=None,
        help="Path of additional data to load for the tasks",
    )
    parser.add_argument(
        "--metric_output_path",
        type=str,
        default="evaluation_results.json",
        help="Path to save the results",
    )
    parser.add_argument(
        "--save_generations",
        action="store_true",
        help="Whether to save code generations",
    )
    parser.add_argument(
        "--load_generations_intermediate_paths",
        type=str,
        nargs="*",
        help="List of paths for saving the intermediate code generations",
    )
    parser.add_argument(
        "--save_generations_path",
        type=str,
        default="generations.json",
        help="Path for saving the code generations",
    )
    parser.add_argument(
        "--save_references",
        action="store_true",
        help="Whether to save reference solutions/tests",
    )
    parser.add_argument(
        "--save_references_path",
        type=str,
        default="references.json",
        help="Path for saving the references solutions/tests",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="prompt",
        help="Prompt type to use for generation in HumanEvalPack tasks",
    )
    parser.add_argument(
        "--max_memory_per_gpu",
        type=str,
        default=None,
        help="Max memroy to allocate per gpu, you can also use 'auto'",
    )
    parser.add_argument(
        "--check_references",
        action="store_true",
        help="Don't run generation but benchmark groundtruth (useful for debugging)",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="test",
        help="Dataset split to evaluate",
    )
    parser.add_argument(
        "--generations_path",
        type=str,
        default=None,
        help="Path to save generations",
    )
    parser.add_argument(
        "--steering_path",
        type=str,
        default=None,
        help="Path to steering vectors",
    )
    parser.add_argument(
        "--save_vecs_path",
        type=str,
        default=None,
        help="Path to save steering vectors",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=1,
        help="Number of iterations to extract steering vector",
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
        # model_kwargs["device_map"] = {"": accelerator.process_index}
    elif args.load_in_4bit:
        print("Loading model in 4bit")
        model_kwargs["load_in_4bit"] = args.load_in_4bit
        model_kwargs["torch_dtype"] = torch.float16
        model_kwargs["bnb_4bit_compute_dtype"] = torch.float16
        # model_kwargs["device_map"] = {"": accelerator.process_index}
    else:
        print(f"Loading model in {args.precision}")
        model_kwargs["torch_dtype"] = dict_precisions[args.precision]

        # if args.max_memory_per_gpu:
        #     if args.max_memory_per_gpu != "auto":
        #         model_kwargs["max_memory"] = get_gpus_max_memory(
        #             args.max_memory_per_gpu, accelerator.num_processes
        #         )
        #         model_kwargs["offload_folder"] = "offload"
        #     else:
        #         model_kwargs["device_map"] = "auto"
        #         print("Loading model in auto mode")

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


def create_tokenizer(args):
    if args.left_padding:
        # left padding is required for some models like chatglm3-6b
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            revision=args.revision,
            trust_remote_code=args.trust_remote_code,
            token=args.use_auth_token,
            padding_side="left",
        )
    else:
        # used by default for most models
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            revision=args.revision,
            trust_remote_code=args.trust_remote_code,
            token=args.use_auth_token,
            truncation_side="left",
            padding_side="right",
        )
    if not tokenizer.eos_token:
        if tokenizer.bos_token:
            tokenizer.eos_token = tokenizer.bos_token
            print("bos_token used as eos_token")
        else:
            raise ValueError("No eos_token or bos_token found")
    try:
        tokenizer.pad_token = tokenizer.eos_token

    # Some models like CodeGeeX2 have pad_token as a read-only property
    except AttributeError:
        print("Not setting pad_token to eos_token")
        pass
    WIZARD_LLAMA_MODELS = [
        "WizardLM/WizardCoder-Python-34B-V1.0",
        "WizardLM/WizardCoder-34B-V1.0",
        "WizardLM/WizardCoder-Python-13B-V1.0",
    ]
    if args.model in WIZARD_LLAMA_MODELS:
        tokenizer.bos_token = "<s>"
        tokenizer.bos_token_id = 1
        print("Changing bos_token to <s>")

    if not tokenizer:
        raise ValueError("Tokenizer not found")
    return tokenizer


def main():
    args = parse_args()

    accelerator = Accelerator()

    model = create_model(args)
    tokenizer = create_tokenizer(args)
    layers = steering.default_layers

    if not args.generations_path:
        raise ValueError("Please provide generations path")
    with open(args.generations_path, "r") as f:
        generations = json.load(f)

    if not args.save_vecs_path:
        raise ValueError("Please provide save vecs path")
    if not os.path.exists(args.save_vecs_path):
        os.makedirs(args.save_vecs_path)

    if args.save_generations_path and not os.path.exists(args.save_generations_path):
        os.makedirs(args.save_generations_path)

    mbpp = MBPP(args.dataset_split) if args.task == "mbpp" else MBPPPlus()

    mbppplus_train_failed_ids = [
        140,
        71,
        52,
        346,
        279,
        119,
        308,
        13,
        332,
        112,
        301,
        3,
        81,
        370,
        362,
        194,
        135,
        63,
        193,
        321,
        316,
        295,
        98,
        35,
        116,
        148,
        326,
        360,
        343,
        107,
        136,
        87,
        373,
        236,
        285,
        351,
        166,
        364,
        33,
        288,
        284,
        73,
        371,
        126,
        134,
        204,
        323,
        293,
        70,
        260,
        56,
        197,
        195,
        239,
        254,
        226,
        286,
        68,
        291,
        361,
        322,
        253,
        224,
        67,
        45,
        129,
        27,
        163,
        39,
        199,
        299,
        4,
        206,
        263,
        61,
        14,
        377,
        168,
        374,
    ]

    total = len(generations)
    for idx, gens in enumerate(generations):
        if idx not in mbppplus_train_failed_ids:
            continue
        print(f"Processing {idx + 1}/{total}...")
        steering.clear_steering_vectors(model)
        prompt = mbpp.get_prompt(mbpp.get_dataset()[idx])
        gen = gens[0]
        gen = mbpp.postprocess_generation(gen, idx, include_prompt=False)
        sol = mbpp.get_solution(idx)
        reference = mbpp.get_reference(mbpp.get_dataset()[idx])

        def get_cum_vec(vecs):
            summed_vecs = vecs[0]
            for vec in vecs[1:]:
                summed_vecs = steering.add_steering_vectors(summed_vecs, vec)
            return steering.normalize_steering_vectors(summed_vecs)

        generations = [gen]
        steering_vecs = []
        passed = False
        step = 0
        with torch.no_grad():
            while not passed and step < args.k:
                step += 1
                if len(steering_vecs):
                    steering.apply_steering_vectors(model, get_cum_vec(steering_vecs))
                gen_vecs = steering.create_steering_vectors(
                    model, tokenizer, layers, generations[-1], include_steering=True
                )
                new_sol_vecs = steering.create_steering_vectors(
                    model, tokenizer, layers, sol, include_steering=True
                )
                steer_vec = steering.subtract_steering_vectors(new_sol_vecs, gen_vecs)
                steering_vecs.append(steer_vec)

                steering.apply_steering_vectors(model, get_cum_vec(steering_vecs))
                next_gen = parallel_generations(
                    mbpp,
                    mbpp.get_dataset(),
                    accelerator,
                    model,
                    tokenizer,
                    n_tasks=1,
                    args=args,
                    curr_sample_idx=idx,
                )
                next_genn = mbpp.postprocess_generation(
                    next_gen[0][0], idx, include_prompt=False
                )
                generations.append(next_genn)
                passed = (
                    mbpp.process_results(
                        [[next_genn]],
                        [reference],
                    )[
                        0
                    ]["pass@1"]
                    == 1.0
                )

        for i, vec in enumerate(steering_vecs):
            save_path = os.path.join(args.save_vecs_path, f"{idx}", f"{i+1}")
            steering.save_steering_vecs(save_path, vec)
        if args.save_generations_path:
            save_path = os.path.join(args.save_generations_path, f"{idx}.json")
            with open(save_path, "w") as f:
                stuff = {
                    "k": args.k,
                    "passed": bool(passed),
                    "generations": generations,
                    "prompt": prompt,
                    "solution": sol,
                }
                json.dump(stuff, f)
        # clear tensors from gpu memory
        for vec in gen_vecs.values():
            del vec
        for vec in steer_vec.values():
            del vec
        torch.cuda.empty_cache()
        print(f"Completed {idx + 1}/{total}")


if __name__ == "__main__":
    main()
