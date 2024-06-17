import os
import json
from bigcode_eval.evaluator import Evaluator
from bigcode_eval.generation import parallel_generations
from bigcode_eval.tasks.mbpp import MBPP
import torch

from accelerate import Accelerator

from bigcode_eval.tasks.mbppplus import MBPPPlus
from steer_res import id_results
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
)

from bigcode_eval.arguments import EvalArguments

import steering

train_ids = [
    327,
    57,
    12,
    140,
    125,
    114,
    71,
    52,
    346,
    279,
    44,
    302,
    216,
    16,
    15,
    47,
    111,
    119,
    258,
    308,
    13,
    287,
    101,
    332,
    368,
    214,
    112,
    229,
    301,
    142,
    3,
    81,
    365,
    174,
    348,
    79,
    110,
    172,
    370,
    362,
    194,
    49,
    183,
    176,
    309,
    135,
    22,
    235,
    274,
    63,
    193,
    40,
    282,
    150,
    321,
    316,
    185,
    295,
    98,
    35,
    23,
    116,
    148,
    326,
    360,
    51,
    337,
    343,
    232,
    186,
    83,
    189,
    181,
    107,
    136,
    36,
    87,
    273,
    373,
    307,
    236,
    311,
    138,
    285,
    351,
    166,
    28,
    117,
    364,
    161,
    205,
    137,
    33,
    108,
    288,
    284,
    255,
    202,
    234,
    73,
    354,
    371,
    126,
    134,
    219,
    204,
    323,
    293,
    70,
    260,
    252,
    46,
    24,
    56,
    78,
    369,
    345,
    32,
    197,
    195,
    239,
    128,
    5,
    344,
    184,
    29,
    254,
    226,
    286,
    192,
    68,
    196,
    164,
    349,
    291,
    75,
    361,
    314,
    322,
    0,
    253,
    224,
    237,
    67,
    256,
    359,
    45,
    129,
    27,
    222,
    160,
    76,
    215,
    163,
    230,
    155,
    50,
    39,
    95,
    333,
    41,
    320,
    199,
    277,
    238,
    153,
    82,
    299,
    4,
    243,
    92,
    206,
    263,
    61,
    14,
    272,
    145,
    20,
    21,
    187,
    124,
    17,
    296,
    303,
    268,
    377,
    168,
    121,
    374,
]
initially_train_failing = [
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
    parser.add_argument(
        "--save_folder",
        type=str,
        default=None,
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

    if args.modeltype == "causal":
        model_kwargs["device_map"] = "cuda"
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
    model.eval()
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

    if args.save_folder is not None and not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    task = MBPP(args.dataset_split) if args.task == "mbpp" else MBPPPlus()

    layers = steering.default_layers
    embedding_dim = 4096
    num_epochs = 4
    failed_ids = initially_train_failing

    dataset = task.get_dataset()
    total = len(dataset)
    with torch.no_grad():
        vec_sum = steering.create_zero_steering_vectors(layers, embedding_dim)
        for epoch in range(num_epochs):
            if args.save_folder is not None and not os.path.exists(
                os.path.join(args.save_folder, f"epoch_{epoch}")
            ):
                os.makedirs(os.path.join(args.save_folder, f"epoch_{epoch}"))
            print(f"Epoch {epoch + 1}/{num_epochs}")
            # steering.clear_steering_vectors(model)
            # create steering vectors from failed idxs
            epoch_vec_sum = steering.create_zero_steering_vectors(layers, embedding_dim)
            for idx, doc in enumerate(dataset):
                if idx not in failed_ids:
                    continue
                print(f"Processing {idx + 1}/{total}...")
                gen = generations[idx][0]
                solution = task.get_solution(idx)
                gen_vec = steering.create_steering_vectors(
                    model, tokenizer, layers, gen, include_steering=True
                )
                solution_vec = steering.create_steering_vectors(
                    model, tokenizer, layers, solution, include_steering=True
                )
                steer_vec = steering.subtract_steering_vectors(solution_vec, gen_vec)
                steer_vec = steering.normalize_steering_vectors(steer_vec)
                epoch_vec_sum = steering.add_steering_vectors(epoch_vec_sum, steer_vec)
                print(f"Completed {idx + 1}/{total}")
            if args.save_folder is not None:
                path = os.path.join(args.save_folder, f"epoch_{epoch}", "vecs_sum")
                steering.save_steering_vecs(path, vec_sum)
            epoch_vec_sum = steering.normalize_steering_vectors(epoch_vec_sum)
            vec_sum = steering.add_steering_vectors(vec_sum, epoch_vec_sum)
            print("Steering vectors created")
            torch.cuda.empty_cache()
            print("Generating code with steering vectors")
            # generate code with steering vectors
            # steer_vec = steering.normalize_steering_vectors(vec_sum)
            steering.apply_steering_vectors(model, vec_sum)
            generations = []
            references = []
            for idx, doc in enumerate(dataset):
                print(f"Processing {idx + 1}/{total}...")
                if train_ids is not None and idx not in train_ids:
                    generations.append([])
                    references.append("")
                    continue
                prompt = task.get_prompt(doc)
                gen = steering.generate_one_completion(
                    model, tokenizer, prompt, max_new_tokens=128
                )
                reference = task.get_reference(doc)
                generations.append([gen])
                references.append(reference)
                print(f"Completed {idx + 1}/{total}")
            # save generations
            if args.save_folder is not None:
                path = os.path.join(
                    args.save_folder, f"epoch_{epoch}", "generations.json"
                )
                with open(path, "w") as f:
                    json.dump(generations, f)
            # evaluate generations
            results, details = task.process_results(
                generations,
                references,
            )
            if args.save_folder is not None:
                path = os.path.join(args.save_folder, f"epoch_{epoch}", "results.json")
                with open(path, "w") as f:
                    json.dump(results, f)
                path = os.path.join(args.save_folder, f"epoch_{epoch}", "details.json")
                with open(path, "w") as f:
                    json.dump(details, f)
            print(f"Train pass rate: {results['pass@1']}")
            passed_ids, failed_ids = id_results(details)

            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
