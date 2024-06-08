import json

prompt = '"""\nWrite a function to find the longest chain which can be formed from the given set of pairs.\nassert max_chain_length([Pair(5, 24), Pair(15, 25),Pair(27, 40), Pair(50, 60)], 4) == 3\n"""\n'

stop_words = [
    "\nclass",
    "\nassert",
    '\n"""',
    "\nprint",
    "\nif",
    "\n<|/",
    "\n```",
]


def _stop_at_stop_token(decoded_string, stop_tokens):
    """
    Produces the prefix of decoded_string that ends at the first occurrence of
    a stop_token.
    WARNING: the decoded_string *must not* include the prompt, which may have stop tokens
    itself.
    """
    min_stop_index = len(decoded_string)
    for stop_token in stop_tokens:
        stop_index = decoded_string.find(stop_token)
        if stop_index != -1 and stop_index < min_stop_index:
            min_stop_index = stop_index
    return decoded_string[:min_stop_index]


with open("/Users/nathanvogt/Downloads/generations_mbpp.json", "r") as f:
    generations = json.load(f)
generation = generations[0][0]
# print(generation)
generation = generation[len(prompt) :]
print(generation)
include_prompt = True
if not include_prompt:
    # slice prompt until second """
    prompt = prompt[: prompt.find('"""', 3) + 3]
p = prompt + _stop_at_stop_token(generation, stop_words)
