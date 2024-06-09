import json

train_failed_ids = [
    0,
    2,
    1,
    7,
    8,
    11,
    14,
    19,
    18,
    16,
    21,
    23,
    25,
    28,
    30,
    29,
    35,
    37,
    36,
    39,
    38,
    40,
    41,
    42,
    47,
    44,
    45,
    49,
    50,
    51,
    55,
    59,
    58,
    60,
    69,
    67,
    70,
    74,
    76,
    83,
    89,
    88,
    95,
    98,
    101,
    103,
    105,
    111,
    112,
    113,
    116,
    120,
    118,
    121,
    130,
    134,
    137,
    141,
    142,
    144,
    122,
    145,
    123,
    146,
    147,
    150,
    151,
    153,
    155,
    158,
    157,
    162,
    160,
    161,
    164,
    166,
    168,
    175,
    176,
    178,
    177,
    181,
    182,
    186,
    190,
    192,
    193,
    200,
    201,
    209,
    211,
    214,
    213,
    218,
    219,
    221,
    225,
    229,
    233,
    237,
    239,
    243,
    250,
    253,
    245,
    258,
    266,
    267,
    272,
    242,
    274,
    278,
    279,
    280,
    281,
    283,
    289,
    299,
    298,
    302,
    304,
    307,
    309,
    310,
    311,
    314,
    313,
    315,
    316,
    317,
    321,
    324,
    325,
    326,
    329,
    306,
    333,
    335,
    337,
    338,
    343,
    351,
    350,
    352,
    353,
    356,
    355,
    359,
    362,
    363,
    364,
    368,
    367,
    366,
    370,
]

path = "/Users/nathanvogt/Downloads/result_details.json"

failed_ids = []

with open(path, "r") as f:
    results = json.load(f)
mbpp_results = results["mbpp"]
for mbpp_result in mbpp_results.values():
    _, result = mbpp_result[0]
    task_id = result["task_id"]
    passed = result["passed"]
    if not passed:
        failed_ids.append(task_id)
print(failed_ids)
