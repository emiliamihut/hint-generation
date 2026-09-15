from pathlib import Path
import csv
import json
import sys

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


BASE = Path("/export/home/acs/stud/m/maria_emilia.mihut/hint-generation")

MODEL_NAME = "openai/gpt-oss-20b"
SHARD_ID = int(sys.argv[1])

input_file = (
    BASE
    / "text-level"
    / "MATEI_TEXT_EXPERIMENT_FINAL"
    / "data"
    / f"descriptions_shard{SHARD_ID}.jsonl"
)

output_folder = (
    BASE
    / "text-level"
    / "broad-taxonomy"
    / "matei-si-diamantele"
    / "text"
)

output_folder.mkdir(parents=True, exist_ok=True)

output_file = (
    output_folder
    / f"matei_text_broad_shard{SHARD_ID}.csv"
)


labels = {
    "0": "GEOMETRIC_SCAN",
    "1": "BFS_RADIUS",
    "2": "DFS_RADIUS",
    "3": "SORTED_MANHATTAN",
    "4": "DP_RADIUS",
    "5": "GREEDY_NEIGHBOR_WALK",
    "6": "UNKNOWN",
}


class_description = """
0 = GEOMETRIC_SCAN
The program directly checks cells or coordinates around an area.
This includes scanning by Manhattan distance, rows, rings or a
bounding rectangle.

1 = BFS_RADIUS
The program explores nearby positions using a FIFO queue.
Positions are processed level by level.

2 = DFS_RADIUS
The program explores neighbouring positions depth-first,
using recursion or a stack.

3 = SORTED_MANHATTAN
The program computes Manhattan distances for many positions
and sorts or orders these distances before obtaining the answer.

4 = DP_RADIUS
The program builds results from previously computed states,
using a table, recurrence or memoization.

5 = GREEDY_NEIGHBOR_WALK
The program repeatedly chooses a locally preferred neighbour
and moves step by step.

6 = UNKNOWN
The main strategy does not fit any of the classes above.
Do not choose UNKNOWN only because the solution is buggy or incomplete.
""".strip()


# Read descriptions

rows = []

with open(input_file, encoding="utf-8") as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))


print(f"Shard {SHARD_ID}: {len(rows)} descriptions")
print("Loading model...")


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto"
)

model.eval()

print("Model loaded.")


# Each possible answer is one digit: 0, 1, ..., 6

choice_token_ids = {}

for choice in labels:
    token_ids = tokenizer.encode(
        choice,
        add_special_tokens=False
    )

    if len(token_ids) != 1:
        raise ValueError(
            f"{choice} is not represented by one token"
        )

    choice_token_ids[choice] = token_ids[0]


results = []


for i, row in enumerate(rows, start=1):

    submission_id = str(row["submission_id"])
    description = row["description"].strip()


    prompt = f"""
Classify the main algorithm described below.

Use ONLY the natural-language description.
You do not have access to the source code.

Choose the broad algorithm family.
Small implementation differences should not create different classes.

Classes:

{class_description}

Description:

{description}

Return exactly one digit from 0 to 6.
""".strip()


    messages = [
        {
            "role": "user",
            "content": prompt
        },
        {
            "role": "assistant",
            "content": "Answer: "
        }
    ]


    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
        continue_final_message=True
    )

    inputs = {
        name: value.to(model.device)
        for name, value in inputs.items()
    }


    with torch.inference_mode():
        output = model(**inputs)


    next_token_logits = output.logits[0, -1]


    class_logits = []

    for choice in labels:
        token_id = choice_token_ids[choice]
        class_logits.append(
            next_token_logits[token_id]
        )


    class_logits = torch.stack(class_logits)

    probabilities = torch.softmax(
        class_logits,
        dim=0
    )


    probability_by_class = {}

    for index, choice in enumerate(labels):
        probability_by_class[choice] = float(
            probabilities[index].item()
        )


    ranked = sorted(
        probability_by_class.items(),
        key=lambda item: item[1],
        reverse=True
    )


    best_choice, best_probability = ranked[0]
    second_choice, second_probability = ranked[1]


    result = {
        "submission_id": submission_id,
        "score": row.get("score", ""),
        "language": row.get("language", ""),

        "text_prediction": labels[best_choice],
        "text_probability": best_probability,

        "text_second_algorithm": labels[second_choice],
        "text_second_probability": second_probability,

        "p_GEOMETRIC_SCAN": probability_by_class["0"],
        "p_BFS_RADIUS": probability_by_class["1"],
        "p_DFS_RADIUS": probability_by_class["2"],
        "p_SORTED_MANHATTAN": probability_by_class["3"],
        "p_DP_RADIUS": probability_by_class["4"],
        "p_GREEDY_NEIGHBOR_WALK": probability_by_class["5"],
        "p_UNKNOWN": probability_by_class["6"],
    }

    results.append(result)


    print(
        f"[{SHARD_ID}] "
        f"{i}/{len(rows)} | "
        f"{submission_id} | "
        f"{labels[best_choice]} "
        f"{best_probability:.4f}"
    )


with open(
    output_file,
    "w",
    encoding="utf-8",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(results[0].keys())
    )

    writer.writeheader()
    writer.writerows(results)


print()
print("Finished.")
print("Predictions:", len(results))
print("Saved:", output_file)

