from pathlib import Path
import csv
import json
import sys

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


BASE = Path("/export/home/acs/stud/m/maria_emilia.mihut/hint-generation")

PROBLEM = "gigel-si-raspandirea-vestilor"
MODEL = "openai/gpt-oss-20b"

SHARD_ID = int(sys.argv[1])

DESCRIPTION_FILE = (
    BASE
    / "text-level"
    / "descriptions-literary-gptoss20b-mechanics-only"
    / PROBLEM
    / f"descriptions_shard{SHARD_ID}.jsonl"
)

OUT_ROOT = (
    BASE
    / "text-level"
    / "broad-taxonomy"
    / PROBLEM
    / "text"
)

OUT_ROOT.mkdir(parents=True, exist_ok=True)


LABELS = {
    "0": "BFS",
    "1": "DIJKSTRA",
    "2": "DFS",
    "3": "FLOYD_WARSHALL",
    "4": "UNKNOWN",
}


CLASS_GUIDE = """
0 = BFS
The described implementation explores the graph using a FIFO queue or an
equivalent level-by-level traversal. Different ways of storing distance
or time still belong to the same BFS class.

1 = DIJKSTRA
The implementation keeps tentative distances and repeatedly processes the
currently smallest-distance or best candidate node, usually using a
priority queue or another minimum-selection mechanism. It then tries to
improve neighbour distances.

2 = DFS
The implementation explores one branch deeper before returning to another
branch, usually through recursive calls or an explicit stack.

3 = FLOYD_WARSHALL
The implementation stores pairwise information in a matrix and updates it
using three nested loops over vertices, considering intermediate vertices.

4 = UNKNOWN
The main strategy genuinely does not fit any concrete class above.
Do not choose UNKNOWN simply because the implementation is buggy,
incomplete, inefficient, or unusual.
""".strip()


with open(DESCRIPTION_FILE, encoding="utf-8") as f:
    rows = [
        json.loads(line)
        for line in f
        if line.strip()
    ]


print(
    f"Shard {SHARD_ID}: {len(rows)} descriptions",
    flush=True
)


print("Loading tokenizer...", flush=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL)


print("Loading GPT-OSS-20B...", flush=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype="auto",
    device_map="auto",
)

model.eval()

print("Model loaded.", flush=True)


token_ids = {}

for choice in LABELS:
    ids = tokenizer.encode(
        choice,
        add_special_tokens=False
    )

    if len(ids) != 1:
        raise RuntimeError(
            f"Choice {choice} is not one token: {ids}"
        )

    token_ids[choice] = ids[0]


results = []


for index, row in enumerate(rows, 1):

    submission_id = str(row["submission_id"])
    description = row["description"].strip()


    prompt = f"""
Classify the MAIN algorithmic strategy using ONLY the natural-language
description below.

You do not have access to the source code.

Focus on the main algorithm family, not small implementation details.

For example, all FIFO queue-based breadth-first variants belong to the
single BFS class, regardless of how distance or time is stored.

Use concrete mechanics from the description:
- FIFO queue or level-by-level traversal;
- priority queue or repeated minimum-distance selection;
- recursive or stack-based depth-first traversal;
- matrix updates with three nested vertex loops.

A buggy or partial implementation can still belong to a concrete class.

Classes:

{CLASS_GUIDE}

DESCRIPTION:
--------------------
{description}
--------------------

Return exactly one digit from 0 to 4.
""".strip()


    messages = [
        {
            "role": "user",
            "content": prompt,
        },
        {
            "role": "assistant",
            "content": "Answer: ",
        },
    ]


    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
        continue_final_message=True,
    )


    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }


    with torch.inference_mode():
        output = model(**inputs)


    logits = output.logits[0, -1]

    choices = list(LABELS.keys())

    class_logits = torch.stack([
        logits[token_ids[c]]
        for c in choices
    ])

    prob_tensor = torch.softmax(
        class_logits,
        dim=0
    )


    probs = {
        c: float(prob_tensor[i].item())
        for i, c in enumerate(choices)
    }


    ordered = sorted(
        probs.items(),
        key=lambda x: x[1],
        reverse=True
    )


    top1, p1 = ordered[0]
    top2, p2 = ordered[1]


    result = {
        "problem": PROBLEM,
        "submission_id": submission_id,
        "language": row.get("language", ""),
        "score": row.get("score", ""),

        "text_prediction": LABELS[top1],
        "text_probability": p1,

        "text_second_algorithm": LABELS[top2],
        "text_second_probability": p2,

        "p_BFS": probs["0"],
        "p_DIJKSTRA": probs["1"],
        "p_DFS": probs["2"],
        "p_FLOYD_WARSHALL": probs["3"],
        "p_UNKNOWN": probs["4"],
    }

    results.append(result)


    print(
        f"[{SHARD_ID}] "
        f"{index}/{len(rows)} | "
        f"{submission_id} | "
        f"{LABELS[top1]} {p1:.4f} | "
        f"2nd={LABELS[top2]} {p2:.4f}",
        flush=True
    )


OUT = (
    OUT_ROOT
    / f"raspandirea_text_broad_shard{SHARD_ID}.csv"
)


with open(
    OUT,
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


print(
    f"Finished shard {SHARD_ID}: {len(results)}",
    flush=True
)

print("Output:", OUT)

