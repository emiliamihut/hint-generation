from pathlib import Path
import csv
import json
import re
import sys

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# CONFIG

BASE = Path(
    "/export/home/acs/stud/m/maria_emilia.mihut/hint-generation"
)

ROOT = BASE / "matei-si-diamantele"

STATEMENT_FILE = (
    BASE
    / "drive_export"
    / "matei-si-diamantele"
    / "statement.json"
)

MODEL = "openai/gpt-oss-20b"

OUT_ROOT = (
    BASE
    / "text-level"
    / "descriptions-literary-gptoss20b"
    / "matei-si-diamantele"
)

OUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


SHARD_ID = int(sys.argv[1])
NUM_SHARDS = int(sys.argv[2])

if SHARD_ID < 0 or SHARD_ID >= NUM_SHARDS:
    raise RuntimeError(
        f"Invalid shard {SHARD_ID}/{NUM_SHARDS}"
    )


# FIND FINAL MATEI CSV

CANDIDATES = [
    ROOT / "predictions/final/matei_FINAL_EXTENDED_CLEAN.csv",
    ROOT / "predictions/final/matei_FINAL_PROB_ALL_PATCHED.csv",
    ROOT / "predictions/final/matei_FINAL_PROB_ALL.csv",
    ROOT / "predictions/final/matei_FINAL_CLEAN.csv",
]

INPUT_CSV = None

for p in CANDIDATES:
    if p.exists():
        INPUT_CSV = p
        break

if INPUT_CSV is None:
    raise FileNotFoundError(
        "No Matei final CSV found."
    )


# HELPERS

def sid(value):
    return str(int(float(value)))


def clean_html(text):
    text = re.sub(
        r"<script.*?</script>",
        " ",
        text,
        flags=re.S | re.I
    )

    text = re.sub(
        r"<style.*?</style>",
        " ",
        text,
        flags=re.S | re.I
    )

    text = re.sub(
        r"<[^>]+>",
        " ",
        text
    )

    text = (
        text
        .replace("&nbsp;", " ")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
        .replace("&quot;", '"')
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    text = re.sub(
        r"\n\s*\n\s*\n+",
        "\n\n",
        text
    )

    return text.strip()


def extract_statement(obj):
    parts = []
    seen = set()

    preferred = {
        "statement",
        "problem_statement",
        "problemStatement",
        "description",
        "body",
        "content",
        "task",
        "text",
        "input",
        "input_format",
        "inputFormat",
        "output",
        "output_format",
        "outputFormat",
        "constraints",
    }

    def add(x):
        if not isinstance(x, str):
            return

        x = clean_html(x)

        if len(x) < 20:
            return

        if x in seen:
            return

        seen.add(x)
        parts.append(x)

    def walk_preferred(x):
        if isinstance(x, dict):
            for k, v in x.items():

                if k in preferred:
                    if isinstance(v, str):
                        add(v)
                    else:
                        walk_all(v)

            for v in x.values():
                if isinstance(v, (dict, list)):
                    walk_preferred(v)

        elif isinstance(x, list):
            for v in x:
                walk_preferred(v)

    def walk_all(x):
        if isinstance(x, str):
            add(x)

        elif isinstance(x, dict):
            for v in x.values():
                walk_all(v)

        elif isinstance(x, list):
            for v in x:
                walk_all(v)

    walk_preferred(obj)

    if len("\n\n".join(parts)) < 100:
        walk_all(obj)

    return "\n\n".join(parts)


def find_source(row):
    submission_id = sid(
        row["submission_id"]
    )

    for col in [
        "file",
        "source_file",
        "source_path",
    ]:
        value = (
            row.get(col, "")
            or ""
        ).strip()

        if value:
            p = Path(value)

            if (
                p.exists()
                and p.suffix.lower()
                in {".cpp", ".c", ".java"}
            ):
                return p

    for ext in [
        ".cpp",
        ".java",
        ".c",
    ]:
        p = (
            ROOT
            / "submissions"
            / f"{submission_id}{ext}"
        )

        if p.exists():
            return p

    matches = [
        p
        for p in ROOT.rglob(
            f"{submission_id}.*"
        )
        if p.suffix.lower()
        in {".cpp", ".c", ".java"}
    ]

    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"Source missing for {submission_id}"
    )


def load_done(path):
    done = set()

    if not path.exists():
        return done

    with open(
        path,
        encoding="utf-8"
    ) as f:
        for line in f:

            line = line.strip()

            if not line:
                continue

            try:
                r = json.loads(line)

                if (
                    r.get("submission_id")
                    and r.get("description")
                ):
                    done.add(
                        str(r["submission_id"])
                    )

            except Exception:
                pass

    return done


def clean_special(text):
    text = re.sub(
        r"<\|[^>]+\|>",
        " ",
        text
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    text = re.sub(
        r"\n\s*\n\s*\n+",
        "\n\n",
        text
    )

    return text.strip()


def extract_final(
    raw_special,
    plain
):
    marker = (
        "<|channel|>"
        "final"
        "<|message|>"
    )

    if marker in raw_special:
        text = raw_special.rsplit(
            marker,
            1
        )[1]

        text = clean_special(text)

        if text:
            return text, "final_channel"

    if "<|message|>" in raw_special:
        text = raw_special.rsplit(
            "<|message|>",
            1
        )[1]

        text = clean_special(text)

        if text:
            return text, "last_message"

    return (
        plain.strip(),
        "decoded_fallback"
    )

# LOAD DATA

with open(
    INPUT_CSV,
    encoding="utf-8-sig",
    newline=""
) as f:
    rows = list(
        csv.DictReader(f)
    )

if len(rows) != 625:
    raise RuntimeError(
        f"Expected 625 Matei rows, got {len(rows)}"
    )


if not STATEMENT_FILE.exists():
    raise FileNotFoundError(
        f"Missing statement: {STATEMENT_FILE}"
    )

with open(
    STATEMENT_FILE,
    encoding="utf-8"
) as f:
    statement_json = json.load(f)

STATEMENT = extract_statement(
    statement_json
)

if len(STATEMENT) < 100:
    raise RuntimeError(
        "Invalid Matei statement.json"
    )


assigned_rows = [
    row
    for index, row in enumerate(rows)
    if index % NUM_SHARDS == SHARD_ID
]


print("=" * 90)
print("MATEI LITERARY DESCRIPTION GENERATION")
print("=" * 90)
print("INPUT:", INPUT_CSV)
print("STATEMENT:", STATEMENT_FILE)
print("TOTAL MATEI:", len(rows))
print("SHARD:", SHARD_ID, "/", NUM_SHARDS)
print("ASSIGNED:", len(assigned_rows))
print(flush=True)


# OUTPUT

JSONL_OUT = (
    OUT_ROOT
    / f"descriptions_shard{SHARD_ID}.jsonl"
)

CSV_OUT = (
    OUT_ROOT
    / f"descriptions_shard{SHARD_ID}.csv"
)

ERROR_OUT = (
    OUT_ROOT
    / f"errors_shard{SHARD_ID}.txt"
)

done = load_done(
    JSONL_OUT
)

print(
    "ALREADY DONE:",
    len(done),
    flush=True
)


# LOAD MODEL

print(
    "Loading tokenizer...",
    flush=True
)

tokenizer = (
    AutoTokenizer
    .from_pretrained(MODEL)
)


print(
    "Loading GPT-OSS-20B...",
    flush=True
)

model = (
    AutoModelForCausalLM
    .from_pretrained(
        MODEL,
        torch_dtype="auto",
        device_map="auto",
    )
)

model.eval()

print(
    "MODEL LOADED",
    flush=True
)


# PROMPT

SYSTEM_PROMPT = """
You inspect source code submitted to a programming contest.

Write a clear natural-language description of the solution that is
actually implemented by the submitted source code.

The description should read like a short technical explanation. 
It should be coherent prose, not a numbered checklist and
not internal reasoning.

Explain the solution from beginning to end: what information the program
reads, which important variables and data structures it maintains, how
the data or state is processed, how positions or values are updated, and
how the final result is obtained.

When the implementation contains recursion, graph exploration, dynamic
state updates, sorting, repeated formulas, geometric scans, greedy
choices, queues, stacks or other important mechanisms, explain how those
mechanisms are used in the actual program.

Describe the implementation that is really present in the code, even if
the submission is incorrect, incomplete or inefficient.

Do not replace it with the intended correct solution.
Do not give a predefined taxonomy label.
Do not output a numeric class.
Do not discuss what the contestant should have done.

The description must preserve enough algorithmic information that
another model, which will not see the source code, could later infer the
main implemented algorithmic strategy.

Return only the final description.
""".strip()


def generate_description(
    code,
    language
):
    user_prompt = f"""
PROBLEM:
Matei si diamantele

PROBLEM STATEMENT:
--------------------
{STATEMENT}
--------------------

SUBMISSION LANGUAGE:
{language}

SUBMITTED SOURCE CODE:
--------------------
{code}
--------------------

Write a concise but informative natural-language description of the
solution implemented in this source code.

Explain it as a coherent technical narrative following the program from
input to output. Mention the important data structures, loops, recursive
calls, state transitions, formulas, searches, geometric conditions or
selection rules that characterize this implementation.

Use approximately 1 to 3 short paragraphs.

Describe what the code actually does.
Do not assign a predefined algorithm label.
Do not propose a different solution.
""".strip()

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": user_prompt
        },
    ]

    inputs = (
        tokenizer
        .apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            reasoning_effort="low",
        )
    )

    input_len = (
        inputs["input_ids"]
        .shape[1]
    )

    inputs = {
        k: v.to(model.device)
        for k, v in inputs.items()
    }

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=700,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[
        0,
        input_len:
    ]

    raw_special = tokenizer.decode(
        generated,
        skip_special_tokens=False
    )

    plain = tokenizer.decode(
        generated,
        skip_special_tokens=True
    )

    return extract_final(
        raw_special,
        plain
    )


# GENERATE SHARD

with open(
    JSONL_OUT,
    "a",
    encoding="utf-8"
) as jout:

    for row in assigned_rows:

        submission_id = sid(
            row["submission_id"]
        )

        if submission_id in done:
            continue

        try:
            source = find_source(
                row
            )

            code = source.read_text(
                encoding="utf-8",
                errors="ignore"
            )

            language = (
                row.get(
                    "language",
                    ""
                )
                or source.suffix
            )

            description, status = (
                generate_description(
                    code,
                    language
                )
            )

            if not description.strip():
                raise RuntimeError(
                    "Empty description"
                )

            record = {
                "problem":
                    "matei-si-diamantele",

                "submission_id":
                    submission_id,

                "language":
                    language,

                "score":
                    row.get(
                        "score",
                        ""
                    ),

                "source_file":
                    str(source),

                "statement_file":
                    str(STATEMENT_FILE),

                "generation_model":
                    MODEL,

                "description_status":
                    status,

                "description":
                    description,
            }

            jout.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                )
                + "\n"
            )

            jout.flush()

            done.add(
                submission_id
            )

            print(
                f"[SHARD {SHARD_ID}] "
                f"[{len(done)}/{len(assigned_rows)}] "
                f"{submission_id} | "
                f"{status} | "
                f"{len(description)} chars",
                flush=True
            )

        except Exception as e:

            with open(
                ERROR_OUT,
                "a",
                encoding="utf-8"
            ) as ef:

                ef.write(
                    f"{submission_id}\t"
                    f"{type(e).__name__}\t"
                    f"{e}\n"
                )

            print(
                f"ERROR | "
                f"{submission_id} | "
                f"{type(e).__name__}: "
                f"{e}",
                flush=True
            )


# BUILD SHARD CSV

records = []

with open(
    JSONL_OUT,
    encoding="utf-8"
) as f:

    for line in f:

        line = line.strip()

        if not line:
            continue

        records.append(
            json.loads(line)
        )


FIELDS = [
    "problem",
    "submission_id",
    "language",
    "score",
    "source_file",
    "statement_file",
    "generation_model",
    "description_status",
    "description",
]


with open(
    CSV_OUT,
    "w",
    encoding="utf-8",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=FIELDS,
        extrasaction="ignore"
    )

    writer.writeheader()
    writer.writerows(records)


print()
print("=" * 90)
print(
    f"SHARD {SHARD_ID} FINISHED"
)
print("=" * 90)

print(
    "GENERATED:",
    len(records)
)

print(
    "JSONL:",
    JSONL_OUT
)

print(
    "CSV:",
    CSV_OUT
)

