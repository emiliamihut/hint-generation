from pathlib import Path
import csv
import json
import re
import sys

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


BASE = Path("/export/home/acs/stud/m/maria_emilia.mihut/hint-generation")

PROBLEM = "gigel-si-raspandirea-vestilor"
ROOT = BASE / PROBLEM

MODEL = "openai/gpt-oss-20b"

STATEMENT_FILE = (
    BASE
    / "drive_export"
    / PROBLEM
    / "statement.json"
)

SHARD_ID = int(sys.argv[1])
NUM_SHARDS = int(sys.argv[2])
MODE = sys.argv[3] if len(sys.argv) > 3 else "full"

if MODE not in {"pilot", "full"}:
    raise RuntimeError("Mode must be pilot or full")

if SHARD_ID < 0 or SHARD_ID >= NUM_SHARDS:
    raise RuntimeError(
        f"Invalid shard {SHARD_ID}/{NUM_SHARDS}"
    )


OUT_ROOT = (
    BASE
    / "text-level"
    / "descriptions-literary-gptoss20b-mechanics-only"
    / PROBLEM
)

if MODE == "pilot":
    OUT_ROOT = OUT_ROOT / "pilot"

OUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


CANDIDATES = [
    ROOT
    / "predictions"
    / "final"
    / "raspandirea_FINAL_CLEAN.csv",

    ROOT
    / "predictions"
    / "final"
    / "raspandirea_FINAL_FAIR.csv",
]


INPUT_CSV = None

for path in CANDIDATES:
    if path.exists():
        INPUT_CSV = path
        break

if INPUT_CSV is None:
    raise FileNotFoundError(
        "Could not find final Raspandirea CSV"
    )


def sid(value):
    return str(int(float(value)))


def clean_html(text):
    text = re.sub(
        r"<script.*?</script>",
        " ",
        text,
        flags=re.S | re.I,
    )

    text = re.sub(
        r"<style.*?</style>",
        " ",
        text,
        flags=re.S | re.I,
    )

    text = re.sub(
        r"<[^>]+>",
        " ",
        text,
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
        text,
    )

    text = re.sub(
        r"\n\s*\n\s*\n+",
        "\n\n",
        text,
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

    def add(value):
        if not isinstance(value, str):
            return

        value = clean_html(value)

        if len(value) < 20:
            return

        if value in seen:
            return

        seen.add(value)
        parts.append(value)

    def walk_all(value):
        if isinstance(value, str):
            add(value)

        elif isinstance(value, dict):
            for child in value.values():
                walk_all(child)

        elif isinstance(value, list):
            for child in value:
                walk_all(child)

    def walk_preferred(value):
        if isinstance(value, dict):

            for key, child in value.items():

                if key in preferred:

                    if isinstance(child, str):
                        add(child)

                    else:
                        walk_all(child)

            for child in value.values():

                if isinstance(
                    child,
                    (dict, list),
                ):
                    walk_preferred(child)

        elif isinstance(value, list):

            for child in value:
                walk_preferred(child)

    walk_preferred(obj)

    if len("\n\n".join(parts)) < 100:
        walk_all(obj)

    return "\n\n".join(parts)


def find_source(row):
    submission_id = sid(
        row["submission_id"]
    )

    for column in [
        "file",
        "source_file",
        "source_path",
    ]:

        value = (
            row.get(column, "")
            or ""
        ).strip()

        if not value:
            continue

        path = Path(value)

        if (
            path.exists()
            and path.suffix.lower()
            in {".cpp", ".c", ".java"}
        ):
            return path


    search_roots = [
        ROOT / "submissions",

        BASE
        / "drive_export"
        / PROBLEM
        / "solutions",
    ]


    for search_root in search_roots:

        for ext in [
            ".cpp",
            ".java",
            ".c",
        ]:

            path = (
                search_root
                / f"{submission_id}{ext}"
            )

            if path.exists():
                return path


    matches = [
        path

        for path in ROOT.rglob(
            f"{submission_id}.*"
        )

        if path.suffix.lower()
        in {
            ".cpp",
            ".c",
            ".java",
        }
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
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            try:

                record = json.loads(line)

                if (
                    record.get("submission_id")
                    and record.get("description")
                ):

                    done.add(
                        str(
                            record["submission_id"]
                        )
                    )

            except Exception:
                pass


    return done


def clean_special(text):
    text = re.sub(
        r"<\|[^>]+\|>",
        " ",
        text,
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    text = re.sub(
        r"\n\s*\n\s*\n+",
        "\n\n",
        text,
    )

    return text.strip()


def extract_final(
    raw_special,
    plain,
):
    marker = (
        "<|channel|>"
        "final"
        "<|message|>"
    )


    if marker in raw_special:

        text = raw_special.rsplit(
            marker,
            1,
        )[1]

        text = clean_special(text)

        if text:
            return (
                text,
                "final_channel",
            )


    if "<|message|>" in raw_special:

        text = raw_special.rsplit(
            "<|message|>",
            1,
        )[1]

        text = clean_special(text)

        if text:
            return (
                text,
                "last_message",
            )


    return (
        plain.strip(),
        "decoded_fallback",
    )


with open(
    INPUT_CSV,
    encoding="utf-8-sig",
    newline="",
) as f:

    rows = list(
        csv.DictReader(f)
    )


if len(rows) != 729:
    raise RuntimeError(
        f"Expected 729 rows, found {len(rows)}"
    )


if not STATEMENT_FILE.exists():
    raise FileNotFoundError(
        STATEMENT_FILE
    )


with open(
    STATEMENT_FILE,
    encoding="utf-8",
) as f:

    statement_json = json.load(f)


STATEMENT = extract_statement(
    statement_json
)


if len(STATEMENT) < 100:
    raise RuntimeError(
        "Invalid statement.json"
    )


assigned_rows = [
    row

    for index, row
    in enumerate(rows)

    if index % NUM_SHARDS
    == SHARD_ID
]


if MODE == "pilot":
    assigned_rows = assigned_rows[:3]


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
    f"Problem: {PROBLEM}\n"
    f"Mode: {MODE}\n"
    f"Input: {INPUT_CSV}\n"
    f"Statement: {STATEMENT_FILE}\n"
    f"Shard: {SHARD_ID}/{NUM_SHARDS}\n"
    f"Assigned: {len(assigned_rows)}\n"
    f"Already done: {len(done)}",
    flush=True,
)


print(
    "Loading tokenizer...",
    flush=True,
)


tokenizer = (
    AutoTokenizer
    .from_pretrained(MODEL)
)


print(
    "Loading GPT-OSS-20B...",
    flush=True,
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
    "Model loaded.",
    flush=True,
)


SYSTEM_PROMPT = """
You are explaining source code written for a programming contest.

Write a simple and clear description of what the submitted program does.

The explanation should be beginner-friendly. Imagine you are explaining
the code to a student who understands basic programming but does not know
advanced algorithms very well.

Follow the program from input to output.

Explain the important data structures in simple words and say what they
store.

Explain what the main loops or recursive calls do.

Explain how values, positions, nodes or states change while the program
runs.

If the code uses a queue, stack, array, matrix, set, map or another
important structure, mention it and explain how it is used.

If elements are stored for later processing, explain which element is
processed next and when new elements are added.

If an array stores information such as visited state, distance, time,
cost or another value, explain how it is initialized and how it changes.

If recursion is used, explain what one function call does, when it calls
itself again and when it stops.

If sorting is actually performed, explain what is sorted and how the
sorted order is later used.

If the program repeatedly chooses one possible next element, explain
what choice it makes.

If the program scans a grid, graph, list, matrix or range of values,
explain what is scanned and what condition is checked.

Use concrete details from the source code, but explain them in simple
language.

Do not make the description unnecessarily mathematical or complicated.

Do not use vague sentences such as "the graph is processed" when the
source code shows exactly how it is processed.

Most importantly, NEVER write the name of the algorithm used by the
program. This rule has priority over all other instructions.

Do not write the conventional name of the algorithm, the name of an
algorithm family, an algorithmic technique name, or an abbreviation.

Do not write names such as breadth-first search, depth-first search,
Dijkstra, Floyd-Warshall, dynamic programming, greedy, or any other
standard algorithm name.

If you recognize the algorithm, keep that recognition private.

Only describe the concrete mechanics: the data structures, processing
order, loops, recursive calls, conditions and update rules.

For example, do not write "the program performs a breadth-first search".
Instead explain that the program keeps a queue, removes the first node,
checks its neighbours, updates newly reached nodes and adds them to the
back of the queue.

The algorithm must only be recognizable from the mechanics described in
the text.

Describe the data structures, processing order, loops, recursive calls,
conditions and update rules instead of giving the algorithm a name.

Describe what the submitted code actually does, even if it is incorrect,
incomplete or inefficient.

Do not correct the code.

Do not replace it with a better solution.

Do not give a taxonomy label.

Do not output a numeric class.

Write one to three short paragraphs.

Use simple and natural technical English.

Return only the description.
""".strip()


def generate_description(
    code,
    language,
):

    user_prompt = f"""
PROBLEM:
Gigel si raspandirea vestilor

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

Explain what this exact program does in simple, beginner-friendly
language.

Start with how the graph is stored.

Then explain how the program starts processing the graph.

Describe what happens to one node when it is processed.

Explain how its neighbours are checked.

If some neighbours are stored to be processed later, explain where they
are stored and in what order they will be processed.

If the program uses an array for visited nodes, distances, times or other
values, explain what the array contains and how its values change.

If the program keeps a separate counter or time value, explain when that
value changes.

If several nodes are processed as one group or level, explain how the
program knows when one group ends.

If a node is stored together with another value, explain what both values
mean.

If recursion is used, explain what one recursive call does and when it
creates another call.

If the program runs the same type of processing several times, explain
when a new run starts.

If there is a matrix with several nested loops, explain what the loops do
to the matrix values.

Finally, explain how the program obtains the value that it prints.

Keep the explanation concrete but easy to understand.

Do not name the algorithm under any circumstances.

Do not write a standard algorithm name, algorithm family name,
algorithmic technique name, or abbreviation.

If you recognize the algorithm, keep its name private and describe only
what the code does step by step.

Before returning the answer, check that you have not written the name or
abbreviation of any algorithm anywhere in the description. If you have,
rewrite that sentence using only the concrete mechanics.

Another model must infer the algorithm later from the description itself.

Do not infer behavior that is not present in the source code.

Return only the description.
""".strip()


    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_prompt,
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
        key: value.to(model.device)

        for key, value
        in inputs.items()
    }


    with torch.inference_mode():

        outputs = model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )


    generated = outputs[
        0,
        input_len:
    ]


    raw_special = tokenizer.decode(
        generated,
        skip_special_tokens=False,
    )


    plain = tokenizer.decode(
        generated,
        skip_special_tokens=True,
    )


    return extract_final(
        raw_special,
        plain,
    )


with open(
    JSONL_OUT,
    "a",
    encoding="utf-8",
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
                errors="ignore",
            )


            language = (
                row.get("language", "")
                or source.suffix.lstrip(".")
            )


            description, status = (
                generate_description(
                    code,
                    language,
                )
            )


            if not description.strip():

                raise RuntimeError(
                    "Empty description"
                )


            record = {
                "problem":
                    PROBLEM,

                "submission_id":
                    submission_id,

                "language":
                    language,

                "score":
                    row.get(
                        "score",
                        "",
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
                    ensure_ascii=False,
                )
                + "\n"
            )


            jout.flush()


            done.add(
                submission_id
            )


            print(
                f"[{SHARD_ID}] "
                f"{len(done)}/{len(assigned_rows)} | "
                f"{submission_id} | "
                f"{status} | "
                f"{len(description)} chars",
                flush=True,
            )


        except Exception as e:

            with open(
                ERROR_OUT,
                "a",
                encoding="utf-8",
            ) as ef:

                ef.write(
                    f"{submission_id}\t"
                    f"{type(e).__name__}\t"
                    f"{e}\n"
                )


            print(
                f"ERROR {submission_id} | "
                f"{type(e).__name__}: {e}",
                flush=True,
            )


records = []


with open(
    JSONL_OUT,
    encoding="utf-8",
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
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=FIELDS,
        extrasaction="ignore",
    )

    writer.writeheader()

    writer.writerows(
        records
    )


print(
    f"Finished shard {SHARD_ID}. "
    f"Generated {len(records)} descriptions."
)

print(
    "JSONL:",
    JSONL_OUT,
)

print(
    "CSV:",
    CSV_OUT,
)

