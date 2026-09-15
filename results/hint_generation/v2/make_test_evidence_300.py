from pathlib import Path
import csv
import json


FINAL_PROJECT = Path(
    "/mnt/d/Hint-generation/FINAL_PROJECT"
)

SELECTED_FOLDER = (
    FINAL_PROJECT
    / "generated hints v2"
)

DRIVE_EXPORT_CANDIDATES = [
    Path("/mnt/d/Hint-generation/hackerrank_scraper/drive_export"),
    FINAL_PROJECT / "drive_export",
]

PROBLEMS = [
    "matei-si-diamantele",
    "gigel-si-raspandirea-vestilor",
    "gigel-si-sosetele",
    "gigel-si-telefonul",
    "organizare-mafiota",
    "cutiile-magice",
]


def find_drive_export():

    for path in DRIVE_EXPORT_CANDIDATES:

        if path.exists():
            return path

    raise FileNotFoundError(
        "Nu am gasit drive_export."
    )


DRIVE_EXPORT = find_drive_export()

OUTPUT_FOLDER = (
    FINAL_PROJECT
    / "test evidence v1"
)

OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


def clean(value):

    if value is None:
        return ""

    return str(value).strip()


def load_csv_matches(
    folder,
    submission_id
):

    matches = []

    if not folder.exists():
        return matches

    for path in folder.rglob("*.csv"):

        try:

            with open(
                path,
                encoding="utf-8-sig",
                newline=""
            ) as f:

                reader = csv.DictReader(f)

                if not reader.fieldnames:
                    continue

                for row in reader:

                    row_id = clean(
                        row.get("submission_id")
                    )

                    if row_id == submission_id:

                        item = dict(row)

                        item["_source_file"] = str(path)

                        matches.append(item)

        except Exception:
            continue

    return matches


def load_json_matches(
    folder,
    submission_id
):

    matches = []

    if not folder.exists():
        return matches


    for path in folder.rglob("*.json"):

        try:

            with open(
                path,
                encoding="utf-8"
            ) as f:

                data = json.load(f)

        except Exception:
            continue


        if isinstance(data, dict):

            candidates = [data]

        elif isinstance(data, list):

            candidates = data

        else:

            continue


        for item in candidates:

            if not isinstance(
                item,
                dict
            ):
                continue

            row_id = clean(
                item.get("submission_id")
            )

            if row_id == submission_id:

                copied = dict(item)

                copied["_source_file"] = str(path)

                matches.append(copied)


    return matches


def load_jsonl_matches(
    folder,
    submission_id
):

    matches = []

    if not folder.exists():
        return matches


    for path in folder.rglob("*.jsonl"):

        try:

            with open(
                path,
                encoding="utf-8"
            ) as f:

                for line in f:

                    line = line.strip()

                    if not line:
                        continue

                    try:
                        item = json.loads(line)

                    except Exception:
                        continue

                    if not isinstance(
                        item,
                        dict
                    ):
                        continue

                    row_id = clean(
                        item.get("submission_id")
                    )

                    if row_id == submission_id:

                        copied = dict(item)

                        copied["_source_file"] = str(path)

                        matches.append(copied)

        except Exception:
            continue


    return matches


def get_test_number(row):

    for key in [
        "test",
        "test_id",
        "test_number",
        "case",
        "case_id",
    ]:

        value = clean(
            row.get(key)
        )

        if value:
            return value

    return ""


def normalize_test_row(row):

    wanted = [
        "test",
        "test_id",
        "test_number",
        "status",
        "verdict",
        "passed",
        "execution_time_seconds",
        "compile_time_seconds",
        "time",
        "runtime",
        "memory",
        "memory_kb",
        "return_code",
        "input",
        "test_input",
        "expected_output",
        "actual_output",
        "output",
        "error",
        "stderr",
        "points",
        "score",
    ]

    result = {}

    for key in wanted:

        value = row.get(key)

        if value is None:
            continue

        value = str(value).strip()

        if value == "":
            continue

        result[key] = value


    result["_source_file"] = row.get(
        "_source_file",
        ""
    )

    return result


def merge_test_rows(
    official_rows,
    local_rows
):

    merged = {}


    for source_name, rows in [
        ("details", official_rows),
        ("details_local", local_rows),
    ]:

        for index, row in enumerate(rows):

            test_number = get_test_number(
                row
            )

            if not test_number:

                test_number = (
                    f"{source_name}_{index + 1}"
                )


            if test_number not in merged:

                merged[test_number] = {
                    "test": test_number
                }


            normalized = normalize_test_row(
                row
            )


            for key, value in normalized.items():

                if key == "_source_file":

                    merged[test_number][
                        f"{source_name}_source"
                    ] = value

                elif (
                    key not in merged[test_number]
                    or not merged[test_number][key]
                ):

                    merged[test_number][key] = value


    tests = list(
        merged.values()
    )


    def sort_key(item):

        value = str(
            item.get("test", "")
        )

        try:
            return (0, int(value))

        except Exception:
            return (1, value)


    tests.sort(
        key=sort_key
    )

    return tests


def summarize_tests(tests):

    total = len(tests)

    passed = 0
    failed = 0

    times = []


    for test in tests:

        passed_value = clean(
            test.get("passed")
        ).lower()

        status = clean(
            test.get(
                "status",
                test.get("verdict", "")
            )
        ).lower()


        is_passed = (
            passed_value
            in ["true", "1", "yes", "passed", "pass", "ok"]
            or status
            in ["passed", "pass", "accepted", "ok"]
        )

        is_failed = (
            passed_value
            in ["false", "0", "no", "failed", "fail"]
            or any(
                word in status
                for word in [
                    "wrong",
                    "fail",
                    "timeout",
                    "runtime",
                    "error",
                    "tle",
                    "wa",
                    "re",
                ]
            )
        )


        if is_passed:
            passed += 1

        elif is_failed:
            failed += 1


        time_value = (
            test.get("execution_time_seconds")
            or test.get("runtime")
            or test.get("time")
        )


        if time_value:

            try:
                times.append(
                    float(time_value)
                )

            except Exception:
                pass


    summary = {
        "tests_found": total,
        "passed": passed,
        "failed": failed,
    }


    if times:

        summary[
            "total_execution_time_seconds"
        ] = sum(times)

        summary[
            "average_execution_time_seconds"
        ] = sum(times) / len(times)

        summary[
            "maximum_execution_time_seconds"
        ] = max(times)


    return summary


for problem in PROBLEMS:

    selected_file = (
        SELECTED_FOLDER
        / f"{problem}_socratic_50.csv"
    )

    problem_root = (
        DRIVE_EXPORT
        / problem
    )

    details_folder = (
        problem_root
        / "details"
    )

    details_local_folder = (
        problem_root
        / "details_local"
    )


    if not selected_file.exists():

        print(
            problem,
            "-> selected CSV missing"
        )

        continue


    with open(
        selected_file,
        encoding="utf-8-sig",
        newline=""
    ) as f:

        selected_rows = list(
            csv.DictReader(f)
        )


    output_file = (
        OUTPUT_FOLDER
        / f"{problem}_test_evidence_50.jsonl"
    )


    found = 0
    missing = 0


    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as out:


        for row in selected_rows:

            submission_id = clean(
                row.get("submission_id")
            )


            official_rows = []

            official_rows.extend(
                load_csv_matches(
                    details_folder,
                    submission_id
                )
            )

            official_rows.extend(
                load_json_matches(
                    details_folder,
                    submission_id
                )
            )

            official_rows.extend(
                load_jsonl_matches(
                    details_folder,
                    submission_id
                )
            )


            local_rows = []

            local_rows.extend(
                load_csv_matches(
                    details_local_folder,
                    submission_id
                )
            )

            local_rows.extend(
                load_json_matches(
                    details_local_folder,
                    submission_id
                )
            )

            local_rows.extend(
                load_jsonl_matches(
                    details_local_folder,
                    submission_id
                )
            )


            tests = merge_test_rows(
                official_rows,
                local_rows
            )


            if tests:
                found += 1

            else:
                missing += 1


            item = {
                "problem": problem,
                "submission_id": submission_id,
                "summary": summarize_tests(
                    tests
                ),
                "tests": tests,
            }


            out.write(
                json.dumps(
                    item,
                    ensure_ascii=False
                )
                + "\n"
            )


    print()
    print(problem)
    print("selected:", len(selected_rows))
    print("with test evidence:", found)
    print("without test evidence:", missing)
    print("output:", output_file)


print()
print("DONE")
print("DRIVE_EXPORT:", DRIVE_EXPORT)
print("OUTPUT:", OUTPUT_FOLDER)
