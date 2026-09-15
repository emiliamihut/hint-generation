# LLM-Based Programming Hint Generation

This project was developed as a team project during **NLP Summer School 2026**. We explored how Large Language Models can be used for educational code debugging without simply giving students the answer.

The main goal was to generate short Socratic hints that guide a student towards the issue while still leaving the actual debugging and reasoning to them.

## What we worked on

The project covered the full pipeline:
- collecting and organizing competitive-programming data;
- cleaning, deduplication and similarity analysis;
- running submissions and processing execution results;
- algorithm profiling directly from code;
- text-level algorithm profiling;
- several iterations of hint-generation prompts;
- GPT-OSS-20B vs. Qwen3.5-9B comparisons;
- manual hint review and leakage analysis;
- trace-level experiments using initial and improved solutions.

## Data

We used programming problems from **HackerRank** and **Infoarena**.

HackerRank:
- 6 problems;
- 4,738 raw scraped submissions;
- 3,223 cleaned and deduplicated submissions;
- 61,618 test execution logs in the final experimental dataset.

Infoarena:
- 66 problems;
- 124,716 indexed active submissions;
- 160,130 submission metadata records;
- 217,313 test-level outcome files.

The repository does not publish the private real-username mappings, raw student submissions or private input/output test files.

## Algorithm profiling

We compared two approaches:

```text
Source Code -> LLM -> Algorithm Label
```

and:

```text
Source Code -> LLM -> Text Description -> Algorithm Label
```

The HackerRank text-level comparison covered 1,354 submissions from two problems.

Agreement between the two routes was around **93.5%** for the general taxonomy and around **82%** for the detailed taxonomy.

## Hint generation

We kept the main prompt iterations used during the project:

- **V1** — early experiments;
- **V2** — problem statement + student code;
- **V3** — intermediate experiments using execution information;
- **V7** — final setup using private failed-test evidence.

The working V4 folders correspond to the final V7 experiment and are grouped under V7 here.

The main HackerRank experiments compared **GPT-OSS-20B** and **Qwen3.5-9B**.

## Manual evaluation

Hints were manually checked for:
- private evidence leakage;
- exact bug / edge-case disclosure;
- strategy disclosure;
- implementation-specific information;
- concrete fixes;
- excessive vagueness.

Each hint received a final **GOOD** or **BAD** verdict.

## Main comparison

Across 89 comparable submissions:

| Model | Condition | GOOD rate |
|---|---|---:|
| GPT-OSS-20B | V2 | 74.16% |
| GPT-OSS-20B | V7 | 61.80% |
| Qwen3.5-9B | V2 | 37.08% |
| Qwen3.5-9B | V7 | 46.07% |

The effect of additional execution context was model-dependent. More context did not automatically produce a better educational hint.

## Head-to-head

For decisive model disagreements:
- GPT-OSS-20B: **77.01%**
- Qwen3.5-9B: **22.99%**

## Human agreement

A subset of 90 hints was independently reviewed by two annotators:
- raw agreement: **84.44%**
- Cohen's kappa: **0.68**

## Trace-level experiment

We also evaluated 100 initial/improved solution pairs.

```text
32 BAD  -> GOOD
31 GOOD -> BAD
17 GOOD -> GOOD
20 BAD  -> BAD
```

The overall success rate changed very little, but the type of failure changed:

```text
Exact bug leakage: 45 -> 22
Too vague:           7 -> 0
Strategy revealed:   2 -> 21
Concrete fix:        1 -> 32
```

This showed that extra information can change the model's failure mode instead of simply making the hint better.

## Main difficulties

The hardest part was controlling specificity.

Too little context can produce generic hints. Too much context can reveal the exact optimization, edge case or correction.

Timeout cases and partially correct solutions were especially difficult.

## Possible next steps

- richer per-test execution evidence;
- official expected outputs where available;
- separate prompting for Wrong Answer, Timeout and Runtime Error;
- better support for partial solutions;
- progressive hint generation;
- automatic leakage checks;
- more problems and more models.

## Team

**Maria-Emilia Mihuț**  
**Alexandru-Ștefan Nuță**

Coordinator: **Radu Iacob**

University POLITEHNICA of Bucharest  
NLP Summer School 2026

## Main takeaway

LLMs can generate useful programming hints, but finding the bug is only one part of the problem. For educational use, the harder part is turning that diagnosis into guidance that still leaves the reasoning to the student.
