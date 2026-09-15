from pathlib import Path
import argparse
import csv
import json
import os
import time
import traceback

ROOT = Path(
    r"C:\RSOS\RSOS_EXPERIMENTS_WF2_PROSPECTIVE_CROSS_SYSTEM_CONFIRMATION"
)

PLANROOT = (
    ROOT
    / "02_INPUTS"
    / "02_ACTIVE_CAUSAL_PLAN"
)

RAW = (
    ROOT
    / "04_RAW"
    / "ACTIVE_CAUSAL"
)

PROMPTS = (
    PLANROOT
    / "WF2_ACTIVE_CAUSAL_FROZEN_PROMPTS.csv"
)

PLAN = (
    PLANROOT
    / "WF2_ACTIVE_CAUSAL_EXECUTION_PLAN.jsonl"
)

RETRY_WAIT = [5,15,45]


def prompt_map():

    out = {}

    with PROMPTS.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        for row in csv.DictReader(f):

            out[
                (
                    int(row["SEED"]),
                    row["CONDITION"]
                )
            ] = row

    return out


def load_plan():

    rows = []

    with PLAN.open(
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            if line.strip():
                rows.append(
                    json.loads(line)
                )

    return rows


def dump_obj(o):

    try:
        return o.model_dump(
            mode="json"
        )
    except Exception:

        try:
            return o.model_dump()
        except Exception:
            return {"repr": repr(o)}


def call_openai(model,prompt,mx):

    from openai import OpenAI

    c = OpenAI(
        api_key=os.environ[
            "OPENAI_API_KEY"
        ]
    )

    r = c.responses.create(
        model=model,
        input=prompt,
        reasoning={
            "effort":"none"
        },
        max_output_tokens=mx,
        store=False
    )

    return (
        r.output_text or "",
        dump_obj(r)
    )


def call_anthropic(model,prompt,mx):

    from anthropic import Anthropic

    c = Anthropic(
        api_key=os.environ[
            "ANTHROPIC_API_KEY"
        ]
    )

    r = c.messages.create(
        model=model,
        max_tokens=mx,
        messages=[
            {
                "role":"user",
                "content":prompt
            }
        ]
    )

    text = "\n".join(
        getattr(x,"text","")
        for x in r.content
        if getattr(x,"text",None)
    )

    return (
        text,
        dump_obj(r)
    )


def call_google(model,prompt,mx):

    from openai import OpenAI

    c = OpenAI(
        api_key=os.environ[
            "GEMINI_API_KEY"
        ],
        base_url=(
            "https://generativelanguage.googleapis.com/"
            "v1beta/openai/"
        )
    )

    r = c.chat.completions.create(
        model=model,
        messages=[
            {
                "role":"user",
                "content":prompt
            }
        ],
        max_tokens=mx,
        extra_body={
            "google":{
                "thinking_config":{
                    "thinking_level":"low",
                    "include_thoughts":False
                }
            }
        }
    )

    return (
        r.choices[0].message.content
        or "",
        dump_obj(r)
    )


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--dry-run",
        action="store_true"
    )

    ap.add_argument(
        "--provider",
        default="ALL",
        choices=[
            "ALL",
            "OPENAI",
            "ANTHROPIC",
            "GOOGLE"
        ]
    )

    args = ap.parse_args()

    plan = load_plan()
    prompts = prompt_map()

    if args.provider != "ALL":

        plan = [
            x for x in plan
            if x["provider"]
            == args.provider
        ]


    print(
        "ACTIVE CAUSAL CELLS:",
        len(plan)
    )


    if args.dry_run:

        print(
            "API CALLS: 0"
        )

        return


    RAW.mkdir(
        parents=True,
        exist_ok=True
    )


    envkeys = {
        "OPENAI":
            "OPENAI_API_KEY",

        "ANTHROPIC":
            "ANTHROPIC_API_KEY",

        "GOOGLE":
            "GEMINI_API_KEY"
    }


    for i,cell in enumerate(
        plan,
        start=1
    ):

        dest = (
            RAW
            /
            f"{cell['cell_id']}.json"
        )


        if dest.exists():

            prev = json.loads(
                dest.read_text(
                    encoding="utf-8"
                )
            )

            if prev.get(
                "status"
            ) == "COMPLETE":

                print(
                    i,
                    len(plan),
                    "SKIP_COMPLETE",
                    cell["cell_id"]
                )

                continue


        key = envkeys[
            cell["provider"]
        ]


        if not os.environ.get(key):

            raise RuntimeError(
                f"Missing {key}"
            )


        source = prompts[
            (
                cell["seed"],
                cell["condition"]
            )
        ]


        prompt = source[
            "PROMPT"
        ]


        result_text = None
        metadata = None
        error = None


        for attempt in range(4):

            try:

                if cell["provider"] == "OPENAI":

                    result_text,metadata = (
                        call_openai(
                            cell["model"],
                            prompt,
                            cell[
                                "max_output_tokens"
                            ]
                        )
                    )

                elif cell["provider"] == "ANTHROPIC":

                    result_text,metadata = (
                        call_anthropic(
                            cell["model"],
                            prompt,
                            cell[
                                "max_output_tokens"
                            ]
                        )
                    )

                else:

                    result_text,metadata = (
                        call_google(
                            cell["model"],
                            prompt,
                            cell[
                                "max_output_tokens"
                            ]
                        )
                    )


                error = None
                break

            except Exception:

                error = traceback.format_exc()

                if attempt < 3:

                    time.sleep(
                        RETRY_WAIT[
                            attempt
                        ]
                    )


        status = (
            "COMPLETE"
            if (
                error is None
                and
                result_text
            )
            else
            "TECHNICAL_FAILURE"
        )


        dest.write_text(
            json.dumps(
                {
                    **cell,

                    "prompt_sha256":
                        source[
                            "PROMPT_SHA256"
                        ],

                    "status":
                        status,

                    "text":
                        result_text,

                    "provider_metadata":
                        metadata,

                    "error":
                        error
                },
                indent=2,
                ensure_ascii=False
            ),
            encoding="utf-8"
        )


        print(
            i,
            len(plan),
            status,
            cell["provider"],
            cell["seed"],
            cell["condition"],
            flush=True
        )


if __name__ == "__main__":
    main()
