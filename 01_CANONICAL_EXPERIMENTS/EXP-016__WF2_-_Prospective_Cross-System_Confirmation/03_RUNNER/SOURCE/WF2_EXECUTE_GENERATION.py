from pathlib import Path
import argparse, json, os, time, traceback

ROOT = Path(r"C:\RSOS\RSOS_EXPERIMENTS_WF2_PROSPECTIVE_CROSS_SYSTEM_CONFIRMATION")
PLANROOT = ROOT / "02_INPUTS" / "01_EXECUTION_PLAN"
RAW = ROOT / "04_RAW"
CELLS = RAW / "CELLS"
LOG = RAW / "WF2_EXECUTION_LOG.jsonl"

SYSTEM = (
    "You are participating in a controlled text continuation experiment. "
    "Use only the supplied conversation excerpt and current user request. "
    "Do not browse, use tools, retrieve external information, or mention "
    "the experiment. Respond naturally and directly."
)

RETRIES = [5, 15, 45]

def load_jsonl(p):
    out=[]
    with p.open("r",encoding="utf-8") as f:
        for line in f:
            if line.strip(): out.append(json.loads(line))
    return out

def dump_obj(o):
    try:
        return o.model_dump(mode="json")
    except Exception:
        try: return o.model_dump()
        except Exception: return {"repr":repr(o)}

def call_openai(model, history, mx):
    from openai import OpenAI
    c=OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    r=c.responses.create(
        model=model,
        instructions=SYSTEM,
        input=history,
        reasoning={"effort":"none"},
        max_output_tokens=mx,
        store=False
    )
    return (r.output_text or "", dump_obj(r))

def call_anthropic(model, history, mx):
    from anthropic import Anthropic
    c=Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    r=c.messages.create(
        model=model,
        system=SYSTEM,
        max_tokens=mx,
        messages=history
    )
    text="\n".join(
        getattr(x,"text","") for x in r.content
        if getattr(x,"text",None)
    )
    return (text,dump_obj(r))

def call_google(model, history, mx):
    from openai import OpenAI
    c=OpenAI(
        api_key=os.environ["GEMINI_API_KEY"],
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    r=c.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":SYSTEM}] + history,
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
    return (r.choices[0].message.content or "",dump_obj(r))

def provider_call(provider,model,history,mx):
    if provider=="OPENAI": return call_openai(model,history,mx)
    if provider=="ANTHROPIC": return call_anthropic(model,history,mx)
    if provider=="GOOGLE": return call_google(model,history,mx)
    raise RuntimeError(provider)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dry-run",action="store_true")
    ap.add_argument("--provider",default="ALL",
                    choices=["ALL","OPENAI","ANTHROPIC","GOOGLE"])
    args=ap.parse_args()

    plan=load_jsonl(PLANROOT/"WF2_EXECUTION_PLAN.jsonl")
    contexts={x["blind_context_id"]:x for x in load_jsonl(
        PLANROOT/"WF2_EXECUTION_CONTEXTS_BLINDED.jsonl"
    )}
    probes=json.loads((PLANROOT/"WF2_PROBE_SETS.json").read_text(encoding="utf-8"))

    if args.provider!="ALL":
        plan=[x for x in plan if x["provider"]==args.provider]

    keys={
        "OPENAI":"OPENAI_API_KEY",
        "ANTHROPIC":"ANTHROPIC_API_KEY",
        "GOOGLE":"GEMINI_API_KEY"
    }

    print("Planned cells:",len(plan))
    print("LABEL KEY LOADED: NO")

    if args.dry_run:
        for p in sorted({x["provider"] for x in plan}):
            print(
                p,
                keys[p],
                "PRESENT" if os.environ.get(keys[p]) else "MISSING"
            )
        print("DRY RUN COMPLETE — API CALLS: 0")
        return

    CELLS.mkdir(parents=True,exist_ok=True)

    for i,cell in enumerate(plan,1):
        dest=CELLS/f"{cell['cell_id']}.json"

        if dest.exists():
            prev=json.loads(dest.read_text(encoding="utf-8"))
            if prev.get("status")=="COMPLETE":
                print(i,len(plan),"SKIP_COMPLETE",cell["cell_id"])
                continue

        if not os.environ.get(keys[cell["provider"]]):
            raise RuntimeError("Missing "+keys[cell["provider"]])

        ctx=contexts[cell["blind_context_id"]]
        ps=probes[cell["condition"]]

        first=(
            "Use this earlier conversation excerpt as context:\n\n"
            "<earlier_conversation>\n"
            +ctx["context_excerpt"]
            +"\n</earlier_conversation>\n\n"
            +ps[0]
        )

        history=[{"role":"user","content":first}]
        scoring=[{"role":"user","content":ps[0]}]
        responses=[]
        status="COMPLETE"
        failure=None

        for turn in range(6):
            if turn>0:
                history.append({"role":"user","content":ps[turn]})
                scoring.append({"role":"user","content":ps[turn]})

            text=meta=None
            err=None

            for attempt in range(4):
                try:
                    text,meta=provider_call(
                        cell["provider"],
                        cell["model_id"],
                        history,
                        cell["max_output_tokens"]
                    )
                    err=None
                    break
                except Exception:
                    err=traceback.format_exc()
                    if attempt<3: time.sleep(RETRIES[attempt])

            if err is not None:
                status="TECHNICAL_FAILURE"
                failure={"turn":turn+1,"error":err}
                break

            if not text:
                status="CONTENT_INCOMPLETE"
                failure={"turn":turn+1,"reason":"EMPTY_TEXT"}
                break

            history.append({"role":"assistant","content":text})
            scoring.append({"role":"assistant","content":text})
            responses.append({
                "turn":turn+1,
                "text":text,
                "provider_metadata":meta
            })

        result={
            "cell_id":cell["cell_id"],
            "provider":cell["provider"],
            "organization":cell["organization"],
            "model_requested":cell["model_id"],
            "condition":cell["condition"],
            "blind_context_id":cell["blind_context_id"],
            "status":status,
            "failure":failure,
            "provider_responses":responses,
            "scoring_messages":scoring,
            "historical_context_in_scoring":False
        }

        dest.write_text(
            json.dumps(result,indent=2,ensure_ascii=False),
            encoding="utf-8"
        )

        with LOG.open("a",encoding="utf-8") as f:
            f.write(json.dumps({
                "cell_id":cell["cell_id"],
                "provider":cell["provider"],
                "condition":cell["condition"],
                "status":status
            })+"\n")

        print(i,len(plan),status,cell["provider"],cell["condition"])

if __name__=="__main__":
    main()
