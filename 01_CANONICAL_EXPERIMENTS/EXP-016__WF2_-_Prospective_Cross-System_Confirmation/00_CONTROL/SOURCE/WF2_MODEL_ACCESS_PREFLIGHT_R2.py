import json
import os
import sys

results = {}

def details(e):
    d = {
        "type": type(e).__name__,
        "message": str(e),
    }
    body = getattr(e, "body", None)
    if body:
        d["body"] = body
    return d

# OPENAI
try:
    from openai import OpenAI

    c = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    r = c.responses.create(
        model="gpt-5.6-terra",
        input="Reply with exactly OK.",
        reasoning={"effort": "none"},
        max_output_tokens=32,
        store=False,
    )

    results["OPENAI"] = {
        "status": "PASS",
        "model": "gpt-5.6-terra",
        "received_text": bool(r.output_text),
    }

except Exception as e:
    results["OPENAI"] = {
        "status": "FAIL",
        **details(e)
    }


# ANTHROPIC
try:
    from anthropic import Anthropic

    c = Anthropic(
        api_key=os.environ["ANTHROPIC_API_KEY"]
    )

    r = c.messages.create(
        model="claude-sonnet-5",
        max_tokens=32,
        thinking={"type": "disabled"},
        messages=[
            {
                "role": "user",
                "content": "Reply with exactly OK."
            }
        ],
    )

    text = "\n".join(
        getattr(x, "text", "")
        for x in r.content
        if getattr(x, "text", None)
    )

    results["ANTHROPIC"] = {
        "status": "PASS",
        "model": "claude-sonnet-5",
        "received_text": bool(text),
    }

except Exception as e:
    results["ANTHROPIC"] = {
        "status": "FAIL",
        **details(e)
    }


# GOOGLE
try:
    from openai import OpenAI

    c = OpenAI(
        api_key=os.environ["GEMINI_API_KEY"],
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    r = c.chat.completions.create(
        model="gemini-3.8-flash",
        reasoning_effort="low",
        messages=[
            {
                "role": "user",
                "content": "Reply with exactly OK."
            }
        ],
        max_tokens=32,
    )

    results["GOOGLE"] = {
        "status": "PASS",
        "model": "gemini-3.8-flash",
        "received_text": bool(r.choices[0].message.content),
    }

except Exception as e:
    results["GOOGLE"] = {
        "status": "FAIL",
        **details(e)
    }


print(json.dumps(results, indent=2))

ok = all(
    x.get("status") == "PASS"
    and x.get("received_text")
    for x in results.values()
)

if ok:
    print("")
    print("ALL MODEL-SPECIFIC PREFLIGHTS: PASS")
    print("SCIENTIFIC PROMPTS USED: NO")
else:
    sys.exit(2)
