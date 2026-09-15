import json
import os
import sys
import traceback

RESULTS = {}

def err_detail(e):
    d = {
        "type": type(e).__name__,
        "message": str(e),
    }

    body = getattr(e, "body", None)
    if body is not None:
        d["body"] = body

    response = getattr(e, "response", None)
    if response is not None:
        try:
            d["status_code"] = response.status_code
        except Exception:
            pass
        try:
            d["response_text"] = response.text
        except Exception:
            pass

    return d


# ------------------------------------------------------------
# OPENAI
# ------------------------------------------------------------

try:
    from openai import OpenAI

    c = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    r = c.responses.create(
        model="gpt-5.6-terra",
        input="Reply with exactly: OK",
        reasoning={"effort": "none"},
        max_output_tokens=16,
        store=False,
    )

    RESULTS["OPENAI"] = {
        "status": "PASS",
        "model": "gpt-5.6-terra",
        "text_received": bool(r.output_text),
    }

except Exception as e:
    RESULTS["OPENAI"] = {
        "status": "FAIL",
        **err_detail(e),
    }


# ------------------------------------------------------------
# ANTHROPIC
# Sonnet 5 thinking explicitly disabled for the corrected
# no-thinking/minimal-comparability configuration.
# ------------------------------------------------------------

try:
    from anthropic import Anthropic

    c = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    r = c.messages.create(
        model="claude-sonnet-5",
        max_tokens=16,
        thinking={"type": "disabled"},
        messages=[
            {
                "role": "user",
                "content": "Reply with exactly: OK",
            }
        ],
    )

    text = "\n".join(
        getattr(x, "text", "")
        for x in r.content
        if getattr(x, "text", None)
    )

    RESULTS["ANTHROPIC"] = {
        "status": "PASS",
        "model": "claude-sonnet-5",
        "text_received": bool(text),
    }

except Exception as e:
    RESULTS["ANTHROPIC"] = {
        "status": "FAIL",
        **err_detail(e),
    }


# ------------------------------------------------------------
# GOOGLE
# Use the documented OpenAI-compatible reasoning_effort field.
# ------------------------------------------------------------

try:
    from openai import OpenAI

    c = OpenAI(
        api_key=os.environ["GEMINI_API_KEY"],
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    r = c.chat.completions.create(
        model="gemini-3.8-flash",
        reasoning_effort="low",
        messages=[
            {
                "role": "user",
                "content": "Reply with exactly: OK",
            }
        ],
        max_tokens=16,
    )

    RESULTS["GOOGLE"] = {
        "status": "PASS",
        "model": "gemini-3.8-flash",
        "text_received": bool(r.choices[0].message.content),
    }

except Exception as e:
    RESULTS["GOOGLE"] = {
        "status": "FAIL",
        **err_detail(e),
    }


print(json.dumps(RESULTS, indent=2, default=str))

if not all(x.get("status") == "PASS" for x in RESULTS.values()):
    sys.exit(2)

print("")
print("ALL THREE PROVIDER PREFLIGHTS: PASS")
print("SCIENTIFIC PROMPTS USED: NO")
