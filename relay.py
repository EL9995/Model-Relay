#!/usr/bin/env python3
"""Relay a task down a chain of local Ollama models and back up.

The caller (a frontier model, or you) is the top of the chain. Going down,
every layer except the last rewrites the task as a simpler instruction for
the model below it. The last model answers. Coming back up, every layer
checks the answer it got from below and either passes it on or corrects it.

Run:   python3 relay.py "2+2"
       python3 relay.py --chain granite4.2:8b,gemma3:4b,gemma3:1b "2+2"
       python3 relay.py --json "2+2"     (full trace as JSON)
       python3 relay.py --blind "2+2"    (every model answers on its own)
       python3 relay.py --rule "Answer in one word." "Capital of Japan?"
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

OLLAMA = "http://localhost:11434"
DEFAULT_CHAIN = "granite4.2:8b,gemma3:4b,granite4.2:3b,gemma3:1b"
NUM_CTX = 2048      # small context keeps each model's memory down
KEEP_ALIVE = "10m"  # keep every layer loaded so the way back up is fast
MAX_TOKENS = 200    # stops a small model that gets stuck repeating itself
BLIND_TOKENS = 512  # blind answers are the output itself, so give them more room

DOWN_PROMPT = (
    "You are layer {n} of {total} in a relay chain of AI models. Your boss gave you a task. "
    "Do NOT solve it. Rewrite it as one short, clear instruction for your assistant, "
    "a smaller model than you. Keep every detail needed to do it.\n\nTask from your boss: {task}"
)
BOTTOM_PROMPT = (
    "You are the last model in a relay chain. Do this task and give only the result.\n\nTask: {task}"
)
UP_PROMPT = (
    "You are layer {n} of {total} in a relay chain of AI models. You gave your assistant this "
    "instruction:\n{instruction}\n\nYour boss's original task was:\n{task}\n\n"
    "Your assistant answered:\n{answer}\n\n"
    "Check the answer against the original task. If it is right, pass it on unchanged. "
    "If it is wrong or incomplete, give the corrected answer."
)

SCHEMAS = {
    "down": {"type": "object", "properties": {"instruction": {"type": "string"}},
             "required": ["instruction"]},
    "bottom": {"type": "object", "properties": {"answer": {"type": "string"}},
               "required": ["answer"]},
    "up": {"type": "object",
           "properties": {"correct": {"type": "boolean"}, "answer": {"type": "string"},
                          "note": {"type": "string"}},
           "required": ["correct", "answer"]},
}


def post(path, body):
    req = urllib.request.Request(OLLAMA + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def ask(model, prompt, kind, rule="", max_tokens=MAX_TOKENS):
    """One call to one model. The bottom model replies in plain text, because
    forcing JSON on a 1B model sends it into loops; the others reply in JSON.
    A rule, if given, goes in as the system prompt."""
    body = {"model": model, "prompt": prompt, "stream": False,
            "think": False, "keep_alive": KEEP_ALIVE,
            "options": {"num_ctx": NUM_CTX, "temperature": 0, "num_predict": max_tokens}}
    if rule:
        body["system"] = rule
    if kind != "bottom":
        body["format"] = SCHEMAS[kind]
    start = time.time()
    try:
        r = post("/api/generate", body)
    except urllib.error.HTTPError as e:
        if e.code != 400:
            raise
        body.pop("think")  # models without a thinking mode reject the flag
        r = post("/api/generate", body)
    secs = time.time() - start
    try:
        out = json.loads(r["response"]) if kind != "bottom" else {"answer": r["response"].strip()}
    except ValueError:
        out = {"answer": r["response"].strip(), "instruction": r["response"].strip()}
    out["cut"] = r.get("done_reason") == "length"  # hit the token cap mid-answer
    toks = r.get("eval_count", 0)
    tps = toks / (r["eval_duration"] / 1e9) if r.get("eval_duration") else 0
    snapshots.append([n for n, _, _ in loaded()])
    return out, round(secs, 1), round(tps)


snapshots = []  # which models were loaded after each call


def loaded():
    return [(m["name"], m.get("size", 0), m.get("size_vram", 0))
            for m in post_get("/api/ps").get("models", [])]


def post_get(path):
    with urllib.request.urlopen(OLLAMA + path, timeout=10) as r:
        return json.load(r)


def relay(task, chain, rule="", on_event=lambda e: None):
    """on_event gets {"type": "call", ...} before each model call and
    {"type": "step", ...} with the trace entry after it (used by server.py)."""
    snapshots.clear()
    total = len(chain)
    trace, instructions = [], [task]

    def call(layer, dir, model, prompt, kind):
        on_event({"type": "call", "dir": dir, "layer": layer, "model": model})
        return ask(model, prompt, kind, rule)

    def step(layer, entry):
        trace.append(entry)
        on_event({"type": "step", "layer": layer, **entry})

    # Down: each layer simplifies the instruction it was given.
    for i, model in enumerate(chain[:-1]):
        out, secs, tps = call(i, "down", model,
                              DOWN_PROMPT.format(n=i + 1, total=total, task=instructions[-1]), "down")
        instructions.append(out.get("instruction", "").strip() or instructions[-1])
        step(i, {"dir": "down", "model": model, "sent": instructions[-1], "secs": secs, "tok_s": tps})

    # Bottom: the smallest model does the work.
    out, secs, tps = call(total - 1, "answer", chain[-1],
                          BOTTOM_PROMPT.format(task=instructions[-1]), "bottom")
    answer = str(out.get("answer", "")).strip()
    step(total - 1, {"dir": "answer", "model": chain[-1], "sent": answer, "secs": secs, "tok_s": tps,
                     "cut": out["cut"]})

    # Up: each layer checks what came back against what it asked for.
    for i in range(total - 2, -1, -1):
        out, secs, tps = call(i, "up", chain[i], UP_PROMPT.format(
            n=i + 1, total=total, instruction=instructions[i + 1],
            task=instructions[i], answer=answer), "up")
        new = str(out.get("answer", "")).strip() or answer
        step(i, {"dir": "up", "model": chain[i], "sent": new, "secs": secs, "tok_s": tps,
                 "correct": out.get("correct"), "changed": new != answer,
                 "note": out.get("note", "")})
        answer = new

    return answer, trace


def blind(task, chain, rule="", on_event=lambda e: None):
    """Every model gets the original task word for word and answers on its own.
    No model sees another's answer. One model at a time, so it fits in memory."""
    snapshots.clear()
    trace = []
    for i, model in enumerate(chain):
        on_event({"type": "call", "dir": "solo", "layer": i, "model": model})
        out, secs, tps = ask(model, task, "bottom", rule, BLIND_TOKENS)
        entry = {"dir": "solo", "model": model, "sent": out["answer"], "secs": secs, "tok_s": tps,
                 "cut": out["cut"]}
        trace.append(entry)
        on_event({"type": "step", "layer": i, **entry})
    return trace


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("task")
    p.add_argument("--chain", default=DEFAULT_CHAIN, help="models, biggest first, comma-separated")
    p.add_argument("--json", action="store_true", help="print the full trace as JSON")
    p.add_argument("--blind", action="store_true",
                   help="every model answers the original task on its own; no relay")
    p.add_argument("--rule", default="", help="a rule every model must follow (system prompt)")
    a = p.parse_args()
    chain = [m.strip() for m in a.chain.split(",") if m.strip()]

    start = time.time()
    if a.blind:
        start = time.time()
        trace = blind(a.task, chain, a.rule)
        if a.json:
            print(json.dumps({"task": a.task, "chain": chain, "rule": a.rule,
                              "secs": round(time.time() - start, 1), "trace": trace}, indent=2))
            return
        print(f"TASK  {a.task}\n")
        for t in trace:
            cut = "  (cut off)" if t["cut"] else ""
            print(f"● {t['model']:<16} {t['secs']:>5}s {t['tok_s']:>4} tok/s{cut}\n    {t['sent']}")
        print(f"\nTIME    {round(time.time() - start, 1)}s for {len(trace)} calls")
        return
    answer, trace = relay(a.task, chain, a.rule)
    mem = loaded()
    result = {"task": a.task, "chain": chain, "answer": answer, "secs": round(time.time() - start, 1),
              "trace": trace,
              "loaded": [{"model": n, "gb": round(s / 1e9, 1), "gpu_pct": round(100 * v / s) if s else 0}
                         for n, s, v in mem]}
    if a.json:
        print(json.dumps(result, indent=2))
        return

    arrows = {"down": "↓", "answer": "●", "up": "↑"}
    print(f"TASK  {a.task}\n")
    for t in trace:
        flag = ""
        if t["dir"] == "up":
            flag = "  (corrected)" if t["changed"] else "  (ok)"
        print(f"{arrows[t['dir']]} {t['model']:<16} {t['secs']:>5}s {t['tok_s']:>4} tok/s{flag}")
        print(f"    {t['sent']}")
    in_chain = [m for m in result["loaded"] if m["model"] in chain]
    peak = max(snapshots, key=len)
    print(f"PEAK    {len(peak)}/{len(chain)} models loaded at once: {', '.join(peak)}")
    print(f"\nANSWER  {answer}")
    print(f"TIME    {result['secs']}s for {len(trace)} calls")
    print(f"MEMORY  {sum(m['gb'] for m in in_chain):.1f} GB across {len(in_chain)}/{len(chain)} "
          f"models still loaded: " + ", ".join(f"{m['model']} {m['gb']}GB" for m in in_chain))


if __name__ == "__main__":
    sys.exit(main())
