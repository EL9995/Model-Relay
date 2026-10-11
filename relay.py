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
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

OLLAMA = "http://localhost:11434"
RUNS = Path(__file__).with_name("runs")  # every run is saved here as one JSON file
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
        entry["layer"] = layer
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
                 "cut": out["cut"], "layer": i}
        trace.append(entry)
        on_event({"type": "step", "layer": i, **entry})
    return trace


def run(task, chain, mode="relay", rule="", on_event=lambda e: None, save=True):
    """Run one relay or blind pass and return its record, saved to runs/ unless save=False."""
    start = time.time()
    if mode == "blind":
        answer, trace = None, blind(task, chain, rule, on_event)
    else:
        answer, trace = relay(task, chain, rule, on_event)
    record = {"mode": mode, "task": task, "rule": rule, "chain": chain, "answer": answer,
              "secs": round(time.time() - start, 1), "calls": len(trace),
              "peak": max(([m for m in snap if m in chain] for snap in snapshots), key=len, default=[]),
              "trace": trace}
    return save_run(record) if save else record


def save_run(record):
    RUNS.mkdir(exist_ok=True)
    stamp = time.strftime("%Y-%m-%d_%H%M%S")
    slug = re.sub(r"[^a-z0-9]+", "-", record["task"].lower()).strip("-")[:40].strip("-") or "run"
    rid, n = f"{stamp}_{slug}", 2
    while (RUNS / f"{rid}.json").exists():
        rid, n = f"{stamp}_{slug}-{n}", n + 1
    record = {"id": rid, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), **record}
    (RUNS / f"{rid}.json").write_text(json.dumps(record, indent=2, ensure_ascii=False))
    return record


def load_run(rid):
    """One saved run by id, or None. Ids are file names, so only plain ones are allowed."""
    path = RUNS / f"{rid}.json"
    if not re.fullmatch(r"[\w-]+", rid) or not path.is_file():
        return None
    return json.loads(path.read_text())


def list_runs():
    """Saved runs, newest first, without their traces."""
    runs = []
    for path in sorted(RUNS.glob("*.json"), reverse=True):
        try:
            r = json.loads(path.read_text())
        except ValueError:
            continue
        runs.append({k: r.get(k) for k in ("id", "time", "mode", "task", "rule", "chain", "answer",
                                           "secs", "calls")})
    return runs


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("task")
    p.add_argument("--chain", default=DEFAULT_CHAIN, help="models, biggest first, comma-separated")
    p.add_argument("--json", action="store_true", help="print the full trace as JSON")
    p.add_argument("--blind", action="store_true",
                   help="every model answers the original task on its own; no relay")
    p.add_argument("--rule", default="", help="a rule every model must follow (system prompt)")
    p.add_argument("--no-save", action="store_true", help="don't save this run to runs/")
    a = p.parse_args()
    chain = [m.strip() for m in a.chain.split(",") if m.strip()]

    r = run(a.task, chain, "blind" if a.blind else "relay", a.rule, save=not a.no_save)
    if a.json:
        print(json.dumps(r, indent=2, ensure_ascii=False))
        return

    arrows = {"down": "↓", "answer": "●", "up": "↑", "solo": "●"}
    print(f"TASK  {a.task}\n")
    for t in r["trace"]:
        flag = "  (cut off)" if t.get("cut") else ""
        if t["dir"] == "up":
            flag = "  (corrected)" if t["changed"] else "  (ok)"
        print(f"{arrows[t['dir']]} {t['model']:<16} {t['secs']:>5}s {t['tok_s']:>4} tok/s{flag}")
        print(f"    {t['sent']}")
    print(f"\nPEAK    {len(r['peak'])}/{len(chain)} models loaded at once: {', '.join(r['peak'])}")
    if r["answer"] is not None:
        print(f"ANSWER  {r['answer']}")
    print(f"TIME    {r['secs']}s for {r['calls']} calls")
    if not a.blind:
        in_chain = [(n, round(sz / 1e9, 1)) for n, sz, _ in loaded() if n in chain]
        print(f"MEMORY  {sum(g for _, g in in_chain):.1f} GB across {len(in_chain)}/{len(chain)} "
              f"models still loaded: " + ", ".join(f"{n} {g}GB" for n, g in in_chain))
    if "id" in r:
        print(f"SAVED   runs/{r['id']}.json")


if __name__ == "__main__":
    sys.exit(main())
