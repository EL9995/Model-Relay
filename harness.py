#!/usr/bin/env python3
"""Score chains of models on a suite of questions with known answers.

Every case in the suite runs against every config: each model on its own (the
baseline), the full relay chain, and optionally the chain with a saboteur at
each layer. Code checks every answer, so no model grades another. Results go
to runs/harness/ and a scorecard is printed at the end.

Run:   python3 harness.py                          (suites/basics.json, default chain)
       python3 harness.py --saboteur               (also a saboteur at each layer)
       python3 harness.py --chain granite4.2:8b,gemma3:1b --only mul,strawberry
       python3 harness.py --show runs/harness/<file>.json   (reprint a scorecard)
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import relay

SUITES = Path(__file__).with_name("suites")
OUT = relay.RUNS / "harness"
NUMBER_WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen twenty".split())}


# ---- checks: each returns (passed, why) ----

def numbers(text):
    """Every number in the text, digits or words up to twenty, in order."""
    found = []
    for m in re.finditer(r"\d[\d,]*(?:\.\d+)?|[a-z]+", text.lower()):
        tok = m.group()
        if tok[0].isdigit():
            found.append(float(tok.replace(",", "")))
        elif tok in NUMBER_WORDS:
            found.append(float(NUMBER_WORDS[tok]))
    return found


def check_number(answer, want, prompt):
    """Right if the answer ends on the expected number ("144 / 12 = 12"), or the first
    number it adds beyond the prompt's is the expected one ("5 cents ($0.05)")."""
    nums = numbers(answer)
    if not nums:
        return False, "no number in the answer"
    given = set(numbers(prompt))
    new = [n for n in nums if n not in given]
    ok = nums[-1] == want or (bool(new) and new[0] == want)
    return ok, f"got {', '.join(f'{n:g}' for n in (new or nums)[:4])}"


def word_count(text):
    return len(re.findall(r"[\w'-]+", text))


def as_list(v):
    return v if isinstance(v, list) else [v]


CHECKS = {
    "number": lambda a, v, c: check_number(a, float(v), c["prompt"]),
    "contains": lambda a, v, c: (any(x.lower() in a.lower() for x in as_list(v)),
                                 f"needs {' or '.join(as_list(v))}"),
    "not_contains": lambda a, v, c: (not any(x.lower() in a.lower() for x in as_list(v)),
                                     f"must not contain {' or '.join(as_list(v))}"),
    "regex": lambda a, v, c: (bool(re.search(v, a)), f"must match /{v}/"),
    "max_words": lambda a, v, c: (word_count(a) <= v, f"{word_count(a)} words, limit {v}"),
    "intact": lambda a, v, c: (c.get("_intact") is v, "message arrived intact" if v else ""),
}


def score(case, answer, record):
    answer = answer.replace("**", "").strip()  # markdown bold isn't part of the answer
    ctx = {**case, "_intact": record.get("intact")}
    fails = []
    for kind, want in case["check"].items():
        ok, why = CHECKS[kind](answer, want, ctx)
        if not ok:
            fails.append(f"{kind}: {why}")
    return not fails, fails


# ---- configs and runs ----

def configs(chain, saboteur, solo=False):
    out = [{"name": f"{m} alone", "chain": [m]} for m in chain]
    if solo:
        return out
    out.append({"name": "relay", "chain": chain})
    if saboteur:
        out += [{"name": f"relay, saboteur L{i + 1}", "chain": chain, "saboteur": i}
                for i in range(len(chain))]
    return out


def run_case(case, cfg):
    mode = case.get("mode")
    if mode == "cipher":
        r = relay.run(case["prompt"], cfg["chain"], "cipher", case.get("rule", ""), save=False,
                      cipher_name=case.get("cipher", "base64"))
    elif len(cfg["chain"]) == 1:
        r = relay.run(case["prompt"], cfg["chain"], "blind", case.get("rule", ""), save=False)
        r["answer"] = r["trace"][0]["sent"]
    else:
        r = relay.run(case["prompt"], cfg["chain"], "relay", case.get("rule", ""), save=False,
                      saboteur=cfg.get("saboteur"))
    return r


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("suite", nargs="?", default="basics", help="suite name in suites/ or a path")
    p.add_argument("--chain", default=relay.DEFAULT_CHAIN, help="models in order, top first")
    p.add_argument("--saboteur", action="store_true", help="also run the chain with a saboteur at each layer")
    p.add_argument("--only", help="comma-separated case ids to run")
    p.add_argument("--solo", action="store_true", help="only test each model on its own, no relay")
    p.add_argument("--show", metavar="FILE", help="print the scorecard of a saved harness run and exit")
    p.add_argument("--rescore", metavar="FILE",
                   help="re-check a saved run's answers with the current checks, without rerunning models")
    a = p.parse_args()

    if a.show:
        report(json.loads(Path(a.show).read_text()))
        return
    if a.rescore:
        rescore(Path(a.rescore))
        return

    path = Path(a.suite) if a.suite.endswith(".json") else SUITES / f"{a.suite}.json"
    suite = json.loads(path.read_text())
    cases = suite["cases"]
    if a.only:
        keep = set(a.only.split(","))
        cases = [c for c in cases if c["id"] in keep]
    chain = [m.strip() for m in a.chain.split(",") if m.strip()]
    cfgs = configs(chain, a.saboteur, a.solo)

    total, n, results = len(cases) * len(cfgs), 0, []
    start = time.time()
    print(f"{suite['name']}: {len(cases)} cases × {len(cfgs)} configs = {total} runs\n")
    for cfg in cfgs:  # one config at a time, so the same models stay loaded
        for case in cases:
            n += 1
            t0 = time.time()
            try:
                r = run_case(case, cfg)
                ok, fails = score(case, r["answer"] or "", r)
                row = {"case": case["id"], "config": cfg["name"], "pass": ok, "fails": fails,
                       "answer": r["answer"], "intact": r.get("intact"), "secs": r["secs"],
                       "tags": case.get("tags", []),
                       "sabotage": (r.get("sabotage") or {}).get("outcome")}
            except Exception as e:  # a crash fails the case; keep going
                row = {"case": case["id"], "config": cfg["name"], "pass": False, "fails": [f"error: {e}"],
                       "answer": None, "secs": round(time.time() - t0, 1), "tags": case.get("tags", []),
                       "sabotage": None}
            results.append(row)
            short = " ".join(str(row["answer"] or "").split())[:50]
            print(f"[{n:>3}/{total}] {cfg['name']:<24} {case['id']:<13} {'✓' if row['pass'] else '✗'} "
                  f"{row['secs']:>5}s  {short}")

    record = {"suite": suite["name"], "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "chain": chain,
              "configs": [c["name"] for c in cfgs], "cases": [c["id"] for c in cases],
              "secs": round(time.time() - start, 1), "results": results}
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{time.strftime('%Y-%m-%d_%H%M%S')}_{suite['name']}.json"
    out.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    report(record)
    print(f"\nSAVED   {out.relative_to(Path(__file__).parent)}")


def rescore(path):
    rec = json.loads(path.read_text())
    cases = {c["id"]: c for c in json.loads((SUITES / f"{rec['suite']}.json").read_text())["cases"]}
    changed = 0
    for row in rec["results"]:
        case = cases.get(row["case"])
        if not case or row["answer"] is None or ("intact" in case["check"] and "intact" not in row):
            continue  # unknown case, a crash, or an old cipher row whose result can't be rechecked
        ok, fails = score(case, row["answer"], {"intact": row.get("intact")})
        changed += ok != row["pass"]
        row["pass"], row["fails"] = ok, fails
    path.write_text(json.dumps(rec, indent=2, ensure_ascii=False))
    report(rec)
    print(f"\nRESCORED {changed} results changed · saved to {path.name}")


def report(rec):
    res = rec["results"]
    by = lambda **kw: [r for r in res if all(r[k] == v for k, v in kw.items())]
    print(f"\nSCORECARD  {rec['suite']} · {len(rec['cases'])} cases · {rec['secs'] / 60:.1f} min\n")
    print(f"{'config':<26} {'correct':>9} {'avg time':>9}  saboteur")
    for name in rec["configs"]:
        rows = by(config=name)
        right = sum(r["pass"] for r in rows)
        sab = [r["sabotage"] for r in rows if r["sabotage"]]
        sab_txt = (f"caught {sab.count('caught')}, slipped {sab.count('slipped')}, "
                   f"didn't lie {sab.count('no_lie')}") if sab else ""
        avg = sum(r["secs"] for r in rows) / max(len(rows), 1)
        print(f"{name:<26} {right:>3}/{len(rows):<3} {right / max(len(rows), 1):>4.0%} {avg:>6.1f}s  {sab_txt}")

    # every case against every config
    cols = rec["configs"]
    print("\n" + " " * 14 + "  ".join(f"{i + 1:>2}" for i in range(len(cols))))
    for cid in rec["cases"]:
        marks = []
        for name in cols:
            r = by(case=cid, config=name)
            marks.append(" ✓" if r and r[0]["pass"] else " ✗")
        print(f"{cid:<14}" + "  ".join(marks))
    print("\n" + "\n".join(f"{i + 1:>2} = {name}" for i, name in enumerate(cols)))


if __name__ == "__main__":
    sys.exit(main())
