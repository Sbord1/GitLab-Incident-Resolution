"""
Measures the triage node (first pass only: retrieval + 1 call).
- scope_set.json: expected OUT_OF_SCOPE or INSUFFICIENT
- evaluation_set.json: expected OK (no valid question should be blocked)

Usage:
    python evaluate_triage.py
"""

import hashlib
import json
from pathlib import Path

from src.agent import (TRIAGE_EFFORT, TRIAGE_MODEL, TRIAGE_SNIPPET_CHARS,
                   TRIAGE_SYSTEM, call_triage)
from src.hybrid_query import search
from src.rag import RETRIEVAL_MODE, TOP_K, collect_sources

HERE = Path(__file__).resolve().parent
EVAL_PATH = HERE / "datasets/evaluation_set.json"
SCOPE_PATH = HERE / "datasets/scope_set.json"
CACHE_PATH = HERE / "../data/processed/triage_cache.json"
OUTPUT_PATH = HERE / "../data/processed/triage_results.json"


def cache_key(question, sources):
    parts = [TRIAGE_MODEL, TRIAGE_EFFORT, str(TRIAGE_SNIPPET_CHARS),
             TRIAGE_SYSTEM, question] + [s["doc_id"] for s in sources]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def pct(a, b):
    return f"{a}/{b}" + (f" ({a / b:.0%})" if b else "")


def main():
    scope = json.loads(SCOPE_PATH.read_text(encoding="utf-8"))
    evals = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    cache = (json.loads(CACHE_PATH.read_text(encoding="utf-8"))
             if CACHE_PATH.exists() else {})

    rows = [("scope", i["scope_id"], i["question"], i["expected_verdict"],
             i["category"]) for i in scope]
    rows += [("eval", i["eval_id"], i["question"], "OK", "covered")
             for i in evals]

    print(f"Triage model: {TRIAGE_MODEL} (effort: {TRIAGE_EFFORT or 'n/a'})\n")
    results = []
    for kind, qid, question, expected, category in rows:
        sources = collect_sources(
            search(question, top_k=TOP_K, mode=RETRIEVAL_MODE))
        key = cache_key(question, sources)
        if key not in cache:
            cache[key] = call_triage(question, sources)
        r = cache[key]

        flag = "ok  " if r["verdict"] == expected else "DIFF"
        note = "" if r["parsed"] else "  (unparseable output -> OK)"
        print(f"{flag} [{qid}] expected={expected:<12} "
              f"got={r['verdict']:<12}{note}")
        if r["verdict"] == "INSUFFICIENT":
            print(f"       rephrased query: {r['query']}")
        results.append({"kind": kind, "id": qid, "question": question,
                        "category": category, "expected": expected, **r})

    CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")

    ev = [r for r in results if r["kind"] == "eval"]
    oos = [r for r in results if r["expected"] == "OUT_OF_SCOPE"]
    ins = [r for r in results if r["expected"] == "INSUFFICIENT"]
    toks = [(r["input_tokens"] or 0, r["output_tokens"] or 0)
            for r in results]

    print("\n" + "=" * 66)
    print("VALID QUESTIONS (evaluation set) - should not be blocked")
    print(f"  incorrect rejections (OUT_OF_SCOPE): "
          f"{pct(sum(r['verdict'] == 'OUT_OF_SCOPE' for r in ev), len(ev))}")
    print(f"  INSUFFICIENT on first pass:           "
          f"{pct(sum(r['verdict'] == 'INSUFFICIENT' for r in ev), len(ev))}")
    print("OUT OF SCOPE - expected OUT_OF_SCOPE")
    print(f"  correct:               {pct(sum(r['verdict'] == 'OUT_OF_SCOPE' for r in oos), len(oos))}")
    print(f"  blocked (not OK):      {pct(sum(r['verdict'] != 'OK' for r in oos), len(oos))}")
    print("IN SCOPE BUT NOT COVERED - expected INSUFFICIENT")
    print(f"  correct:               {pct(sum(r['verdict'] == 'INSUFFICIENT' for r in ins), len(ins))}")
    print(f"  blocked (not OK):      {pct(sum(r['verdict'] != 'OK' for r in ins), len(ins))}")
    print(f"unparseable outputs: {sum(not r['parsed'] for r in results)}/{len(results)}")
    if toks:
        print(f"average tokens per call: input {sum(t[0] for t in toks) / len(toks):.0f}, "
              f"output {sum(t[1] for t in toks) / len(toks):.0f}")

    OUTPUT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(f"\nDetails saved to {OUTPUT_PATH}.")


if __name__ == "__main__":
    main()