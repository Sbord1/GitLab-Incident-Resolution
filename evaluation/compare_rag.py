"""
Compare RAG and baseline (same LLM, no documents) on the evaluation set.
An LLM judge assigns a score of 0/1/2 by comparing the answer with the expected document.

Usage:
    python compare_rag.py --limit 3     # smoke test
    python compare_rag.py               # all questions
"""

import argparse
import json
import re
from pathlib import Path
from dotenv import load_dotenv

from src.rag import LLM_MODEL, get_client
from src.agent import run_agent as rag_answer

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

HERE = Path(__file__).resolve().parent

EVAL_PATH = HERE / "datasets" / "evaluation_set.json"
KB_PATH = HERE.parent / "data" / "processed" / "knowledge_base.json"
OUTPUT_PATH = HERE.parent / "data" / "processed" / "compare_results.json"

BASELINE_SYSTEM = """You are an assistant for on-call engineers of GitLab.com.
No documents are provided: answer only from your own knowledge.
If you do not know, say so clearly instead of guessing.
Do not give numeric confidence scores.

Answer format (skip a section if you have nothing reliable to say):
**Probable cause**
**Diagnostic steps**
**Possible remedies**
Keep it concise."""

JUDGE_SYSTEM = """You grade answers to on-call engineering questions.
You receive a question, a REFERENCE (the correct source document) and an ANSWER.
Score the ANSWER only against the REFERENCE:
2 = identifies the cause or procedure described in the reference
1 = partially correct or only generic advice that is compatible with it
0 = wrong, missing, off-topic, or says it cannot answer
Ignore style and length. Do not reward claims that contradict the reference.
Reply with ONLY a JSON object: {"score": 0|1|2, "reason": "<one sentence>"}"""


def ask(system, user, max_tokens=3000):
    response = get_client().responses.create(
        model=LLM_MODEL,
        instructions=system,
        input=user,
        max_output_tokens=max_tokens,
    )
    return response.output_text or ""


def baseline_answer(question):
    return ask(BASELINE_SYSTEM, question)


def judge(question, reference, answer_text):
    raw = ask(
        JUDGE_SYSTEM,
        f"QUESTION:\n{question}\n\nREFERENCE:\n{reference}\n\nANSWER:\n{answer_text}",
        max_tokens=2000,
    )
    print("JUDGE - raw response:", repr(raw))
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        data = json.loads(match.group(0))
        return int(data["score"]), data.get("reason", "")
    except (AttributeError, ValueError, KeyError, TypeError):
        return None, f"unparseable judgment: {raw[:100]}"


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    eval_set = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    kb_data = json.loads(KB_PATH.read_text(encoding="utf-8"))
    documents = kb_data["documents"] if isinstance(kb_data, dict) else kb_data

    kb = {d["doc_id"]: d for d in documents}
    if args.limit:
        eval_set = eval_set[: args.limit]

    details = []
    for item in eval_set:
        q = item["question"]
        expected = set(item["expected_source_doc_ids"])
        reference = "\n\n".join(
                f"{kb[d]['title']}\n{kb[d]['raw_text']}"
                for d in expected
                if d in kb
            )
        print(f"[{item['eval_id']}] {q[:80]}...")

        rag = rag_answer(q)
        print(f"rag text {rag['answer']}")
        base_text = baseline_answer(q)
        print(f"base text {base_text}")
        rag_score, rag_reason = judge(q, reference, rag["answer"])
        base_score, base_reason = judge(q, reference, base_text)

        retrieved = {s["doc_id"] for s in rag["sources"]}
        cited = {s["doc_id"] for s in rag["sources"] if s["n"] in rag["cited"]}
        print(f"   RAG={rag_score}  baseline={base_score}  "
              f"expected retrieved={bool(expected & retrieved)}")

        details.append({
            "eval_id": item["eval_id"],
            "question": q,
            "expected": sorted(expected),
            "rag": {"answer": rag["answer"], "score": rag_score,
                    "reason": rag_reason,
                    "expected_retrieved": bool(expected & retrieved),
                    "expected_cited": bool(expected & cited),
                    "invalid_citations": rag["invalid_citations"]},
            "baseline": {"answer": base_text, "score": base_score,
                         "reason": base_reason},
        })

    n = len(details)
    print("\n" + "=" * 60)
    print(f"RESULTS ({n} questions, model {LLM_MODEL})")
    for name in ("rag", "baseline"):
        scores = [d[name]["score"] for d in details]
        full = sum(1 for s in scores if s == 2)
        print(f"{name:<9} average score {mean(scores):.2f}/2   "
              f"score 2: {full}/{n}")
    print(f"RAG: expected retrieved {sum(d['rag']['expected_retrieved'] for d in details)}/{n}, "
          f"expected cited {sum(d['rag']['expected_cited'] for d in details)}/{n}, "
          f"invalid citations {sum(bool(d['rag']['invalid_citations']) for d in details)}/{n}")

    OUTPUT_PATH.write_text(
        json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResults saved to {OUTPUT_PATH}.")


if __name__ == "__main__":
    main()