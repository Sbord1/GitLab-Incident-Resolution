"""
LangGraph agent: retrieve -> triage -> generate -> check.

Triage outcomes: OK, INSUFFICIENT (rephrase and retrieve once),
or OUT_OF_SCOPE. Retry generation once if citations are invalid.
"""

import argparse
import json
import os
import re
import time
from typing import Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from src.hybrid_query import search
from src.rag import (LLM_MODEL, RETRIEVAL_MODE, TOP_K, check_citations,
                     collect_sources, generate, get_client)
from src.telemetry import get_tracer


MAX_RETRIES = 1              # Retry generation
MAX_RETRIEVAL_RETRIES = 1    # Retry retrieval

TRIAGE_MODEL = os.environ.get("TRIAGE_MODEL", LLM_MODEL)
TRIAGE_EFFORT = os.environ.get("TRIAGE_EFFORT", "low")
TRIAGE_SNIPPET_CHARS = 350
VERDICTS = ("OK", "INSUFFICIENT", "OUT_OF_SCOPE")

REFUSAL_TEXT = ("This question is outside the scope of this assistant, which "
                "covers infrastructure incidents of GitLab.com.")
INSUFFICIENT_TEXT = ("The knowledge base does not contain enough relevant "
                     "information to answer this question.")

# Phrases indicating insufficient source information
INSUFFICIENT_RE = re.compile(
    r"not enough|insufficient|do(?:es)? not (?:contain|provide|include|cover)"
    r"|cannot (?:answer|determine)|can't (?:answer|determine)"
    r"|no (?:relevant )?information",
    re.IGNORECASE,
)


TRIAGE_SYSTEM = """You triage questions for an assistant that helps on-call \
engineers operate the infrastructure of GitLab.com. It answers from a \
knowledge base of past incidents and runbooks.
You receive a QUESTION and numbered SOURCES (title and start of text) \
retrieved for it. Reply with one verdict:

- OUT_OF_SCOPE: the question is clearly not about operating GitLab.com's own \
infrastructure or services. Examples: unrelated technology or general \
knowledge questions, non-technical questions, problems in a person's own \
code, pipelines or applications, self-managed installations, product usage \
how-tos.
- INSUFFICIENT: the question is in scope, but none of the sources is \
relevant enough to answer it.
- OK: at least one source is relevant to the question.

Rules:
- A symptom that could be a platform-wide problem (errors, outages, delays \
affecting many users) is in scope.
- Be lenient: if unsure between OUT_OF_SCOPE and another verdict, do not \
choose OUT_OF_SCOPE; if unsure between OK and INSUFFICIENT, choose OK.
- If the verdict is INSUFFICIENT, also write "query": the same problem \
rephrased with the technical terms (components, error types) an incident \
report would use. Do not add facts that are not in the question.

Reply with ONLY a JSON object: {"verdict": "OK|INSUFFICIENT|OUT_OF_SCOPE", \
"query": "<rephrased query or empty>"}"""


class AgentState(TypedDict, total=False):
    question: str
    search_query: Optional[str]
    sources: list
    verdict: str
    triage_parsed: bool
    retry_retrieval: bool
    retrieval_retries: int
    answer: str
    cited: list
    invalid_citations: list
    retries: int
    feedback: Optional[str]
    warning: Optional[str]
    outcome: str
    usage_by_node: dict


def _add(a, b):
    """Sum tokens; return None only if neither count is available."""
    if a is None and b is None:
        return None
    return (a or 0) + (b or 0)


def _track(state, node, res):
    """Copy usage_by_node and add the tokens from `res` to the node."""
    by_node = dict(state.get("usage_by_node") or {})
    prev = by_node.get(node, {"input_tokens": None, "output_tokens": None})
    by_node[node] = {
        "input_tokens": _add(prev["input_tokens"], res["input_tokens"]),
        "output_tokens": _add(prev["output_tokens"], res["output_tokens"]),
    }
    return by_node


def _set_usage(span, res):
    """Token counts only. The span must not carry prompt or document text."""
    if res.get("input_tokens") is not None:
        span.set_attribute("input_tokens", res["input_tokens"])
    if res.get("output_tokens") is not None:
        span.set_attribute("output_tokens", res["output_tokens"])


# --- Triage LLM call (separate for testing) -------------------------------

def call_triage(question, sources):
    blocks = "\n".join(
        f"[{s['n']}] ({s['doc_type']}) {s['title']} "
        f"/ {s.get('section') or 'General'}: "
        f"{s['text'][:TRIAGE_SNIPPET_CHARS]}"
        for s in sources
    )
    kwargs = {}
    if TRIAGE_EFFORT:
        kwargs["reasoning"] = {"effort": TRIAGE_EFFORT}
    response = get_client().responses.create(
        model=TRIAGE_MODEL,
        instructions=TRIAGE_SYSTEM,
        input=f"QUESTION:\n{question}\n\nSOURCES:\n{blocks}",
        max_output_tokens=1000,
        **kwargs,
    )
    usage = getattr(response, "usage", None)

    # Fail-open: default to OK if the output cannot be parsed.
    verdict, query, parsed = "OK", "", False
    match = re.search(r"\{.*\}", response.output_text or "", re.DOTALL)
    try:
        data = json.loads(match.group(0))
        v = str(data.get("verdict", "")).strip().upper()
        if v in VERDICTS:
            verdict, query, parsed = v, str(data.get("query") or ""), True
    except (AttributeError, ValueError, TypeError):
        pass

    return {
        "verdict": verdict,
        "query": query,
        "parsed": parsed,
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
    }


# --- Nodes: receive state and return only updated fields ------------------

def retrieve_node(state: AgentState):
    with get_tracer().start_as_current_span("retrieve") as span:
        query = state.get("search_query") or state["question"]
        started = time.perf_counter()
        results = search(query, top_k=TOP_K, mode=RETRIEVAL_MODE)
        latency_ms = (time.perf_counter() - started) * 1000.0
        sources = collect_sources(results)
        span.set_attribute("retrieval.latency_ms", latency_ms)
        span.set_attribute("retrieval.source_count", len(sources))
        span.set_attribute("retrieval.attempt", state.get("retrieval_retries", 0))
        span.set_attribute("retrieval.mode", RETRIEVAL_MODE)
        return {"sources": sources}


def triage_node(state: AgentState):
    with get_tracer().start_as_current_span("triage") as span:
        t = call_triage(state["question"], state["sources"])
        update = {
            "verdict": t["verdict"],
            "triage_parsed": t["parsed"],
            "usage_by_node": _track(state, "triage", t),
            "retry_retrieval": False,
        }
        rewritten = t["query"].strip()
        if (t["verdict"] == "INSUFFICIENT" and rewritten
                and state.get("retrieval_retries", 0) < MAX_RETRIEVAL_RETRIES):
            update["search_query"] = rewritten
            update["retrieval_retries"] = state.get("retrieval_retries", 0) + 1
            update["retry_retrieval"] = True
        span.set_attribute("triage.verdict", t["verdict"])
        span.set_attribute("triage.parsed", t["parsed"])
        span.set_attribute("triage.retry_retrieval", update["retry_retrieval"])
        _set_usage(span, t)
        return update


def route_after_triage(state: AgentState):
    if state["retry_retrieval"]:
        return "retrieve"
    return {"OK": "generate", "OUT_OF_SCOPE": "refuse"}.get(
        state["verdict"], "insufficient")


def refuse_node(state: AgentState):
    return {"answer": REFUSAL_TEXT, "cited": [], "invalid_citations": [],
            "outcome": "out_of_scope", "warning": None}


def insufficient_node(state: AgentState):
    return {"answer": INSUFFICIENT_TEXT, "cited": [], "invalid_citations": [],
            "outcome": "insufficient", "warning": None}


def generate_node(state: AgentState):
    with get_tracer().start_as_current_span("generate") as span:
        gen = generate(state["question"], state["sources"],
                       feedback=state.get("feedback"))
        span.set_attribute("generate.attempt", state.get("retries", 0))
        span.set_attribute("generate.has_feedback", state.get("feedback") is not None)
        _set_usage(span, gen)
        return {"answer": gen["text"],
                "usage_by_node": _track(state, "generate", gen)}


def check_node(state: AgentState):
    with get_tracer().start_as_current_span("check") as span:
        sources = state["sources"]
        cited, invalid = check_citations(state["answer"], sources)
        retries = state.get("retries", 0)

        problem = None
        if invalid:
            problem = (f"Your previous answer cited non-existent sources {invalid}. "
                       f"Cite only [1]..[{len(sources)}].")
        elif not cited and not INSUFFICIENT_RE.search(state["answer"]):
            problem = ("Your previous answer had no citations. Cite sources "
                       "inline as [n], or state clearly that the sources are "
                       "insufficient.")

        update = {"cited": cited, "invalid_citations": invalid,
                  "feedback": None, "warning": None, "outcome": "answered"}
        if problem and retries < MAX_RETRIES:
            update["feedback"] = problem        # Route back to generate
            update["retries"] = retries + 1
        elif problem:
            update["warning"] = "Citation check failed after retry: " + problem
        span.set_attribute("check.cited_count", len(cited))
        span.set_attribute("check.invalid_count", len(invalid))
        span.set_attribute("check.will_retry", update.get("feedback") is not None)
        span.set_attribute("check.retries", update.get("retries", retries))
        return update


def route_after_check(state: AgentState):
    return "retry" if state.get("feedback") else "done"


# --- Graph -----------------------------------------------------------------

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("triage", triage_node)
    g.add_node("refuse", refuse_node)
    g.add_node("insufficient", insufficient_node)
    g.add_node("generate", generate_node)
    g.add_node("check", check_node)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "triage")
    g.add_conditional_edges("triage", route_after_triage, {
        "retrieve": "retrieve", "generate": "generate",
        "refuse": "refuse", "insufficient": "insufficient"})
    g.add_edge("generate", "check")
    g.add_conditional_edges("check", route_after_check,
                            {"retry": "generate", "done": END})
    g.add_edge("refuse", END)
    g.add_edge("insufficient", END)
    return g.compile()


_graph = None


def run_agent(question: str) -> dict:
    """Same output format as rag.answer(), plus outcome, retries, warning and usage."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    with get_tracer().start_as_current_span("agent.run") as span:
        s = _graph.invoke({"question": question, "retries": 0,
                           "retrieval_retries": 0, "usage_by_node": {}})
        by_node = s.get("usage_by_node", {})
        total_in = total_out = None
        for u in by_node.values():
            total_in = _add(total_in, u["input_tokens"])
            total_out = _add(total_out, u["output_tokens"])
        _set_usage(span, {"input_tokens": total_in, "output_tokens": total_out})
        span.set_attribute("outcome", s["outcome"])
        span.set_attribute("verdict", s.get("verdict") or "")
        span.set_attribute("retries", s.get("retries", 0))
        span.set_attribute("retrieval_retries", s.get("retrieval_retries", 0))
        span.set_attribute("warning", bool(s.get("warning")))
        return {
            "question": question,
            "answer": s["answer"],
            "sources": s["sources"],
            "cited": s["cited"],
            "invalid_citations": s["invalid_citations"],
            "outcome": s["outcome"],
            "verdict": s.get("verdict"),
            "triage_parsed": s.get("triage_parsed"),
            "retries": s["retries"],
            "retrieval_retries": s["retrieval_retries"],
            "warning": s.get("warning"),
            "usage": {"input_tokens": total_in, "output_tokens": total_out,
                      "by_node": by_node},
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="+")
    args = parser.parse_args()
    out = run_agent(" ".join(args.question))

    print(f"\nQuestion: {out['question']}\n" + "=" * 80)
    print(out["answer"])
    print("\n" + "-" * 80 + "\nRetrieved sources (* = cited):")
    for s in out["sources"]:
        mark = "*" if s["n"] in out["cited"] else " "
        print(f"{mark} [{s['n']}] {s['doc_id']} - {s['title']}\n      {s['url']}")
    print(f"\noutcome: {out['outcome']} | triage: {out['verdict']} | "
          f"retry retrieval: {out['retrieval_retries']} | "
          f"retry generation: {out['retries']}")
    print(f"tokens: {out['usage']}")
    if out["warning"]:
        print(f"WARNING: {out['warning']}")


if __name__ == "__main__":
    main()