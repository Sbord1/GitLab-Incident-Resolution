"""Agent tests: generation retry and triage branches."""

import src.agent as agent

REAL_GENERATE = agent.generate
REAL_TRIAGE = agent.call_triage
REAL_SEARCH = agent.search
Q = "GitLab returns HTTP 500 errors on the personal preferences page"
ZERO = {"input_tokens": 0, "output_tokens": 0}
BAD = {"text": "Fake answer citing a missing source [9].", **ZERO}
GOOD = {"text": "Fake grounded answer [1].", **ZERO}


def tri(verdict, query=""):
    return {"verdict": verdict, "query": query, "parsed": True, **ZERO}


def no_generate(*a, **k):
    raise AssertionError("generate() should not have been called")


def run(fake_generate, fake_triage, fake_search=None):
    agent.generate, agent.call_triage = fake_generate, fake_triage
    if fake_search:
        agent.search = fake_search
    try:
        return agent.run_agent(Q)
    finally:
        agent.generate, agent.call_triage = REAL_GENERATE, REAL_TRIAGE
        agent.search = REAL_SEARCH


def case_1_retry_recovers():
    """Invalid citation on the first attempt, real LLM with feedback on the second."""
    seen = []

    def gen(question, sources, feedback=None):
        seen.append(feedback)
        return BAD if len(seen) == 1 else REAL_GENERATE(
            question, sources, feedback=feedback)

    out = run(gen, lambda q, s: tri("OK"))
    print("\n[1] feedback received:", seen, "| retries:", out["retries"])
    assert len(seen) == 2 and seen[0] is None and seen[1]
    assert out["retries"] == 1 and out["warning"] is None
    assert out["invalid_citations"] == [] and out["outcome"] == "answered"
    print("OK")


def case_2_retry_fails():
    calls = []
    out = run(lambda q, s, feedback=None: calls.append(1) or BAD,
              lambda q, s: tri("OK"))
    print("\n[2] generation calls:", len(calls), "| warning:", out["warning"])
    assert len(calls) == 2 and out["retries"] == 1
    assert out["warning"] and out["invalid_citations"] == [9]
    print("OK")


def case_3_out_of_scope():
    out = run(no_generate, lambda q, s: tri("OUT_OF_SCOPE"))
    print("\n[3] outcome:", out["outcome"])
    assert out["outcome"] == "out_of_scope" and out["cited"] == []
    assert out["retries"] == 0 and out["retrieval_retries"] == 0
    print("OK")


def case_4_rewrite_then_ok():
    verdicts = iter([tri("INSUFFICIENT", "rewritten technical query"),
                     tri("OK")])
    queries = []

    def spy(q, **kw):
        queries.append(q)
        return REAL_SEARCH(q, **kw)

    out = run(lambda q, s, feedback=None: GOOD,
              lambda q, s: next(verdicts), fake_search=spy)
    print("\n[4] queries used:", queries)
    assert queries == [Q, "rewritten technical query"]
    assert out["retrieval_retries"] == 1 and out["outcome"] == "answered"
    print("OK")


def case_5_insufficient_twice():
    calls = []

    def tri_always(q, s):
        calls.append(1)
        return tri("INSUFFICIENT", "rewritten technical query")

    out = run(no_generate, tri_always)
    print("\n[5] triage calls:", len(calls), "| outcome:", out["outcome"])
    assert len(calls) == 2 and out["retrieval_retries"] == 1
    assert out["outcome"] == "insufficient"
    print("OK")


if __name__ == "__main__":
    case_1_retry_recovers      # 1 real LLM call
    case_2_retry_fails()
    case_3_out_of_scope()
    case_4_rewrite_then_ok()
    case_5_insufficient_twice()