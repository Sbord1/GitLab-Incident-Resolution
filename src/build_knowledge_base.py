"""
Builds the v2 knowledge base directly from raw GitLab exports.

Input : ../docs/raw_issues.json     (list of incidents)
        ../docs/raw_runbooks.json   (object with "runbooks" key)

Output: ../docs/knowledge_base_v2.json
"""

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from src.prepare_documents import first_h1, is_meaningful, prepare_document

HERE = Path(__file__).resolve().parent
DEFAULT_ISSUES = HERE / "../data/raw/raw_issues.json"
DEFAULT_RUNBOOKS = HERE / "../data/raw/raw_runbooks.json"
DEFAULT_OUT = HERE / "../data/processed/knowledge_base.json"


# Minimum number of indexable text characters required for a record.
# Records below this threshold are considered empty or template-only.
MIN_INDEXABLE_CHARS = 100


# Cross-document filtering: if a section's normalized text is IDENTICAL
# across at least REPEAT_MIN_DOCS different incidents, it's considered
# template boilerplate rather than incident-specific information.
# Short texts (< REPEAT_MIN_CHARS) are excluded because they could contain
# actual answers (e.g. "Saturation").
# This rule applies only to incidents: runbooks don't share templates.
REPEAT_MIN_DOCS = 30
REPEAT_MIN_CHARS = 40


# LINE-level filtering: a normalized line (outside code blocks) that is
# identical across at least LINE_MIN_DOCS incidents and is at least
# LINE_MIN_CHARS characters long is considered template text,
# even inside partially completed sections.
LINE_MIN_DOCS = 100
LINE_MIN_CHARS = 40

SCHEMA_VERSION = 3


def slugify(path):
    p = re.sub(r"^docs/", "", path)
    p = re.sub(r"\.md$", "", p, flags=re.IGNORECASE)
    return re.sub(r"[^a-z0-9]+", "_", p.lower()).strip("_")


def runbook_doc_ids(runbooks):
    """Generate stable doc_ids from canonical paths. If two paths produce
    the same slug, ALL members of the group receive a hash suffix based on the path."""
    slugs = {r["path"]: slugify(r["path"]) for r in runbooks}
    counts = Counter(slugs.values())
    ids = {}
    for path, slug in slugs.items():
        if counts[slug] > 1:
            slug = f"{slug}_{hashlib.sha1(path.encode()).hexdigest()[:8]}"
        ids[path] = f"runbook_{slug}"
    return ids


def _indexable_chars(prepared):
    return sum(len(s["text"]) for s in prepared["sections"]
               if s["origin"] == "section")


def build_incident(issue):
    """Returns (document | None, skip_reason | None, prepared)."""
    analysis = issue.get("analysis") or {}
    description = issue.get("description") or ""
    prepared = prepare_document(description, kind="incident")

    if analysis.get("reason") == "test_artifact":
        return None, "test_artifact", prepared
    if not description.strip():
        return None, "empty_description", prepared

    return {
        "doc_id": f"incident_{issue['iid']}",
        "type": "incident",
        "title": issue.get("title") or f"Incident {issue['iid']}",
        "url": issue.get("url"),
        "raw_text": description,
        "sections": prepared["sections"],
        "metadata": {
            "iid": issue["iid"],
            "labels": issue.get("labels", []),
            "created_at": issue.get("created_at"),
            "updated_at": issue.get("updated_at"),
            "closed_at": issue.get("closed_at"),
            "analysis": analysis,
        },
    }, None, prepared


def build_runbook(rb, doc_id):
    content = rb.get("content") or ""
    prepared = prepare_document(content, kind="runbook")

    if not content.strip():
        return None, "empty_content", prepared

    title = (first_h1(content)
             or Path(rb["path"]).stem.replace("-", " ").replace("_", " "))
    return {
        "doc_id": doc_id,
        "type": "runbook",
        "title": title,
        "url": rb.get("blob_url") or rb.get("url"),
        "raw_text": content,
        "sections": prepared["sections"],
        "metadata": {
            "path": rb["path"],
            "service": rb.get("service"),
            "filename": rb.get("filename"),
            "doc_type": rb.get("doc_type"),
            "alert_name": rb.get("alert_name"),
            "blob_url": rb.get("blob_url"),
            "published_url": rb.get("published_url"),
            "raw_url": rb.get("url"),
        },
    }, None, prepared


def repeat_key(text):
    """Comparison key: normalize whitespace and ignore checkbox state
    ([x] and [ ] are treated as the same template)."""
    t = re.sub(r"\[[xX]\]", "[ ]", text)
    return re.sub(r"\s+", " ", t).strip()


def drop_repeated_sections(docs, min_docs=REPEAT_MIN_DOCS,
                           min_chars=REPEAT_MIN_CHARS):
    """Remove identical text from incident sections (in place) when it appears
    in at least min_docs documents. Returns verifiable statistics."""
    holders, sample = defaultdict(set), {}
    for d in docs:
        if d["type"] != "incident":
            continue
        for s in d["sections"]:
            k = repeat_key(s["text"])
            if len(k) >= min_chars:
                holders[k].add(d["doc_id"])
                sample.setdefault(k, s["section_path"].split(" > ")[-1])
    repeated = {k for k, v in holders.items() if len(v) >= min_docs}

    n = chars = 0
    for d in docs:
        if d["type"] != "incident":
            continue
        kept = []
        for s in d["sections"]:
            if repeat_key(s["text"]) in repeated:
                n += 1
                chars += len(s["text"])
            else:
                kept.append(s)
        d["sections"] = kept

    top = sorted(repeated, key=lambda k: -len(holders[k]) * len(k))[:50]
    return {
        "sections_dropped": n,
        "chars_dropped": chars,
        "distinct_texts": len(repeated),
        "top_texts": [{"docs": len(holders[k]), "chars": len(k),
                       "heading": sample[k], "text": k[:160]} for k in top],
    }


def _iter_lines(text):
    """Returns (line, in_code). Lines inside ``` / ~~~ blocks (including
    the fences themselves) are marked as code and are never modified."""
    in_code = False
    for line in text.split("\n"):
        if line.strip().startswith(("```", "~~~")):
            in_code = not in_code
            yield line, True
            continue
        yield line, in_code


def drop_repeated_lines(docs, min_docs=LINE_MIN_DOCS, min_chars=LINE_MIN_CHARS):
    """Remove template lines from incident sections (in place).
    Sections left without any meaningful content are also removed."""
    holders, sample = defaultdict(set), {}
    for d in docs:
        if d["type"] != "incident":
            continue
        for s in d["sections"]:
            for line, in_code in _iter_lines(s["text"]):
                if in_code:
                    continue
                k = repeat_key(line)
                if len(k) >= min_chars:
                    holders[k].add(d["doc_id"])
                    sample.setdefault(k, line.strip())
    repeated = {k for k, v in holders.items() if len(v) >= min_docs}

    lines_removed = chars = emptied = 0
    for d in docs:
        if d["type"] != "incident":
            continue
        new_sections = []
        for s in d["sections"]:
            kept, changed, has_code = [], False, False
            for line, in_code in _iter_lines(s["text"]):
                if in_code:
                    has_code = True
                    kept.append(line)
                elif repeat_key(line) in repeated:
                    lines_removed += 1
                    chars += len(line)
                    changed = True
                else:
                    kept.append(line)
            if not changed:
                new_sections.append(s)
                continue
            text = re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()
            if not has_code and not is_meaningful(text):
                emptied += 1
                continue
            new_sections.append({**s, "text": text})
        d["sections"] = new_sections

    top = sorted(repeated, key=lambda k: -len(holders[k]) * len(k))[:50]
    return {
        "lines_removed": lines_removed,
        "chars_removed": chars,
        "distinct_lines": len(repeated),
        "sections_emptied": emptied,
        "top_lines": [{"docs": len(holders[k]), "chars": len(k),
                       "text": sample[k][:160]} for k in top],
    }


def build_knowledge_base(issues, runbooks):
    """Pure, testable function: raw lists -> deterministic KB dictionary."""
    docs, skipped = [], Counter()
    discarded, discarded_chars, notes = Counter(), Counter(), Counter()
    chars_in = 0

    def absorb(prepared):
        nonlocal chars_in
        discarded.update(prepared["discarded"])
        discarded_chars.update(prepared["discarded_chars"])
        notes.update(prepared["kept_notes"])
        chars_in += prepared["chars_in"]

    for issue in sorted(issues, key=lambda i: i["iid"]):
        doc, reason, prepared = build_incident(issue)
        absorb(prepared)
        if doc:
            docs.append(doc)
        else:
            skipped[f"incident:{reason}"] += 1

    runbooks = sorted(runbooks, key=lambda r: r["path"].lower())
    ids = runbook_doc_ids(runbooks)
    for rb in runbooks:
        doc, reason, prepared = build_runbook(rb, ids[rb["path"]])
        absorb(prepared)
        if doc:
            docs.append(doc)
        else:
            skipped[f"runbook:{reason}"] += 1

    repeated = drop_repeated_sections(docs)
    repeated_lines = drop_repeated_lines(docs)
    final_docs, skipped_docs = [], []
    for d in docs:
        n = _indexable_chars({"sections": d["sections"]})
        if n < MIN_INDEXABLE_CHARS:
            skipped[f"{d['type']}:too_little_indexable_text"] += 1
            skipped_docs.append({"doc_id": d["doc_id"], "title": d["title"],
                                 "indexable_chars": n,
                                 "closed_at": d["metadata"].get("closed_at")})
        else:
            final_docs.append(d)
    step = max(1, len(skipped_docs) // 15)
    skipped_examples = skipped_docs[::step][:15]   # deterministic sample
    docs = final_docs
    chars_kept = sum(len(s["text"]) for d in docs for s in d["sections"]
                     if s["origin"] == "section")

    section_types = Counter(
        s["section_type"] for d in docs for s in d["sections"]
    )
    stats = {
        "documents": len(docs),
        "incidents": sum(d["type"] == "incident" for d in docs),
        "runbooks": sum(d["type"] == "runbook" for d in docs),
        "sections": sum(len(d["sections"]) for d in docs),
        "sections_by_type": dict(sorted(section_types.items())),
        "skipped_documents": dict(sorted(skipped.items())),
        "discarded_blocks": dict(sorted(discarded.items())),
        "discarded_chars": dict(sorted(discarded_chars.items())),
        "kept_notes": dict(sorted(notes.items())),
        "repeated_across_docs": repeated,
        "repeated_lines": repeated_lines,
        "line_min_docs": LINE_MIN_DOCS,
        "line_min_chars": LINE_MIN_CHARS,
        "skipped_examples": skipped_examples,
        "repeat_min_docs": REPEAT_MIN_DOCS,
        "repeat_min_chars": REPEAT_MIN_CHARS,
        "chars_in": chars_in,
        "chars_kept_in_sections": chars_kept,
        "min_indexable_chars": MIN_INDEXABLE_CHARS,
    }
    return {"schema_version": SCHEMA_VERSION, "stats": stats, "documents": docs}


def print_report(stats):
    print("\n=== Knowledge base v2 report ===")
    print(f"Documents: {stats['documents']} "
          f"({stats['incidents']} incidents, {stats['runbooks']} runbooks)")
    print(f"Sections kept: {stats['sections']}")
    print("Sections by type:", stats["sections_by_type"])
    print("Skipped documents:", stats["skipped_documents"] or "none")
    print("Discarded blocks (count):", stats["discarded_blocks"] or "none")
    print("Discarded blocks (characters):", stats["discarded_chars"] or "none")
    print("Notes (generic headings kept because they contain content):",
          stats["kept_notes"] or "none")
    rep = stats["repeated_across_docs"]
    print(f"Text repeated in >= {stats['repeat_min_docs']} incidents: "
          f"{rep['distinct_texts']} distinct texts, {rep['sections_dropped']} sections, "
          f"{rep['chars_dropped']} characters removed")
    for t in rep["top_texts"][:12]:
        print(f"   {t['docs']:5d} docs | {t['chars']:4d} chars | {t['heading']!r}: {t['text'][:90]}")
    rl = stats["repeated_lines"]
    print(f"Template lines (>= {stats['line_min_docs']} incidents): "
          f"{rl['distinct_lines']} distinct lines, {rl['lines_removed']} lines, "
          f"{rl['chars_removed']} characters removed, "
          f"{rl['sections_emptied']} sections emptied")
    for t in rl["top_lines"][:12]:
        print(f"   {t['docs']:5d} docs | {t['chars']:4d} chars | {t['text'][:90]}")
    print("Examples of skipped documents (too little useful text after filtering):")
    for e in stats["skipped_examples"]:
        print(f"   {e['doc_id']:<28} {e['indexable_chars']:4d} chars | "
              f"{(e['closed_at'] or '')[:10]} | {e['title'][:70]}")
    cin, ckept = stats["chars_in"], stats["chars_kept_in_sections"]
    if cin:
        print(f"Characters: {cin} input -> {ckept} in sections "
              f"({100 * (cin - ckept) / cin:.1f}% excluded)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--issues", default=str(DEFAULT_ISSUES))
    parser.add_argument("--runbooks", default=str(DEFAULT_RUNBOOKS))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    issues = json.loads(Path(args.issues).read_text(encoding="utf-8"))
    runbooks = json.loads(
        Path(args.runbooks).read_text(encoding="utf-8"))["runbooks"]
    print(f"Loaded {len(issues)} raw incidents and {len(runbooks)} raw runbooks.")

    kb = build_knowledge_base(issues, runbooks)
    Path(args.out).write_text(
        json.dumps(kb, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print_report(kb["stats"])
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()