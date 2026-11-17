"""
Deterministic document preparation (no LLM calls).

Input : raw Markdown text from an incident or a runbook.
Output: list of "clean" sections + counters for discarded content.

Principles:
- Markdown is read line by line, preserving code blocks
  (``` / ~~~): a `# comment` inside a code block is NOT a heading.
- ONLY content that can be safely identified as boilerplate
  or an empty placeholder is discarded. When in doubt, the section is kept.
- The text of retained sections is not rewritten: it remains original.
- Each section preserves its heading path (section_path).
"""

import re
from collections import Counter

# ---------------------------------------------------------------------------
# Regex and constants
# ---------------------------------------------------------------------------

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
FENCE_OPEN_RE = re.compile(r"^\s*(`{3,}|~{3,})")
HTML_COMMENT_RE = re.compile(r"<!--.*?-->")
FRONT_MATTER_RE = re.compile(r"\A---\s*\n.*?\n---\s*(?:\n|\Z)", re.DOTALL)
TOC_RE = re.compile(r"^\s*\[\[_TOC_\]\]\s*$", re.MULTILINE)

HTML_TAG_RE = re.compile(r"</?[A-Za-z][^>]*>")
LIST_MARKER_RE = re.compile(r"(?:^|\s)(?:\d+[.)]|[-*+])(?=\s|$)")

# `{+ text +}` is the GitLab syntax used by templates for placeholders.
PLACEHOLDER_WRAP_RE = re.compile(r"\{[+-]\s*(.*?)\s*[+-]\}", re.DOTALL)

# Values that carry no information on their own. This list is intentionally short:
# "none" and "under investigation" are NOT included because they can be
# valid answers ("Customer impact: none").
EMPTY_VALUE_RE = re.compile(
    r"(?:tbd|tba|todo|to be determined|unknown|n/?a|\.+|\u2026|-+|_+)",
    re.IGNORECASE,
)

# Template instruction phrases (the same ones used by get_incidents.py).
TEMPLATE_PHRASES = (
    "start with the following",
    "provide any relevant",
    "include any additional",
    "how could detection time be improved",
)

# Generic template headings. The section is discarded only if its body
# consists of a checklist/link list (see _is_template_list_line).
BOILERPLATE_HEADINGS = {
    "create related issues",
    "guidelines",
    "resources",
    "table of contents",
    "toc",
}
BOILERPLATE_HEADING_CONTAINS = ("checklist",)

# Inline fields (e.g. CMOC): "1. Root cause: ...", "**Mitigation:** ...".
INLINE_FIELD_RE = re.compile(
    r"^\s*(?:(?:\d+[.)]|[-*+])\s*)?\**\s*"
    r"(root causes?|causes?|mitigation|resolution|resolved by|"
    r"corrective actions?|customer impact|service impact|"
    r"impact duration|impact|symptoms?)"
    r"\s*\**\s*:\s*\**\s*(.*)$",
    re.IGNORECASE,
)
FIELD_TYPES = {
    "root cause": "root_cause", "root causes": "root_cause",
    "cause": "root_cause", "causes": "root_cause",
    "mitigation": "resolution", "resolution": "resolution",
    "resolved by": "resolution",
    "corrective action": "corrective_action",
    "corrective actions": "corrective_action",
    "customer impact": "symptoms", "service impact": "symptoms",
    "impact duration": "symptoms", "impact": "symptoms",
    "symptom": "symptoms", "symptoms": "symptoms",
}

# Heading classification: the first matching rule wins.
# The section heading is checked first, followed by its ancestors (nearest first).
TYPE_RULES = [
    ("root_cause", re.compile(r"root cause|\bcauses?\b")),
    ("corrective_action", re.compile(r"corrective action|follow[- ]?up|action items")),
    ("resolution", re.compile(r"resolution|mitigation|remediation|resolved|\bfix(es)?\b|recovery")),
    ("diagnosis", re.compile(r"diagnos|investigat|troubleshoot|debug|\bchecks?\b|how to (check|find|identify)")),
    ("symptoms", re.compile(r"symptom|impact|detection|what happened|summary")),
    ("timeline", re.compile(r"timeline|current status|key events|status updates")),
    ("procedure", re.compile(r"procedure|\bsteps?\b|how to|instructions|runbook")),
]

SECTION_TYPES = (
    "root_cause", "resolution", "diagnosis", "symptoms", "timeline",
    "corrective_action", "procedure", "other",
)


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def normalize_heading(heading):
    """':books: References?' -> 'references'. Used only for comparisons."""
    if not heading:
        return ""
    h = re.sub(r":[a-z0-9_+-]+:", " ", heading, flags=re.IGNORECASE)
    h = re.sub(r"[^A-Za-z0-9]+", " ", h)
    return h.strip().lower()


def is_meaningful(text):
    """Return True if the text contains at least one line with actual information
    (not empty, not a placeholder such as TBD / {+ TBD +})."""
    unwrapped = PLACEHOLDER_WRAP_RE.sub(lambda m: m.group(1), text)
    unwrapped = HTML_TAG_RE.sub(" ", unwrapped)   # </summary>, </details>, ...
    for line in unwrapped.splitlines():
        # Remove list markers ("1.", "-", "*") and lines containing only dashes.
        line = LIST_MARKER_RE.sub(" ", line)
        cleaned = line.strip(" \t*_`>-:#")
        if not cleaned:
            continue
        if EMPTY_VALUE_RE.fullmatch(cleaned.lower()):
            continue
        if re.search(r"[A-Za-z0-9]", cleaned):
            return True
    return False


def _is_template_text_only(body):
    lines = [l.strip().lower() for l in body.splitlines() if l.strip()]
    return bool(lines) and all(
        any(p in l for p in TEMPLATE_PHRASES) for l in lines
    )


_URL_RE = re.compile(r"https?://|\]\(")


def _is_template_list_line(line):
    """Typical template list line: a checkbox, a short bullet with a link,
    or just a URL."""
    s = line.strip()
    if re.match(r"^[-*+]\s+\[[ xX]\]", s):
        return True
    if _URL_RE.search(s) and len(s.split()) <= 15:
        return True
    return False


# ---------------------------------------------------------------------------
# Markdown parsing (aware of fences and HTML comments)
# ---------------------------------------------------------------------------

def parse_markdown(text):
    """Split the text into sections, one per heading.

    Each section: {path: [ancestor headings..., heading], heading, level,
    lines: [(line_text, in_code)]}. Lines before the first heading
    form a section with heading None.
    """
    sections = []
    stack = []
    cur = {"path": [], "heading": None, "level": 0, "lines": []}
    fence = None  # (character, length) of the opening fence
    in_comment = False

    def flush():
        if cur["heading"] is not None or any(l.strip() for l, _ in cur["lines"]):
            sections.append(cur)

    for raw in text.splitlines():
        line = raw.rstrip()

        if fence:
            cur["lines"].append((line, True))
            m = re.match(r"^\s*(`{3,}|~{3,})\s*$", line)
            if m and m.group(1)[0] == fence[0] and len(m.group(1)) >= fence[1]:
                fence = None
            continue

        # HTML comments (including multi-line comments) outside code blocks.
        if in_comment:
            end = line.find("-->")
            if end == -1:
                continue
            line = line[end + 3:]
            in_comment = False
        line = HTML_COMMENT_RE.sub("", line)
        start = line.find("<!--")
        if start != -1:
            line = line[:start]
            in_comment = True

        m = FENCE_OPEN_RE.match(line)
        if m:
            fence = (m.group(1)[0], len(m.group(1)))
            cur["lines"].append((line, True))
            continue

        h = HEADING_RE.match(line)
        if h:
            flush()
            level = len(h.group(1))
            title = re.sub(r"\s+#+\s*$", "", h.group(2)).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            cur = {"path": [t for _, t in stack], "heading": title,
                   "level": level, "lines": []}
            continue

        cur["lines"].append((line, False))

    flush()
    return sections


# ---------------------------------------------------------------------------
# Inline fields (CMOC and similar)
# ---------------------------------------------------------------------------

def extract_inline_fields(lines):
    """Find 'Root cause: ...' fields outside code blocks.

    Returns (cleaned_lines, fields, n_removed_placeholder_lines).
    - Field with a meaningful value -> returned in `fields` (the line stays in the text).
    - Line with an explicit placeholder value (TBD, {+ TBD +}) and no continuation
      lines -> removed. A line with a label and an empty value -> kept.
    """
    cleaned, fields, stripped = [], [], 0
    i, n = 0, len(lines)
    while i < n:
        text, in_code = lines[i]
        m = None if in_code else INLINE_FIELD_RE.match(text)
        if not m:
            cleaned.append(lines[i])
            i += 1
            continue

        label = re.sub(r"\s+", " ", m.group(1).lower())
        value_parts = [m.group(2).strip()]
        j = i + 1
        while j < n:
            t, c = lines[j]
            if c or not t.strip() or INLINE_FIELD_RE.match(t):
                break
            value_parts.append(t.strip())
            j += 1
        value = "\n".join(p for p in value_parts if p)
        has_continuation = j > i + 1

        if is_meaningful(value):
            fields.append({"label": m.group(1).strip(),
                           "section_type": FIELD_TYPES[label],
                           "text": value})
            cleaned.extend(lines[i:j])
        elif m.group(2).strip() and not has_continuation:
            stripped += 1          # explicit placeholder: removed
        else:
            cleaned.extend(lines[i:j])   # label only: keep it
        i = j
    return cleaned, fields, stripped


# ---------------------------------------------------------------------------
# Classification and filtering
# ---------------------------------------------------------------------------

def classify_section(path, kind, body, has_code):
    for title in reversed(path):
        t = normalize_heading(title)
        for stype, rx in TYPE_RULES:
            if rx.search(t):
                return stype
    if kind == "runbook" and (has_code or re.search(r"^\s*\d+[.)]\s", body, re.M)):
        return "procedure"
    return "other"


def discard_reason(heading_norm, body, has_code):
    """Return the discard reason, or None if the section should be kept."""
    if not body.strip() and not has_code:
        return "heading_only"
    if not has_code and not is_meaningful(body):
        return "empty_placeholder"
    if not has_code and _is_template_text_only(body):
        return "template_instruction_text"
    is_boilerplate_heading = (
        heading_norm in BOILERPLATE_HEADINGS
        or any(s in heading_norm for s in BOILERPLATE_HEADING_CONTAINS)
    )
    if is_boilerplate_heading and not has_code:
        non_empty = [l for l in body.splitlines() if l.strip()]
        if non_empty and all(_is_template_list_line(l) for l in non_empty):
            return "boilerplate_template_list"
    return None


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def prepare_document(text, kind="incident"):
    """Return {"sections": [...], "discarded": Counter, "discarded_chars":
    Counter, "kept_notes": Counter, "chars_in": int, "chars_kept": int}.

    Each section: {index, section_path, section_type, origin, text}.
    origin = "section" or "inline_field" (a field extracted from a section:
    its text is also present in the original section).
    """
    discarded, discarded_chars, notes = Counter(), Counter(), Counter()
    text = text or ""

    if kind == "runbook":
        stripped = FRONT_MATTER_RE.sub("", text, count=1)
        if stripped != text:
            discarded["front_matter"] += 1
            discarded_chars["front_matter"] += len(text) - len(stripped)
            text = stripped
        toc_hits = len(TOC_RE.findall(text))
        if toc_hits:
            discarded["toc_marker"] += toc_hits
            text = TOC_RE.sub("", text)

    sections = []

    def add(path, stype, origin, body):
        sections.append({
            "index": len(sections),
            "section_path": " > ".join(path),
            "section_type": stype,
            "origin": origin,
            "text": body,
        })

    for sec in parse_markdown(text):
        lines, fields, n_stripped = extract_inline_fields(sec["lines"])
        discarded["empty_placeholder_line"] += n_stripped
        body = "\n".join(l for l, _ in lines).strip()
        has_code = any(c for _, c in lines)
        heading_norm = normalize_heading(sec["heading"])

        reason = discard_reason(heading_norm, body, has_code)
        if reason:
            discarded[reason] += 1
            discarded_chars[reason] += len(body)
            continue

        if (heading_norm in BOILERPLATE_HEADINGS
                or any(s in heading_norm for s in BOILERPLATE_HEADING_CONTAINS)):
            notes["boilerplate_heading_kept_has_content"] += 1

        stype = classify_section(sec["path"], kind, body, has_code)
        add(sec["path"], stype, "section", body)
        for f in fields:
            add(sec["path"] + [f["label"]], f["section_type"],
                "inline_field", f["text"])

    chars_kept = sum(len(s["text"]) for s in sections if s["origin"] == "section")
    return {
        "sections": sections,
        "discarded": discarded,
        "discarded_chars": discarded_chars,
        "kept_notes": notes,
        "chars_in": len(text),
        "chars_kept": chars_kept,
    }


def first_h1(text):
    """Return the first level-1 heading outside code blocks (or None)."""
    for sec in parse_markdown(text or ""):
        if sec["level"] == 1:
            return sec["heading"]
    return None