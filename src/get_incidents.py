"""
Downloads public GitLab incidents from gl-infra/production
closed since 2024 and builds raw_issues.json.

Goals:
1. Download incidents closed after 2024.
2. Remove obvious noise from GitLab templates in descriptions.
3. Assign an approximate quality score to help with manual filtering.
4. Sort incidents with the best candidates at the top.

Output:
    raw_issues.json
"""

import os
import re
import json
import time
import requests
from collections import Counter


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_PATH = "gitlab-com/gl-infra/production"
PROJECT_ID = requests.utils.quote(PROJECT_PATH, safe="")
API_BASE = "https://gitlab.com/api/v4"

LABELS = "incident"
STATE = "closed"
CLOSED_AFTER = "2024-01-01T00:00:00Z"

PER_PAGE = 100
OUTPUT_JSON = "../data/raw/raw_issues.json"

TOKEN = os.environ.get("GITLAB_TOKEN")
HEADERS = {"PRIVATE-TOKEN": TOKEN} if TOKEN else {}


# ---------------------------------------------------------------------------
# Patterns for cleaning and quality analysis
# ---------------------------------------------------------------------------

HTML_COMMENT_RE = re.compile(
    r"<!--.*?-->",
    re.DOTALL
)


# Everything after this section is removed from the description.
# It's usually GitLab template content: links, runbooks, corrective actions,
# security notes, deployment guidance, etc.
REFERENCES_SECTION_RE = re.compile(
    r"#{1,6}\s*(?::books:\s*)?References and helpful links.*",
    re.IGNORECASE | re.DOTALL,
)


TEST_ARTIFACT_MARKERS = [
    "Failure/Error",
    "Finished in",
    "Failed examples:",
    "rspec ./",
    "examples, 0 failures",
]


TEMPLATE_PLACEHOLDER_PHRASES = [
    "start with the following",
    "provide any relevant",
    "include any additional",
    "how could detection time be improved",
]


# Patterns used to identify the beginning of useful sections.
# Finding a heading alone is not enough: its associated content
# must also pass the has_meaningful_signal() check.
ROOT_CAUSE_RE = re.compile(
    r"(?:#{1,4}\s*(?:root cause|causes?|what (?:was|were) the root cause)|\broot cause\s*:|\bcauses?\s*:)",
    re.IGNORECASE,
)

RESOLUTION_RE = re.compile(
    r"(?:#{1,4}\s*(?:resolution|mitigation|corrective actions?)|\bresolved by\b|\bmitigated by\b|\bmitigation\s*:|\bresolution\s*:)",
    re.IGNORECASE,
)

IMPACT_RE = re.compile(
    r"(?:#{1,4}\s*(?:impact|customer impact)|\bcustomer impact\s*:|\bservice impact\s*:|\bimpact\s*:)",
    re.IGNORECASE,
)


# Boundaries used to avoid including text from the next section.
SECTION_HEADING_RE = re.compile(r"(?m)^#{1,6}\s+.+$")
NUMBERED_FIELD_RE = re.compile(
    r"(?m)^\s*\d+\.\s*(?:customer impact|service impact|impact duration|root cause)\s*:",
    re.IGNORECASE,
)


# Content that doesn't provide any meaningful information about the incident.
EMPTY_SIGNAL_RE = re.compile(
    r"^(?:"
    r"tbd|todo|unknown|n/?a|none|not known|under investigation|"
    r"\.{1,}|-+|_+|"
    r"\{\+.*?\+\}"
    r")$",
    re.IGNORECASE | re.DOTALL,
)


def clean_description(description: str) -> str:
    """
    Clean the description BEFORE saving it to raw_issues.json.

    Removes:
    - GitLab template HTML comments
    - The "References and helpful links" section and everything after it
    - Excessive blank lines

    Does not summarize or rewrite the incident content.
    """
    text = description or ""
    text = HTML_COMMENT_RE.sub("", text)
    text = REFERENCES_SECTION_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _signal_content(text: str, match: re.Match, max_chars: int = 1200) -> str:
    """Extract the text associated with a signal without including the next section."""
    start = match.end()
    end = min(len(text), start + max_chars)

    heading = SECTION_HEADING_RE.search(text, start, end)
    numbered = NUMBERED_FIELD_RE.search(text, start, end)
    boundaries = [m.start() for m in (heading, numbered) if m]
    if boundaries:
        end = min(boundaries)

    content = text[start:end].strip()
    # Remove Markdown punctuation and separators immediately after the label.
    return re.sub(r"^[\s:*\`_-]+", "", content).strip()


def _is_meaningful(content: str) -> bool:
    """Return True only if the content isn't an empty placeholder or template."""
    if not content:
        return False

    first_line = content.splitlines()[0].strip()
    normalized = re.sub(r"\s+", " ", first_line).strip(" `*_:-")

    if not normalized or EMPTY_SIGNAL_RE.fullmatch(normalized):
        return False

    lower = normalized.lower()
    if "{+" in normalized or "+}" in normalized:
        return False
    if lower in {"tbd", "todo", "unknown", "n/a", "none", "..."}:
        return False

    # Require at least a small amount of actual textual content.
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9_.:/-]*", normalized)
    return len(words) >= 3


def has_meaningful_signal(text: str, pattern: re.Pattern) -> bool:
    """
    A signal only counts if at least one occurrence contains actual information.
    Example: `Root cause: {+ TBD +}` -> False.
    """
    for match in pattern.finditer(text):
        # Inline phrases such as "resolved by X" already include part of
        # the meaning in the match; the following text must still specify X.
        content = _signal_content(text, match)
        if _is_meaningful(content):
            return True
    return False


def has_real_content(text: str) -> bool:
    """Filter out incidents that are mostly empty or made up of placeholders."""
    cleaned = text.strip()
    if len(cleaned) < 50:
        return False

    lower = cleaned.lower()
    placeholder_count = sum(phrase in lower for phrase in TEMPLATE_PLACEHOLDER_PHRASES)
    explicit_placeholders = len(re.findall(r"\{\+.*?\+\}|\bTBD\b|^\s*\.\.\.\s*$", cleaned, re.I | re.M | re.S))

    # Multiple unfilled template markers make the record unreliable for the KB.
    return placeholder_count < 2 and explicit_placeholders < 3


def classify(description: str) -> dict:
    """Assign an approximate quality score for later manual filtering."""
    text = description.strip()
    lower = text.lower()

    if any(marker.lower() in lower for marker in TEST_ARTIFACT_MARKERS):
        return {
            "quality": "discard", "reason": "test_artifact", "score": 0,
            "signals": {
                "has_root_cause": False, "has_resolution": False,
                "has_impact": False, "content_length": len(text),
            },
        }

    has_root_cause = has_meaningful_signal(text, ROOT_CAUSE_RE)
    has_resolution = has_meaningful_signal(text, RESOLUTION_RE)
    has_impact = has_meaningful_signal(text, IMPACT_RE)

    content_length = len(text)
    real_content = has_real_content(text)
    signals = {
        "has_root_cause": has_root_cause,
        "has_resolution": has_resolution,
        "has_impact": has_impact,
        "content_length": content_length,
    }

    if not real_content:
        return {
            "quality": "discard", "reason": "template_or_too_little_content",
            "score": 0, "signals": signals,
        }

    score = 0
    if has_root_cause:
        score += 2
    if has_resolution:
        score += 2
    if has_impact:
        score += 1
    if content_length >= 500:
        score += 1
    if content_length >= 1500:
        score += 1

    quality = "high" if score >= 5 else "medium" if score >= 3 else "low"
    return {
        "quality": quality, "reason": "score_based",
        "score": score, "signals": signals,
    }


# ---------------------------------------------------------------------------
# Fetch GitLab
# ---------------------------------------------------------------------------

def fetch_issues():
    """
    Fetch all closed issues with the 'incident' label
    from the specified period.
    """

    issues = []
    page = 1

    while True:

        params = {
            "state": STATE,
            "labels": LABELS,
            "closed_after": CLOSED_AFTER,
            "per_page": PER_PAGE,
            "page": page,
        }

        response = requests.get(
            f"{API_BASE}/projects/{PROJECT_ID}/issues",
            params=params,
            headers=HEADERS,
            timeout=30,
        )

        response.raise_for_status()

        batch = response.json()

        if not batch:
            break

        issues.extend(batch)

        print(
            f"Page {page}: "
            f"downloaded {len(batch)} issues "
            f"(total: {len(issues)})"
        )

        page += 1

        time.sleep(
            0.5 if TOKEN else 1.5
        )

    return issues


# ---------------------------------------------------------------------------
# Build raw dataset
# ---------------------------------------------------------------------------

def build_raw_record(issue: dict) -> dict:
    """
    Build the record saved to raw_issues.json.

    The description is cleaned to remove obvious noise BEFORE
    being classified and saved.
    """

    # Description returned by the API
    original_description = issue.get("description") or ""

    # Cleaning
    description = clean_description(original_description)

    # Classification based on the cleaned version
    classification = classify(description)

    return {
        "iid": issue["iid"],

        "title": issue.get(
            "title",
            ""
        ),

        "url": issue.get(
            "web_url"
        ),

        "description": description,

        "labels": issue.get(
            "labels",
            []
        ),

        "created_at": issue.get(
            "created_at"
        ),

        "updated_at": issue.get(
            "updated_at"
        ),

        "closed_at": issue.get(
            "closed_at"
        ),

        "analysis": {
            "quality":
                classification["quality"],

            "score":
                classification["score"],

            "reason":
                classification["reason"],

            "signals":
                classification["signals"],

            "num_labels":
                len(issue.get("labels", [])),
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    print(
        f"Downloading incidents from {PROJECT_PATH} "
        f"closed after {CLOSED_AFTER}..."
    )

    issues = fetch_issues()

    print(
        f"\nTotal issues downloaded: {len(issues)}"
    )

    if not issues:
        print("No issues found.")
        return

    # ---------------------------------------------------------
    # Build dataset
    # ---------------------------------------------------------

    records = [
        build_raw_record(issue)
        for issue in issues
    ]

    # ---------------------------------------------------------
    # Sorting
    #
    # 1. high > medium > low > discard
    # 2. Highest score first
    # 3. Most recent incidents first
    # ---------------------------------------------------------

    QUALITY_ORDER = {
        "high": 3,
        "medium": 2,
        "low": 1,
        "discard": 0,
    }

    records.sort(
        key=lambda record: (
            QUALITY_ORDER.get(
                record["analysis"]["quality"],
                0
            ),
            record["analysis"].get(
                "score",
                0
            ),
            record.get("closed_at") or "",
        ),
        reverse=True,
    )

    # ---------------------------------------------------------
    # Save JSON
    # ---------------------------------------------------------

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            records,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"\nDataset saved to: {OUTPUT_JSON}"
    )

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------

    quality_counts = Counter(
        record["analysis"]["quality"]
        for record in records
    )

    print(
        "\n=== Quality distribution ==="
    )

    for quality in [
        "high",
        "medium",
        "low",
        "discard",
    ]:

        count = quality_counts.get(
            quality,
            0
        )

        pct = (
            100 * count / len(records)
            if records
            else 0
        )

        print(
            f"{quality:10s}: "
            f"{count:4d} "
            f"({pct:.1f}%)"
        )

    # ---------------------------------------------------------
    # Top 30 candidates
    # ---------------------------------------------------------

    print(
        "\n=== Best candidates "
        "for manual review ==="
    )

    top_candidates = [
        record
        for record in records
        if record["analysis"]["quality"] == "high"
    ][:30]

    for record in top_candidates:

        print(
            f"#{record['iid']:5d} | "
            f"score={record['analysis']['score']} | "
            f"labels={record['analysis']['num_labels']:2d} | "
            f"{record['title']}"
        )

        print(
            f"  {record['url']}"
        )


if __name__ == "__main__":
    main()