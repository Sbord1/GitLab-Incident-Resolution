"""
Download public Markdown runbooks from the gitlab-com/runbooks repository
and export them to raw_runbooks.json.

Strategy:
1. Recursively reads the tree under docs/ via the GitLab API.
2. Counts all .md files and shows the distribution by service.
3. If the total number of .md files is <= MAX_ALL_RUNBOOKS, downloads all runbooks.
4. If it exceeds the threshold, downloads only the services in SELECTED_SERVICES.
5. Saves the Markdown in a nearly raw format, without summarizing it for the knowledge base.

Note: README.md is excluded by default because it is often a service overview
rather than an operational runbook. Set INCLUDE_README = True to include it.
"""

import json
import os
import time
from collections import Counter
from urllib.parse import quote

import requests


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_PATH = "gitlab-com/runbooks"
PROJECT_ID = quote(PROJECT_PATH, safe="")
API_BASE = "https://gitlab.com/api/v4"
REF = "master"
ROOT_PATH = "docs"
OUTPUT_JSON = "../data/raw/raw_runbooks.json"

PER_PAGE = 100
REQUEST_TIMEOUT = 30
SLEEP_SECONDS = 0.15

# If the total number of operational Markdown files is below this threshold, download everything.
MAX_ALL_RUNBOOKS = 1000

INCLUDE_README = False

# Used only if the repository exceeds MAX_ALL_RUNBOOKS.
# Names = directories directly under docs/.
SELECTED_SERVICES = {
    "gitaly",
    "patroni",
    "postgres-dr-delayed",
    "pgbouncer",
    "redis",
    "sidekiq",
    "api",
    "web",
    "prometheus",
    "alertmanager",
    "kas",
    "logging",
    "elastic",
    "cloudflare",
    "ci-runners",
}

TOKEN = os.environ.get("GITLAB_TOKEN")
HEADERS = {"PRIVATE-TOKEN": TOKEN} if TOKEN else {}


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def api_get(url: str, params=None) -> requests.Response:
    """GET with basic rate-limit and temporary error handling."""
    for attempt in range(5):
        response = requests.get(
            url,
            params=params,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
        )

        if response.status_code == 429:
            wait = int(response.headers.get("Retry-After", "5"))
            print(f"Rate limit: waiting {wait}s...")
            time.sleep(wait)
            continue

        if response.status_code >= 500 and attempt < 4:
            wait = 2 ** attempt
            print(f"GitLab {response.status_code}: retrying in {wait}s...")
            time.sleep(wait)
            continue

        response.raise_for_status()
        return response

    raise RuntimeError("Unable to complete the GitLab API request")


# ---------------------------------------------------------------------------
# Repository tree
# ---------------------------------------------------------------------------

def fetch_repository_tree() -> list[dict]:
    """Recursively fetches all items under docs/."""
    items = []
    page = 1
    url = f"{API_BASE}/projects/{PROJECT_ID}/repository/tree"

    while True:
        response = api_get(
            url,
            params={
                "path": ROOT_PATH,
                "ref": REF,
                "recursive": "true",
                "per_page": PER_PAGE,
                "page": page,
            },
        )

        batch = response.json()
        if not batch:
            break

        items.extend(batch)
        print(f"Tree page {page}: {len(batch)} items (total {len(items)})")

        next_page = response.headers.get("X-Next-Page")
        if not next_page:
            break

        page = int(next_page)
        time.sleep(SLEEP_SECONDS)

    return items


def service_from_path(path: str) -> str:
    """docs/gitaly/alerts/X.md -> gitaly"""
    parts = path.split("/")
    if len(parts) >= 2 and parts[0] == ROOT_PATH:
        return parts[1]
    return "uncategorized"


def is_markdown_runbook(item: dict) -> bool:
    if item.get("type") != "blob":
        return False

    path = item.get("path", "")
    if not path.lower().endswith(".md"):
        return False

    if not INCLUDE_README and path.lower().endswith("/readme.md"):
        return False

    return True


# ---------------------------------------------------------------------------
# Document type
# ---------------------------------------------------------------------------

def infer_doc_type(path: str) -> str:
    """Simple structural classification, not semantic."""
    lower = path.lower()

    if "/alerts/" in lower:
        return "alert_playbook"

    return "troubleshooting_guide"


def alert_name_from_path(path: str, doc_type: str):
    if doc_type != "alert_playbook":
        return None
    return os.path.splitext(os.path.basename(path))[0]


# ---------------------------------------------------------------------------
# File download
# ---------------------------------------------------------------------------

def fetch_markdown(path: str) -> str:
    encoded_path = quote(path, safe="")
    url = (
        f"{API_BASE}/projects/{PROJECT_ID}/repository/files/"
        f"{encoded_path}/raw"
    )

    response = api_get(url, params={"ref": REF})
    return response.text.strip()


def gitlab_blob_url(path: str) -> str:
    return f"https://gitlab.com/{PROJECT_PATH}/-/blob/{REF}/{path}"


def gitlab_raw_url(path: str) -> str:
    return f"https://gitlab.com/{PROJECT_PATH}/-/raw/{REF}/{path}"


def published_url(path: str) -> str:
    # docs/gitaly/alerts/X.md ->
    # https://runbooks.gitlab-static.net/gitaly/alerts/X/
    relative = path
    prefix = ROOT_PATH + "/"
    if relative.startswith(prefix):
        relative = relative[len(prefix):]

    if relative.lower().endswith(".md"):
        relative = relative[:-3]

    return f"https://runbooks.gitlab-static.net/{relative}/"


# ---------------------------------------------------------------------------
# Raw record
# ---------------------------------------------------------------------------

def build_raw_record(item: dict) -> dict:
    path = item["path"]
    service = service_from_path(path)
    doc_type = infer_doc_type(path)
    alert_name = alert_name_from_path(path, doc_type)

    content = fetch_markdown(path)

    return {
        "path": path,
        "service": service,
        "filename": os.path.basename(path),
        "doc_type": doc_type,
        "alert_name": alert_name,
        "url": gitlab_raw_url(path),
        "blob_url": gitlab_blob_url(path),
        "published_url": published_url(path),
        "content": content,
        "content_length": len(content),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Scanning repository {PROJECT_PATH}, path {ROOT_PATH}/ ...")

    tree = fetch_repository_tree()
    markdown_files = [item for item in tree if is_markdown_runbook(item)]

    service_counts = Counter(
        service_from_path(item["path"])
        for item in markdown_files
    )

    print("\n=== Markdown runbook count ===")
    print(f"Total operational .md files: {len(markdown_files)}")
    print(f"README files included: {INCLUDE_README}")

    print("\n=== Distribution by service ===")
    for service, count in sorted(
        service_counts.items(),
        key=lambda x: (-x[1], x[0]),
    ):
        print(f"{service:35s} {count:4d}")

    if len(markdown_files) <= MAX_ALL_RUNBOOKS:
        selected = markdown_files
        mode = "all"
        print(
            f"\nTotal <= threshold ({MAX_ALL_RUNBOOKS}): "
            "downloading all runbooks."
        )
    else:
        selected = [
            item
            for item in markdown_files
            if service_from_path(item["path"]) in SELECTED_SERVICES
        ]
        mode = "selected_services"
        print(
            f"\nTotal > threshold ({MAX_ALL_RUNBOOKS}): "
            "downloading only the selected services."
        )
        print("Requested services:", ", ".join(sorted(SELECTED_SERVICES)))

    print(f"Files to download: {len(selected)}")

    records = []
    failures = []

    for index, item in enumerate(selected, start=1):
        path = item["path"]
        print(f"[{index}/{len(selected)}] {path}")

        try:
            records.append(build_raw_record(item))
        except requests.RequestException as exc:
            print(f"  ERROR: {exc}")
            failures.append({"path": path, "error": str(exc)})

        time.sleep(SLEEP_SECONDS)

    records.sort(key=lambda r: (r["service"], r["path"].lower()))

    output = {
        "source": {
            "project": PROJECT_PATH,
            "ref": REF,
            "root_path": ROOT_PATH,
            "selection_mode": mode,
            "include_readme": INCLUDE_README,
            "total_markdown_runbooks_found": len(markdown_files),
            "selected_for_download": len(selected),
            "downloaded": len(records),
            "failed": len(failures),
            "service_counts": dict(sorted(service_counts.items())),
            "selected_services_if_filtered": (
                sorted(SELECTED_SERVICES)
                if mode == "selected_services"
                else []
            ),
            "failures": failures,
        },
        "runbooks": records,
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nDataset saved to: {OUTPUT_JSON}")
    print(f"Runbooks downloaded: {len(records)}")
    print(f"Failed downloads: {len(failures)}")


if __name__ == "__main__":
    main()