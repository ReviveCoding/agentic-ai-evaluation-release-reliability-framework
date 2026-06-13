from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
REPORT_JSON = ROOT / "reports" / "public_release_audit.json"
REPORT_MD = ROOT / "reports" / "public_release_audit.md"
MAX_TRACKED_BYTES = 5 * 1024 * 1024

REQUIRED_PATHS = (
    "LICENSE",
    "NOTICE",
    "README.md",
    "SECURITY.md",
    "CONTRIBUTING.md",
    "CODE_OF_CONDUCT.md",
    "SUPPORT.md",
    ".github/CODEOWNERS",
    ".github/dependabot.yml",
    ".github/pull_request_template.md",
    ".github/workflows/ci.yml",
    ".github/workflows/codeql.yml",
    ".github/workflows/release-check.yml",
    "docs/THREAT_MODEL.md",
    "docs/EVALUATION_PROTOCOL.md",
    "docs/REPOSITORY_MAP.md",
    "docs/GITHUB_SETTINGS_CHECKLIST.md",
    "pyproject.toml",
)

TEXT_SUFFIXES = {
    ".cfg",
    ".cff",
    ".css",
    ".csv",
    ".dockerignore",
    ".gitignore",
    ".html",
    ".ini",
    ".json",
    ".md",
    ".ps1",
    ".py",
    ".rst",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

TEXT_FILENAMES = {
    "Dockerfile",
    "Makefile",
    "LICENSE",
    "NOTICE",
}

GENERATED_AUDIT_REPORTS = {
    "reports/public_release_audit.json",
    "reports/public_release_audit.md",
}

FORBIDDEN_PREFIXES = (
    ".venv/",
    "_doc_backups/",
    "_patch_backups/",
    "build/",
    "dist/",
    "models/",
    "outputs/",
    "payload/",
)

SECRET_PATTERNS = (
    (
        "private key",
        re.compile(
            r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"
        ),
    ),
    ("AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    (
        "GitHub token",
        re.compile(r"\bgh(?:p|o|u|s|r)_[A-Za-z0-9]{20,}\b"),
    ),
    (
        "Slack token",
        re.compile(r"\bxox(?:b|p|a|r|s)-[A-Za-z0-9-]{20,}\b"),
    ),
)

# Build audit tokens from fragments so this source file does not contain
# the complete forbidden placeholders and cannot flag itself.
PLACEHOLDER_TOKENS = (
    "<" + "PRODUCT" + "_REF",
    "<" + "YOUR_GITHUB" + "_REPOSITORY_URL>",
    "YOUR_GITHUB" + "_REPOSITORY_URL",
    "REPLACE" + "_WITH" + "_REAL" + "_VALUE",
    "TODO" + "_REPLACE" + "_ME",
)


def run_git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def tracked_files() -> list[str]:
    output = run_git("ls-files", "-z")
    return [item for item in output.split("\0") if item]


def is_text_candidate(relative_path: str) -> bool:
    path = Path(relative_path)
    return (
        path.name in TEXT_FILENAMES
        or path.suffix.lower() in TEXT_SUFFIXES
        or path.name.startswith(".")
    )


def read_text(relative_path: str) -> str | None:
    path = ROOT / relative_path
    try:
        raw = path.read_bytes()
    except OSError:
        return None

    if b"\0" in raw:
        return None

    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("utf-8", errors="replace")


def is_forbidden_helper(relative_path: str) -> bool:
    path = Path(relative_path)
    lower = relative_path.lower()
    name = path.name.lower()

    if any(lower.startswith(prefix.lower()) for prefix in FORBIDDEN_PREFIXES):
        return True

    if lower.startswith("data/processed/") and name != ".gitkeep":
        return True

    if ".before_" in name or name.endswith(".bak") or name.endswith(".backup"):
        return True

    if len(path.parts) == 1:
        if name.startswith(("apply_", "patch_", "fix_", "resume_")):
            return True
        if name.endswith(".ps1") and "hardening" in name:
            return True

    return False


def check_workflow_pinning(relative_path: str, text: str) -> list[str]:
    failures: list[str] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped.startswith("uses:"):
            continue

        reference = stripped.removeprefix("uses:").strip()
        reference = reference.split(" #", maxsplit=1)[0].strip()

        if reference.startswith(("./", "docker://")):
            continue

        if "@" not in reference:
            failures.append(
                f"workflow action without version in {relative_path}:{line_number}"
            )
            continue

        _, version = reference.rsplit("@", maxsplit=1)
        if not re.fullmatch(r"[0-9a-fA-F]{40}", version):
            failures.append(
                "workflow action is not pinned to a full 40-character SHA "
                f"in {relative_path}:{line_number}: {reference}"
            )
    return failures


def markdown_report(
    status: str,
    tracked_count: int,
    workflow_count: int,
    failures: Iterable[str],
) -> str:
    failure_list = list(failures)
    lines = [
        "# Public Release Audit",
        "",
        f"- Status: **{status}**",
        f"- Tracked files inspected: **{tracked_count}**",
        f"- Workflow files inspected: **{workflow_count}**",
        f"- Failures: **{len(failure_list)}**",
        "",
    ]

    if failure_list:
        lines.extend(["## Failures", ""])
        lines.extend(f"- {failure}" for failure in failure_list)
    else:
        lines.extend(
            [
                "## Result",
                "",
                "No blocking public-release issues were found.",
            ]
        )

    lines.append("")
    return "\n".join(lines)


def main() -> int:
    failures: list[str] = []

    try:
        files = tracked_files()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"Unable to enumerate tracked files: {exc}", file=sys.stderr)
        return 2

    tracked_set = set(files)

    for required in REQUIRED_PATHS:
        if required not in tracked_set and not (ROOT / required).exists():
            failures.append(f"missing required public-release file: {required}")

    workflow_count = 0

    for relative_path in files:
        normalized = relative_path.replace("\\", "/")
        path = ROOT / normalized

        if is_forbidden_helper(normalized):
            failures.append(f"forbidden generated/helper path is tracked: {normalized}")

        try:
            size = path.stat().st_size
        except OSError:
            failures.append(f"tracked file is missing from working tree: {normalized}")
            continue

        if size > MAX_TRACKED_BYTES:
            failures.append(
                f"tracked file exceeds 5 MiB: {normalized} ({size} bytes)"
            )

        if not is_text_candidate(normalized):
            continue

        text = read_text(normalized)
        if text is None:
            continue

        if normalized not in GENERATED_AUDIT_REPORTS:
            for token in PLACEHOLDER_TOKENS:
                if token in text:
                    failures.append(
                        f"unresolved placeholder {token!r} in {normalized}"
                    )

            for label, pattern in SECRET_PATTERNS:
                if pattern.search(text):
                    failures.append(f"possible {label} in {normalized}")

        if normalized.startswith(".github/workflows/") and normalized.endswith(
            (".yml", ".yaml")
        ):
            workflow_count += 1
            failures.extend(check_workflow_pinning(normalized, text))

    pyproject_path = ROOT / "pyproject.toml"
    if pyproject_path.exists():
        pyproject_text = pyproject_path.read_text(
            encoding="utf-8", errors="replace"
        )
        if "Apache-2.0" not in pyproject_text:
            failures.append(
                "pyproject.toml does not declare the Apache-2.0 license"
            )
        if "ReviveCoding/agentic-ai-evaluation-release-reliability-framework" not in (
            pyproject_text
        ):
            failures.append(
                "pyproject.toml is missing the canonical GitHub project URL"
            )

    status = "PASS" if not failures else "FAIL"
    payload = {
        "status": status,
        "tracked_file_count": len(files),
        "workflow_count": workflow_count,
        "failures": failures,
    }

    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    REPORT_MD.write_text(
        markdown_report(status, len(files), workflow_count, failures),
        encoding="utf-8",
    )

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())