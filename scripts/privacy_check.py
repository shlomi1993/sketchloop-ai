from __future__ import annotations

import re
import subprocess
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_DIRS = {"private", "data", "runs", "models", "tmp", "output", ".dist"}
PRIVATE_SUFFIXES = {".pdf", ".safetensors", ".ckpt", ".pt", ".pth"}

# Label each pattern by what it suggests, so findings can be reported without echoing the match.
PATTERNS = {
    "email address": re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"),
    "personal home path": re.compile(r"(?:/(?:Users|home)/|[A-Za-z]:\\Users\\)[^\s/\\]+"),
    "possible nine-digit personal identifier": re.compile(r"(?<!\d)\d{9}(?!\d)"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "possible API credential": re.compile(r"\b(?:sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9_]{20,})\b"),
}


def git(*args: str, root: Path = ROOT) -> bytes:
    return subprocess.check_output(["git", *args], cwd=root)


def public_paths(root: Path = ROOT) -> list[str]:
    """
    Include tracked and unignored untracked files; never traverse ignored data.
    """
    raw = git("ls-files", "--cached", "--others", "--exclude-standard", "-z", root=root)
    return sorted(set(p.decode("utf-8") for p in raw.split(b"\0") if p))


def is_public_article(name: str) -> bool:
    """
    Tell whether a path is an openly licensed article PDF allowed for publication.

    Args:
        name (str): Repository-relative POSIX path.

    Returns:
        bool: True for PDFs placed directly in docs/articles/.
    """
    path = PurePosixPath(name)
    return path.parent == PurePosixPath("docs/articles") and path.suffix.lower() == ".pdf"


def path_findings(name: str) -> list[str]:
    # Openly licensed article PDFs are the one allowed exception to the PDF rule.
    if is_public_article(name):
        return []

    # Flag model weights, PDFs, private data folders, and env files by name alone, case-insensitively.
    path = PurePosixPath(name.lower())
    if (
        path.suffix in PRIVATE_SUFFIXES
        or any(part in PRIVATE_DIRS for part in path.parts)
        or (path.name.startswith(".env") and path.name != ".env.example")
    ):
        return ["private artifact path"]
    return []


def content_findings(data: bytes) -> list[str]:
    # Detect PDFs by magic bytes so a renamed source document is still caught.
    if b"%PDF-" in data[:1024]:
        return ["PDF content (including renamed source documents)"]

    # Only text content can be pattern-matched.
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return []  # Binary additions require manual review; this is not a DLP system.
    return [label for label, pattern in PATTERNS.items() if pattern.search(text)]


def check(root: Path = ROOT) -> list[str]:
    # Scan worktree files that a commit could publish.
    findings = []
    for name in public_paths(root):
        path = root / name
        issues = path_findings(name)

        # Never follow a symlink, since its target may lie outside the repository.
        if path.is_symlink():
            issues.append("symlink requires explicit publication review")
        elif path.is_file() and not is_public_article(name):
            issues += content_findings(path.read_bytes())
        findings.extend(f"worktree {name}: {issue}" for issue in issues)

    # Scan index blobs too: a cleaned worktree must not hide an older staged secret.
    entries = git("ls-files", "--stage", "-z", root=root).split(b"\0")
    for entry in filter(None, entries):
        # Each entry is "mode object stage<TAB>path".
        metadata, raw_name = entry.split(b"\t", 1)
        mode, object_id, stage = metadata.decode("ascii").split()
        name = raw_name.decode("utf-8")
        issues = path_findings(name)
        if stage != "0":
            issues.append("unresolved merge entry")

        # Symlink and submodule entries have no file content to scan, so they need a human look.
        if mode in {"120000", "160000"}:
            issues.append("symlink/submodule requires explicit publication review")
        elif not is_public_article(name):
            issues += content_findings(git("cat-file", "blob", object_id, root=root))
        findings.extend(f"index {name}: {issue}" for issue in issues)
    return findings


def main() -> int:
    # List only the file and finding label, never the matched text.
    findings = check()
    if findings:
        print("Publication guard failed (matched content is withheld):")
        for finding in findings:
            print(f"- {finding}")
        return 1
    print("Publication guard passed for publishable worktree files and index blobs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
