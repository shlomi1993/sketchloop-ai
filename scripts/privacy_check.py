from __future__ import annotations

import codecs
import re
import subprocess

from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
NUL = b"\0"

# Label each pattern by what it suggests, so findings can be reported without echoing the match.
PATTERNS = {
    "email address": re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"),
    "personal home path": re.compile(r"(?:/(?:Users|home)/|[A-Za-z]:\\Users\\)[^\s/\\]+"),
    "possible nine-digit personal identifier": re.compile(r"(?<!\d)\d{9}(?!\d)"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "possible API credential": re.compile(r"\b(?:sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9_]{20,})\b"),
}


def git(*args: str, root: Path = ROOT, stdin: bytes | None = None,
        allowed_exit_codes: tuple[int, ...] = (0,)) -> subprocess.CompletedProcess[bytes]:
    """
    Run Git in the repository and validate its exit code.

    Args:
        args (str): Git subcommand and arguments.
        root (Path, optional): Repository directory. Defaults to ROOT.
        stdin (bytes | None, optional): Bytes passed to standard input. Defaults to None.
        allowed_exit_codes (tuple[int, ...], optional): Accepted exit codes. Defaults to (0,).

    Returns:
        subprocess.CompletedProcess[bytes]: Exit code and captured stdout and stderr as bytes.
    """
    result = subprocess.run(["git", *args], cwd=root, input=stdin, capture_output=True)
    if result.returncode not in allowed_exit_codes:
        result.check_returncode()

    return result


def public_paths(root: Path = ROOT) -> list[str]:
    """
    Include tracked and unignored untracked files; never traverse ignored data.
    """
    raw = git("ls-files", "--cached", "--others", "--exclude-standard", "-z", root=root).stdout
    return sorted(set(encoded_path.decode("utf-8") for encoded_path in raw.split(NUL) if encoded_path))


def find_ignored_paths(relative_paths: list[str], root: Path, *, worktree: Path | None = None) -> set[str]:
    """
    Find paths excluded by Git ignore rules, including tracked files.

    Args:
        relative_paths (list[str]): Repository-relative paths to check.
        root (Path): Repository directory.
        worktree (Path | None, optional): Alternate directory containing ignore files. Defaults to None.

    Returns:
        set[str]: Paths matching the effective ignore rules.
    """
    if not relative_paths:
        return set()

    # Ask Git once for all paths, preserving spaces and newlines in file paths.
    stdin = NUL.join(relative_path.encode("utf-8") for relative_path in relative_paths) + NUL
    options = [] if worktree is None else ["--work-tree", str(worktree)]
    result = git(*options, "check-ignore", "--no-index", "--stdin", "-z", root=root, stdin=stdin, allowed_exit_codes=(0, 1))
    return {encoded_path.decode("utf-8") for encoded_path in result.stdout.split(NUL) if encoded_path}


def find_staged_ignored_paths(relative_paths: list[str], root: Path) -> set[str]:
    """
    Check paths against staged ignore files in a temporary directory without changing the checkout or index.

    Args:
        relative_paths (list[str]): Repository-relative paths to check.
        root (Path): Repository directory.

    Returns:
        set[str]: Paths excluded by staged ignore rules and local or global exclusions.
    """
    # Reconstruct only normal staged ignore files, leaving the real worktree and index untouched.
    entries = git("ls-files", "--stage", "-z", "--", ".gitignore", ":(glob)**/.gitignore", root=root).stdout
    with TemporaryDirectory(prefix="sketchloop-ignore-") as directory:
        worktree = Path(directory)
        for entry in filter(None, entries.split(NUL)):
            metadata, encoded_path = entry.split(b"\t", 1)
            mode, object_id, stage = metadata.decode("ascii").split()
            if stage != "0" or mode not in {"100644", "100755"}:  # Regular files, non-executable or executable.
                continue

            # Materialize nested rules so Git applies their original scope and exceptions.
            path = worktree / encoded_path.decode("utf-8")
            path.parent.mkdir(parents=True, exist_ok=True)
            blob = git("cat-file", "blob", object_id, root=root).stdout
            path.write_bytes(blob)

        return find_ignored_paths(relative_paths, root, worktree=worktree)


def is_public_article(relative_path: str) -> bool:
    """
    Return True if the path points to a PDF directly under `docs/articles/`, where openly licensed articles are stored.
    """
    path = PurePosixPath(relative_path)
    return path.parent == PurePosixPath("docs/articles") and path.suffix.lower() == ".pdf"


def path_findings(relative_path: str) -> list[str]:
    """
    Flag environment-file paths independently of Git ignore rules.

    Args:
        relative_path (str): Repository-relative POSIX path.

    Returns:
        list[str]: Finding labels, or an empty list for an allowed path.
    """
    # Openly licensed article PDFs are the one allowed exception to the PDF rule.
    if is_public_article(relative_path):
        return []

    # Keep case-insensitive environment-file protection even without ignore rules.
    path = PurePosixPath(relative_path.lower())
    return ["private artifact path"] if path.name.startswith(".env") and path.name != ".env.example" else []


def content_findings(data: bytes) -> list[str]:
    """
    Detect PDF signatures and sensitive patterns in UTF-8 or BOM-marked UTF-16 text.
    Skip other content that cannot be decoded.

    Args:
        data (bytes): File or index blob contents.

    Returns:
        list[str]: Finding labels without the matched content.
    """
    # Detect PDFs by magic bytes so a renamed source document is still caught.
    if b"%PDF-" in data[:1024]:
        return ["PDF content (including renamed source documents)"]

    # Detect byte-order marks before decoding Windows UTF-16 text.
    encoding = "utf-16" if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)) else "utf-8"
    try:
        text = data.decode(encoding)
    except UnicodeDecodeError:
        return []  # Binary additions require manual review; this is not a DLP system.
    return [label for label, pattern in PATTERNS.items() if pattern.search(text)]


def check(root: Path = ROOT) -> list[str]:
    """
    Collect publication findings from repository paths, working files, and staged blobs.

    Args:
        root (Path, optional): Repository directory. Defaults to ROOT.

    Returns:
        list[str]: Findings naming the worktree or index path and issue without quoting matched content.
    """
    # Use Git's ignore rules instead of maintaining duplicate directory and suffix lists.
    relative_paths = public_paths(root)
    ignored = find_ignored_paths(relative_paths, root)
    staged_ignored = find_staged_ignored_paths(relative_paths, root)

    # Scan worktree files that a commit could publish.
    findings = []
    for relative_path in relative_paths:
        path = root / relative_path
        issues = path_findings(relative_path)
        if relative_path in ignored:
            issues.append("ignored artifact path")

        # Never follow a symlink, since its target may lie outside the repository.
        if path.is_symlink():
            issues.append("symlink requires explicit publication review")
        elif path.is_file() and not is_public_article(relative_path):
            issues += content_findings(path.read_bytes())
        findings.extend(f"worktree {relative_path}: {issue}" for issue in issues)

    # Scan index blobs too: a cleaned worktree must not hide an older staged secret.
    entries = git("ls-files", "--stage", "-z", root=root).stdout.split(NUL)
    for entry in filter(None, entries):
        # Each entry is "mode object stage<TAB>path".
        metadata, encoded_path = entry.split(b"\t", 1)
        mode, object_id, stage = metadata.decode("ascii").split()
        relative_path = encoded_path.decode("utf-8")
        issues = path_findings(relative_path)
        if relative_path in ignored or relative_path in staged_ignored:
            issues.append("ignored artifact path")

        if stage != "0":
            issues.append("unresolved merge entry")

        # Symlink and submodule entries have no file content to scan, so they need a human look.
        if mode in {"120000", "160000"}:
            issues.append("symlink/submodule requires explicit publication review")
        elif not is_public_article(relative_path):
            blob = git("cat-file", "blob", object_id, root=root).stdout
            issues += content_findings(blob)
        findings.extend(f"index {relative_path}: {issue}" for issue in issues)
    return findings


def main() -> int:
    """
    Run the publication guard and print its findings or a success message.

    Returns:
        int: 1 when findings exist, otherwise 0.
    """
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
