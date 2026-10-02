import pytest
import subprocess

from pathlib import Path

from privacy_check import check, content_findings, path_findings


@pytest.mark.parametrize(
    "name",
    ["project.pdf", "copies/brief.PDF", "runs/run/image.png", ".env.local", "docs/articles/nested/paper.pdf", "docs/paper.pdf"],
)
def test_private_artifact_paths(name: str) -> None:
    """
    Private artifact paths must be flagged by name alone.
    """
    assert path_findings(name), f"{name} must be flagged as private"


@pytest.mark.parametrize("name", ["docs/PROJECT.md", ".env.example", "docs/articles/Open Paper.pdf"])
def test_public_paths(name: str) -> None:
    """
    Publishable paths, including open article PDFs in docs/articles/, must pass.
    """
    assert not path_findings(name), f"{name} must be publishable"


def test_renamed_pdf() -> None:
    """
    PDF content must be flagged by its bytes, even under another file name.
    """
    assert content_findings(b"%P" + b"DF-1.7\nprivate source"), "PDF content must be flagged even when renamed"


# Build each sample from pieces so this file never contains a literal match for the guard to flag.
@pytest.mark.parametrize("value", ["sample" + "@" + "example.invalid", "/Users" + "/fictional-account/file.txt",
                                   "123" + "456" + "789", "sk-" + "x" * 25, "-----BEGIN " + "PRIVATE KEY-----"],
    ids=["email", "home-path", "identifier", "api-key", "private-key"])
def test_synthetic_sensitive_patterns(value: str) -> None:
    """
    Each sensitive pattern must be detected in synthetic text.
    """
    assert content_findings(value.encode()), "Sensitive pattern must be flagged"


def test_public_project_text() -> None:
    """
    Ordinary project text must not trigger a finding.
    """
    assert not content_findings(b"R06: preserve model revision and iteration lineage."), "Plain project text must pass"


def test_index_and_ignored_pdf(tmp_path: Path) -> None:
    """
    The guard must skip ignored files but catch private content staged in the index.
    """
    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    # An ignored PDF stays local, so it must not fail the guard.
    git("init")
    (tmp_path / ".gitignore").write_text("*.pdf\n")
    (tmp_path / "project.pdf").write_bytes(b"%P" + b"DF-1.7\n")
    assert not check(tmp_path), "Ignored private source must not block local work"

    # Force-staging it puts it in the index, where the guard must catch it.
    git("add", "-f", "project.pdf")
    assert any("index project.pdf" in item for item in check(tmp_path)), "Force-staged PDF must be flagged"
    git("rm", "--cached", "project.pdf")

    # Stage a secret, then clean the worktree copy, so only the index blob still holds it.
    secret = "sk-" + "x" * 25
    (tmp_path / "notes.md").write_text(secret)
    git("add", "notes.md")
    (tmp_path / "notes.md").write_text("Clean working copy")

    # The guard must flag the staged blob without echoing the secret.
    results = check(tmp_path)
    assert any("index notes.md" in item for item in results), "Staged secret must be flagged despite a clean worktree"
    assert not any(secret in item for item in results), "Do not print matches"
