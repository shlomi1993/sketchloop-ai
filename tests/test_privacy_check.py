import pytest
import subprocess

from pathlib import Path

from privacy_check import check, content_findings, path_findings


@pytest.mark.parametrize(
    "name",
    ["project.pdf", "copies/brief.PDF", "runs/run/image.png", ".env.local"],
)
def test_private_artifact_paths(name: str) -> None:
    assert path_findings(name)


@pytest.mark.parametrize("name", ["docs/PROJECT.md", ".env.example"])
def test_public_paths(name: str) -> None:
    assert not path_findings(name)


def test_renamed_pdf() -> None:
    assert content_findings(b"%P" + b"DF-1.7\nprivate source")


@pytest.mark.parametrize("value", ["sample" + "@" + "example.invalid", "/Users" + "/fictional-account/file.txt",
                                   "123" + "456" + "789", "sk-" + "x" * 25, "-----BEGIN " + "PRIVATE KEY-----"],
    ids=["email", "home-path", "identifier", "api-key", "private-key"])
def test_synthetic_sensitive_patterns(value: str) -> None:
    assert content_findings(value.encode())


def test_public_project_text() -> None:
    assert not content_findings(b"R06: preserve model revision and iteration lineage.")


def test_index_and_ignored_pdf(tmp_path: Path) -> None:
    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init")
    (tmp_path / ".gitignore").write_text("*.pdf\n")
    (tmp_path / "project.pdf").write_bytes(b"%P" + b"DF-1.7\n")
    assert not check(tmp_path), "Ignored private source must not block local work"
    git("add", "-f", "project.pdf")
    assert any("index project.pdf" in item for item in check(tmp_path))
    git("rm", "--cached", "project.pdf")
    secret = "sk-" + "x" * 25
    (tmp_path / "notes.md").write_text(secret)
    git("add", "notes.md")
    (tmp_path / "notes.md").write_text("Clean working copy")
    results = check(tmp_path)
    assert any("index notes.md" in item for item in results)
    assert not any(secret in item for item in results), "Do not print matches"
