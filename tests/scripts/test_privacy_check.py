import codecs
import pytest

from pathlib import Path
from privacy_check import check, content_findings, git


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    """
    Provide an isolated Git repository with the project's ignore rules.
    """
    git("init", root=tmp_path)
    (tmp_path / ".gitignore").write_bytes((Path(__file__).resolve().parents[2] / ".gitignore").read_bytes())
    return tmp_path


def test_renamed_pdf() -> None:
    """
    PDF content must be flagged by its bytes, even under another file name.
    """
    # The guard flags PDF bytes regardless of the file name.
    assert content_findings(b"%P" + b"DF-1.7\nprivate source"), "PDF content must be flagged even when renamed"


# Build each sample from pieces so this file never contains a literal match for the guard to flag.
@pytest.mark.parametrize("value", ["sample" + "@" + "example.invalid", "/Users" + "/fictional-account/file.txt",
                                   "123" + "456" + "789", "sk-" + "x" * 25, "-----BEGIN " + "PRIVATE KEY-----"],
    ids=["email", "home-path", "identifier", "api-key", "private-key"])
def test_synthetic_sensitive_patterns(value: str) -> None:
    """
    Each sensitive pattern must be detected in synthetic text.
    """
    # The guard flags the synthetic sample.
    assert content_findings(value.encode()), "Sensitive pattern must be flagged"


def test_publication_paths(repository: Path) -> None:
    """
    Keep local artifacts private and reject force-staged artifacts while allowing public examples.
    """
    # Create private artifacts and allowed public examples under the real ignore policy.
    private_paths = ["project.PdF", "model.SafeTensors", "model.CKPT", "model.PT", "model.PtH",
                     "private/artifact.txt", ".env.local", "docs/articles/nested/paper.pdf"]
    public_paths = ["docs/PROJECT.md", ".env.example", "docs/articles/Open Paper.pdf"]
    for relative_path in private_paths + public_paths:
        path = repository / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("Synthetic artifact")
    (repository / "docs/articles/Open Paper.pdf").write_bytes(b"%P" + b"DF-1.7\nSynthetic article")

    # Ignored research artifacts stay local while public text and article PDFs pass.
    assert not check(repository), "Local artifacts and allowed public files must pass"
    git("add", "-f", "--", ".", root=repository)
    findings = check(repository)

    # Force-staged private artifacts must be blocked in both worktree and index checks.
    for relative_path in private_paths:
        for location in ("worktree", "index"):
            expected = f"{location} {relative_path}: ignored artifact path"
            assert expected in findings, f"Private artifact escaped publication checks: {relative_path}"

    # The public exceptions must still pass after staging.
    for relative_path in public_paths:
        assert not any(relative_path in finding for finding in findings), f"Public file blocked: {relative_path}"


@pytest.mark.parametrize("folder", ["", "nested"])
def test_staged_ignore_rules_survive_unstaged_edits(repository: Path, folder: str) -> None:
    """
    Detect staged private artifacts even after working ignore rules are removed.
    """
    # Stage a private artifact together with root or nested ignore rules.
    base = repository / folder
    artifact = base / "private/artifact.txt"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("Synthetic artifact")
    ignore_file = base / ".gitignore"
    ignore_file.write_text("/private/\n")
    git("add", "-f", "--", ".", root=repository)

    # Removing the working copy's rule must not hide a file excluded by the staged policy.
    ignore_file.write_text("")
    relative_path = artifact.relative_to(repository).as_posix()
    assert f"index {relative_path}: ignored artifact path" in check(repository), "Staged ignore rules must protect the index"


@pytest.mark.parametrize("encoding,bom", [("utf-8", b""), ("utf-16-le", codecs.BOM_UTF16_LE), ("utf-16-be", codecs.BOM_UTF16_BE)])
def test_staged_secret_survives_clean_worktree(repository: Path, encoding: str, bom: bytes) -> None:
    """
    Detect staged secrets in supported encodings without disclosing them or flagging clean working text.
    """
    # Stage a synthetic credential using each supported text encoding.
    secret = "sk-" + "x" * 25
    notes = repository / "notes.txt"
    notes.write_bytes(bom + f"Synthetic credential: {secret}".encode(encoding))
    git("add", "notes.txt", root=repository)
    assert "worktree notes.txt: possible API credential" in check(repository), "Worktree credentials must be detected"

    # Cleaning the working copy must not hide the secret still staged in the index.
    notes.write_bytes(bom + "Clean working copy".encode(encoding))
    findings = check(repository)
    assert "index notes.txt: possible API credential" in findings, "Staged credentials must be detected"
    assert not any("worktree notes.txt" in finding for finding in findings), "Clean text must not trigger findings"
    assert not any(secret in finding for finding in findings), "Findings must not disclose credentials"
