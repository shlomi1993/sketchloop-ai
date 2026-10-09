import sys

from pathlib import Path
from urllib.request import urlretrieve


ROOT = Path(__file__).resolve().parents[1]
TARGET_DIR = ROOT / "docs" / "articles" / "local"

# Why a script instead of committing the PDFs: these papers use arXiv's default license, which lets only arXiv
# distribute them. Committing them to this public repository would be redistribution, so each reader downloads them
# from arXiv into the git-ignored docs/articles/local/ folder. Openly licensed papers (for example CC BY, like
# "Rethinking Sketching") are committed directly in docs/articles/ instead. See docs/PRIVACY.md and docs/REFERENCES.md.
ARTICLES = {
    "Latent Diffusion Models.pdf": "https://arxiv.org/pdf/2112.10752",
    "ControlNet.pdf": "https://arxiv.org/pdf/2302.05543",
    "Latent Consistency Models.pdf": "https://arxiv.org/pdf/2310.04378",
    "LCM-LoRA.pdf": "https://arxiv.org/pdf/2311.05556",
}


def main() -> int:
    """
    Download the arXiv papers into the git-ignored docs/articles/local/ folder, skipping files already present.

    Returns:
        int: Process exit code, 0 when every paper is present.
    """
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    failures = 0
    for name, url in ARTICLES.items():
        target = TARGET_DIR / name

        # Keep existing copies so reruns are quick and offline-safe.
        if target.exists():
            print(f"Present: {name}")
            continue

        # Report a failed download and continue with the rest.
        try:
            urlretrieve(url, target)
            print(f"Downloaded: {name}")
        except OSError as error:
            print(f"Failed: {name} ({error}). Check the network and rerun.", file=sys.stderr)
            failures += 1

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
