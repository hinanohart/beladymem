"""Machine-enforced honesty contract (runs in pytest and in CI).

* The three NON-CLAIMs appear verbatim in the README.
* No hype phrases anywhere in the README or source.
* The deterministic core (oracle) contains no randomness.
* Every file/text open in src declares an explicit utf-8 encoding.
"""

from pathlib import Path

from beladymem.report import NON_CLAIMS

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "beladymem"

# Phrases that would over-sell a measurement instrument. Deliberately multi-word
# so they never false-positive on ordinary prose ("the first reference", "best
# fixed policy", ...).
HYPE = [
    "state-of-the-art",
    "state of the art",
    "world-class",
    "world class",
    "revolutionary",
    "guaranteed best",
    "fully automatic",
    "100% accurate",
    "solves reproducibility",
    "blazing fast",
    "industry-leading",
    "unprecedented",
]


def test_non_claims_verbatim_in_readme():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for nc in NON_CLAIMS:
        assert nc in readme, f"NON-CLAIM missing from README: {nc!r}"


def test_no_hype_in_readme_and_source():
    texts = [(ROOT / "README.md").read_text(encoding="utf-8").lower()]
    for py in SRC.rglob("*.py"):
        texts.append(py.read_text(encoding="utf-8").lower())
    blob = "\n".join(texts)
    found = [h for h in HYPE if h in blob]
    assert not found, f"hype phrases present: {found}"


def test_deterministic_core_has_no_randomness():
    # The oracle and the point-estimate path must be deterministic. (Bootstrap
    # CIs and robustness corruptions legitimately use a seeded RNG in metrics.py.)
    oracle = (SRC / "oracle.py").read_text(encoding="utf-8")
    assert "random" not in oracle
    assert "np.random" not in oracle


def test_src_text_io_declares_utf8():
    offenders = []
    for py in SRC.rglob("*.py"):
        for lineno, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
            for call in (".open(", "open(", ".read_text(", ".write_text("):
                if call in line and "encoding" not in line and "def " not in line:
                    # allow set.open-like false hits only when an io call is present
                    if "open(" in line or "_text(" in line:
                        offenders.append(f"{py.name}:{lineno}: {line.strip()}")
                    break
    assert not offenders, "text IO without explicit encoding:\n" + "\n".join(offenders)
