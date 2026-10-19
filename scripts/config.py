"""Path configuration.

Set environment variables to point at your local copies of the derived
MOVER / MIMIC-IV tables and at an output directory. Defaults assume this
repository is placed next to a `data/` folder:

    <repo>/data/MOVER/derived/...
    <repo>/data/MIMIC-IV/derived/...
    <repo>/results/...

No patient-level data are redistributed with this repository.
"""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]

MOVER_DERIVED = Path(os.environ.get("MOVER_DERIVED", ROOT / "data" / "MOVER" / "derived"))
MIMIC_DERIVED = Path(os.environ.get("MIMIC_DERIVED", ROOT / "data" / "MIMIC-IV" / "derived"))
RESULTS_ROOT = Path(os.environ.get("PAIN_RESPONSE_RESULTS", ROOT / "results"))

for p in (RESULTS_ROOT,):
    p.mkdir(parents=True, exist_ok=True)
