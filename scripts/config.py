from pathlib import Path
import os

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("PAIN_RESPONSE_DATA_ROOT", PACKAGE_ROOT / "data"))
MOVER_ROOT = Path(os.environ.get("MOVER_ROOT", DATA_ROOT / "MOVER"))
MOVER_DERIVED = Path(os.environ.get("MOVER_DERIVED", DATA_ROOT / "derived" / "MOVER"))
MIMIC_ROOT = Path(os.environ.get("MIMIC_ROOT", DATA_ROOT / "MIMIC-IV" / "mimic-iv-3.1"))
MIMIC_DERIVED = Path(os.environ.get("MIMIC_DERIVED", DATA_ROOT / "derived" / "MIMIC-IV"))
RESULTS_ROOT = Path(os.environ.get("PAIN_RESPONSE_RESULTS", PACKAGE_ROOT / "results"))

for path in (MOVER_DERIVED, MIMIC_DERIVED, RESULTS_ROOT):
    path.mkdir(parents=True, exist_ok=True)
