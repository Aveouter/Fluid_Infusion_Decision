"""Add completeness/interval artifacts to the V7 verification manifest."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs/experiments/old-event-v7-20261021"
FILES = ("data_completeness_audit.json", "data_completeness_audit.md",
         "interval_missingness.json", "interval_missingness.md",
         "replay_verification.json")
PROTOCOLS = ("docs/old_event_v7_history_protocol_20261021.json",
             "docs/old_event_v7_missing_protocol_20261021.json",
             "docs/old_event_v7_values_protocol_20261021.json")
CODE = ("utils/old_fluid_data_v1.py",
        "reports/audit_old_data_completeness_20261021.py",
        "reports/evaluate_completeness_missingness_20261021.py",
        "reports/report_completeness_missingness_20261021.py",
        "reports/train_old_event_physiology_controls_20261021.py",
        "tests/test_old_data_completeness.py")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    path = REPORT / "verification.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    for name in FILES:
        path_name = Path(name)
        payload[f"{path_name.stem}_{path_name.suffix.lstrip('.')}_sha256"] = sha(REPORT / name)
    payload["completeness_code_sha256"] = {name: sha(ROOT / name) for name in CODE}
    payload["control_protocol_sha256"] = {name: sha(ROOT / name) for name in PROTOCOLS}
    payload["completeness_note"] = (
        "Aggregate-only completeness/interval audit. Raw patient rows and "
        "patient-level predictions remain local; no patient identifiers are published."
    )
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
