#!/usr/bin/env python3
"""Score saved model outputs against clinician-authored required/forbidden phrases."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True, help="JSONL with required and forbidden phrase lists")
    parser.add_argument("--predictions", required=True, help="JSONL objects with id and output")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    def read_jsonl(path: str):
        return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]

    cases = {item["id"]: item for item in read_jsonl(args.cases)}
    predictions = {item["id"]: str(item.get("output") or "") for item in read_jsonl(args.predictions)}
    rows = []
    totals = Counter()
    for case_id, case in cases.items():
        output = predictions.get(case_id, "")
        missing = [phrase for phrase in case.get("required", []) if phrase not in output]
        violations = [phrase for phrase in case.get("forbidden", []) if phrase in output]
        passed = bool(output) and not missing and not violations
        rows.append({"id": case_id, "category": case.get("category"), "passed": passed, "missing": missing, "violations": violations})
        totals["passed" if passed else "failed"] += 1
    report = {"summary": dict(totals), "pass_rate": totals["passed"] / max(1, len(cases)), "cases": rows}
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False))
    if totals["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

