# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare saved MLA tensor payloads across two runs without loading pickle.

Identity includes inputs, output, FP32 reference, shapes and dtypes. This is a
same-Torch-version regression check, not an acceptance test against FP32 gold.
The original gold verdicts are retained independently.
"""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def payloads(path: Path) -> dict[str, str]:
    result = {}
    with zipfile.ZipFile(path) as archive:
        for entry in archive.namelist():
            relative = entry.split("/", 1)[-1]
            if relative in ("data.pkl", "byteorder") or relative.startswith("data/"):
                result[relative] = hashlib.sha256(archive.read(entry)).hexdigest()
    if "data.pkl" not in result or not any(k.startswith("data/") for k in result):
        raise ValueError(f"Not a PyTorch tensor archive: {path}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text())
    candidate = json.loads(args.candidate.read_text())

    def index(data):
        cases = data["cases"]
        result = {(c["kv_length"], c["query_length"]): c for c in cases}
        if not cases or len(result) != len(cases):
            raise ValueError("Missing or duplicate MLA cases")
        return result

    left, right = index(baseline), index(candidate)
    if left.keys() != right.keys():
        raise ValueError("MLA case sets differ")
    rows = []
    for length, query in sorted(left):
        suffix = f".L{length}.Q{query}.pt"
        a = payloads(Path(str(args.baseline) + suffix))
        b = payloads(Path(str(args.candidate) + suffix))
        rows.append(
            {
                "kv_length": length,
                "query_length": query,
                "identical": a == b,
                "different_entries": sorted(
                    k for k in a.keys() | b.keys() if a.get(k) != b.get(k)
                ),
            }
        )
    result = {
        "status": "IDENTICAL_TENSOR_PAYLOADS"
        if all(r["identical"] for r in rows)
        else "DIFFERENT_TENSOR_PAYLOADS",
        "scope": __doc__,
        "baseline_gold_status": baseline["status"],
        "candidate_gold_status": candidate["status"],
        "cases": rows,
    }
    args.output.write_text(json.dumps(result, indent=2))
    print(result["status"])
    raise SystemExit(0 if all(r["identical"] for r in rows) else 2)


if __name__ == "__main__":
    main()
