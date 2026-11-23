# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare saved/live Raon evaluation state without importing Torch or using GPUs.

Accuracy is paired by completed question identity. Throughput is integrated over
the same log time window; it is descriptive when request histories differ.
The tool is read-only except for its explicitly requested output JSON.
"""

import argparse
import datetime as dt
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

LOG = re.compile(
    r"INFO (\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*"
    r"Avg generation throughput: ([\d.]+) tokens/s, Running: (\d+) reqs"
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def completed(root, profile):
    rows = {}
    digest = hashlib.sha256()
    for path in sorted((root / "state" / profile / "mmlu/completed").glob("*.json")):
        raw = path.read_bytes()
        row = json.loads(raw)
        index = row["index"]
        if index in rows or path.stem != f"{index:06d}":
            raise ValueError(f"Duplicate/misnamed question: {path.name}")
        expected = row["prediction"] is not None and row["prediction"] == row["answer"]
        if row["correct"] != expected:
            raise ValueError(f"Inconsistent grading: {path.name}")
        if row["output_tokens"] != len(row["token_ids"]):
            raise ValueError(f"Inconsistent output token count: {path.name}")
        digest.update(path.name.encode() + b"\0" + raw)
        # Never export prompts, generated text, or raw output token IDs.
        rows[index] = {
            k: row[k]
            for k in (
                "question_id",
                "answer",
                "prediction",
                "correct",
                "category",
                "prompt_tokens_sha256",
                "output_tokens",
                "segments",
                "finish_reason",
            )
        }
    if not rows:
        raise ValueError(f"No completed {profile} questions")
    return rows, digest.hexdigest()


def paired(off, on):
    common = sorted(off.keys() & on.keys())
    if not common:
        raise ValueError("No common completed questions")
    for index in common:
        for key in ("question_id", "answer", "category", "prompt_tokens_sha256"):
            if off[index][key] != on[index][key]:
                raise ValueError(f"Question {index} has a different {key}")
    result = {"questions": len(common)}
    for name, rows in (("sd_off", off), ("sd_on", on)):
        correct = sum(rows[i]["correct"] for i in common)
        result[name] = {
            "correct": correct,
            "accuracy_percent": 100 * correct / len(common),
            "mean_output_tokens": statistics.mean(
                rows[i]["output_tokens"] for i in common
            ),
            "unparsed": sum(rows[i]["prediction"] is None for i in common),
        }
    result["accuracy_delta_pp"] = (
        result["sd_on"]["accuracy_percent"] - result["sd_off"]["accuracy_percent"]
    )
    result["both_correct"] = sum(off[i]["correct"] and on[i]["correct"] for i in common)
    result["off_only_correct"] = sum(
        off[i]["correct"] and not on[i]["correct"] for i in common
    )
    result["on_only_correct"] = sum(
        on[i]["correct"] and not off[i]["correct"] for i in common
    )
    result["both_wrong"] = sum(
        not off[i]["correct"] and not on[i]["correct"] for i in common
    )
    result["prediction_agreement_percent"] = (
        100
        * sum(off[i]["prediction"] == on[i]["prediction"] for i in common)
        / len(common)
    )
    result["question_indices"] = common
    return result


def log_points(path, year):
    raw = path.read_bytes()
    points = []
    for match in LOG.finditer(raw.decode("utf-8", errors="replace")):
        stamp = dt.datetime.strptime(f"{year}-{match[1]}", "%Y-%m-%d %H:%M:%S")
        rate = float(match[2])
        if not math.isfinite(rate) or rate < 0:
            raise ValueError("Invalid throughput")
        points.append((stamp.replace(tzinfo=dt.UTC).timestamp(), rate, int(match[3])))
    if len(points) < 2 or any(b[0] <= a[0] for a, b in zip(points, points[1:])):
        raise ValueError(f"Missing/nonmonotonic throughput log: {path.name}")
    return points, hashlib.sha256(raw).hexdigest()


def window_rate(points, begin, end):
    seconds, tokens, rates, running = 0.0, 0.0, [], []
    for previous, current in zip(points, points[1:]):
        width = min(end, current[0]) - max(begin, previous[0])
        if width <= 0:
            continue
        if current[0] - previous[0] > 30:
            raise ValueError("Throughput log has a gap greater than 30 seconds")
        seconds += width
        tokens += current[1] * width
        rates.append(current[1])
        running.append(current[2])
    if not math.isclose(seconds, end - begin, abs_tol=0.01):
        raise ValueError("Incomplete common throughput window")
    return {
        "output_tokens_per_second": tokens / seconds,
        "covered_seconds": seconds,
        "log_samples": len(rates),
        "min_sample_tps": min(rates),
        "max_sample_tps": max(rates),
        "running_requests_range": [min(running), max(running)],
    }


def compare(off_root, on_root, job=None, window=300, year=None):
    started = dt.datetime.now(dt.UTC)
    roots = {"sd_off": Path(off_root), "sd_on": Path(on_root)}
    if window <= 0:
        raise ValueError("Window must be positive")
    if job is None:
        candidates = [
            p.name
            for p in (roots["sd_off"] / "jobs").iterdir()
            if p.name.isdigit()
            and (p / "sd_off/result.json").exists()
            and (roots["sd_on"] / "jobs" / p.name / "sd_on/result.json").exists()
        ]
        if not candidates:
            raise ValueError("No allocation with results from both profiles")
        job = max(candidates, key=int)
    saved = {
        k: read_json(r / "jobs" / job / k / "result.json") for k, r in roots.items()
    }
    for key in (
        "manifest_sha256",
        "target_revision",
        "dataset_revision",
        "context_limit",
        "protocol",
    ):
        if saved["sd_off"][key] != saved["sd_on"][key]:
            raise ValueError(f"Evaluation protocol mismatch: {key}")
    rows, result = (
        {},
        {
            "status": "SNAPSHOT",
            "job": job,
            "started_utc": started.isoformat(),
            "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "manifest_sha256": saved["sd_off"]["manifest_sha256"],
            "profiles": {},
            "limitations": [
                "Live state is read sequentially; counts may advance during collection.",
                "Early completed subsets are not full scores or random samples.",
                "Throughput shares a time window but not requests or cache histories.",
                "Original AR and patched SD differ beyond the SD switch; not an A/B.",
                "Rounded offline-engine log rates are not HTTP latency measurements.",
            ],
        },
    )
    points = {}
    for name, root in roots.items():
        rows[name], state_hash = completed(root, name)
        n = len(rows[name])
        correct = sum(r["correct"] for r in rows[name].values())
        execution = read_json(root / "jobs" / job / "execution.json")
        points[name], log_hash = log_points(
            root / "jobs" / job / f"{name}.log", year or started.year
        )
        result["profiles"][name] = {
            "evaluation_status": saved[name]["status"],
            "completed": n,
            "correct": correct,
            "accuracy_percent": 100 * correct / n,
            "unparsed": sum(r["prediction"] is None for r in rows[name].values()),
            "resumed_questions": sum(
                len(r["segments"]) > 1 for r in rows[name].values()
            ),
            "node": execution["node"],
            "image": execution["image"],
            "completed_state_sha256": state_hash,
            "log_sha256": log_hash,
            "protocol": saved[name]["protocol"],
        }
    result["paired"] = paired(rows["sd_off"], rows["sd_on"])
    uninterrupted = {
        k: {i: r for i, r in v.items() if len(r["segments"]) == 1}
        for k, v in rows.items()
    }
    if uninterrupted["sd_off"].keys() & uninterrupted["sd_on"].keys():
        result["paired_never_resumed"] = paired(
            uninterrupted["sd_off"], uninterrupted["sd_on"]
        )
    end = min(p[-1][0] for p in points.values())
    begin = end - window
    result["throughput"] = {
        "begin_utc": dt.datetime.fromtimestamp(begin, dt.UTC).isoformat(),
        "end_utc": dt.datetime.fromtimestamp(end, dt.UTC).isoformat(),
        **{k: window_rate(p, begin, end) for k, p in points.items()},
    }
    result["throughput"]["sd_on_over_off"] = (
        result["throughput"]["sd_on"]["output_tokens_per_second"]
        / result["throughput"]["sd_off"]["output_tokens_per_second"]
    )
    result["collected_utc"] = dt.datetime.now(dt.UTC).isoformat()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sd-off-root", type=Path, required=True)
    p.add_argument("--sd-on-root", type=Path, required=True)
    p.add_argument("--job", help="Default: latest common allocation")
    p.add_argument("--window-seconds", type=int, default=300)
    p.add_argument("--log-year", type=int)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = compare(a.sd_off_root, a.sd_on_root, a.job, a.window_seconds, a.log_year)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    temp = a.output.with_suffix(".tmp")
    temp.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temp.replace(a.output)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k in ("job", "collected_utc", "throughput")
            }
        )
    )
    for key in ("paired", "paired_never_resumed"):
        if key in result:
            print(
                key,
                json.dumps(
                    {k: v for k, v in result[key].items() if k != "question_indices"}
                ),
            )
    for key, profile in result["profiles"].items():
        print(
            key, profile["completed"], profile["correct"], profile["accuracy_percent"]
        )


if __name__ == "__main__":
    main()
