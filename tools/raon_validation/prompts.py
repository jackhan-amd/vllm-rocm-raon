# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rebuild a fixed five-shot cohort from pinned local MMLU-Pro data."""

import argparse
import hashlib
import json
from pathlib import Path

from .common import write_json


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def example(row, answered):
    options = [x for x in row["options"] if x != "N/A"]
    text = "Question:\n" + row["question"] + "\nOptions:\n"
    text += "".join(f"{chr(65 + i)}. {option}\n" for i, option in enumerate(options))
    if answered:
        text += (
            row["cot_content"].replace(
                "A: Let's think step by step.", "Answer: Let's think step by step."
            )
            + "\n\n"
        )
    else:
        text += "Answer: Let's think step by step."
    return text


def main():
    import pyarrow.parquet as pq

    from vllm.tokenizers import get_tokenizer

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--cot-template", required=True)
    p.add_argument("--indices", type=int, nargs="+", required=True)
    p.add_argument("--reference-manifest")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    tokenizer = get_tokenizer(a.tokenizer, trust_remote_code=False)
    data = Path(a.data_dir)
    tests = pq.read_table(data / "test-00000-of-00001.parquet").to_pylist()
    validation = pq.read_table(data / "validation-00000-of-00001.parquet").to_pylist()
    initial = Path(a.cot_template).read_text()
    subjects = sorted({r["category"] for r in tests})
    groups = {
        s: sorted(
            [r for r in tests if r["category"] == s],
            key=lambda r: digest(str(r["question_id"])),
        )
        for s in subjects
    }
    ordered = [
        groups[s][i]
        for i in range(max(map(len, groups.values())))
        for s in subjects
        if i < len(groups[s])
    ]
    reference = (
        json.loads(Path(a.reference_manifest).read_text())
        if a.reference_manifest
        else None
    )
    rows = []
    for i in a.indices:
        q = ordered[i]
        shots = [r for r in validation if r["category"] == q["category"]][:5]
        assert len(shots) == 5
        text = initial.replace("{$}", q["category"]) + "\n"
        text += "".join(example(r, True) for r in shots) + example(q, False)
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=True,
            truncate_history_thinking=False,
            return_dict=False,
        )
        if reference:
            assert reference[i]["prompt_tokens_sha256"] == digest(json.dumps(ids)), i
        rows.append(
            {"id": str(i), "prompt_token_ids": ids, "prompt_sha256": digest(text)}
        )
    write_json(a.output, rows)


if __name__ == "__main__":
    main()
