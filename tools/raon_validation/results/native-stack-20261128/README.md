# Native stack evidence

`speed30.json` contains the final-model thirty-cell speed matrix. The fresh
full MMLU-Pro result is `rda-full-mmlu.json`; the paired comparison is
`paired-quality-analysis.json`. The original frozen baseline is unchanged.

`release/` and `session/` preserve raw receipts with short local filenames for
Windows compatibility. Each directory's `manifest.json` maps those names back
to the original remote paths and records exact byte hashes. Evidence files
disable Git line-ending conversion so their hashes remain reproducible.

The source fixture has one trailing blank line in the tested RDA kernel.
It is intentionally preserved to retain the exact deployed source hash;
whitespace checking excludes only `blank-at-eof` for this archival commit.
Raw JSON receipts retain CRLF where originally generated; whitespace checking
recognizes `cr-at-eol` for these byte-preserved artifacts.

The local pre-commit entrypoint failed before running hooks because its cache
database under the user profile was read-only. This archival branch preserves
the tested sources without auto-formatting. Source AST checks, staged-source
hash verification, whitespace checks and the recorded GPU/model validation
were completed separately. This is not a claim of a passing pre-commit suite
or an upstream-ready PR.

The image was built before the full evaluation finished. Its immutable source
matches the evaluated candidate. Dense layout and later mHC hi-only candidates
are separate experiments and are excluded from the image and speed30 matrix.
