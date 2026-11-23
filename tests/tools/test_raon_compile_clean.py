# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reject changed code caches even when file size and timestamps are retained."""

import os
import tempfile
import unittest
from pathlib import Path

from tools.raon_validation.compile_clean import cache_changes, cache_snapshot


class TestCompileCacheAudit(unittest.TestCase):
    def test_rewritten_binary_is_detected_with_preserved_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "kernel.hsaco"
            binary.write_bytes(b"old")
            stat = binary.stat()
            before = cache_snapshot([root])
            binary.write_bytes(b"new")
            os.utime(binary, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            self.assertEqual(
                cache_changes(before, cache_snapshot([root])), [str(binary)]
            )

    def test_additions_removals_and_noncode_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / "old.so"
            old.write_bytes(b"code")
            before = cache_snapshot([root])
            (root / "run.log").write_text("compile log")
            self.assertEqual(cache_changes(before, cache_snapshot([root])), [])
            old.unlink()
            new = root / "new.pkl"
            new.write_bytes(b"code")
            self.assertEqual(
                cache_changes(before, cache_snapshot([root])),
                sorted([str(old), str(new)]),
            )


if __name__ == "__main__":
    unittest.main()
