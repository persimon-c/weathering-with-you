import csv
import os
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

from scripts.run_all import (
    check_environment,
    clean_cache,
    compute_sha256,
    main,
    prune_runs_csv,
    scan_for_disallowed_imports,
    verify_checksums,
)


class TestRunAll(unittest.TestCase):
    def test_check_environment_passes(self):
        # Current environment has Python 3, numpy, matplotlib, and zero disallowed ML libraries
        check_environment(ROOT)

    def test_scan_for_disallowed_imports_detects_violations(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            src_dir = temp_root / "src"
            src_dir.mkdir()

            clean_file = src_dir / "clean.py"
            clean_file.write_text("import numpy as np\nimport math\n", encoding="utf-8")
            self.assertEqual(scan_for_disallowed_imports(temp_root), [])

            # Inject disallowed library import
            bad_file = src_dir / "bad.py"
            bad_file.write_text("import sklearn.metrics\nfrom pandas import DataFrame\n", encoding="utf-8")
            violations = scan_for_disallowed_imports(temp_root)
            self.assertEqual(len(violations), 2)
            mods = {mod for _, mod in violations}
            self.assertIn("sklearn", mods)
            self.assertIn("pandas", mods)

    def test_compute_sha256(self):
        with tempfile.NamedTemporaryFile("wb", delete=False) as f:
            f.write(b"hello world\n")
            f.flush()
            temp_path = Path(f.name)

        try:
            expected = "a948904f2f0f479b8f8197694b30184b0d2ed1c1cd2a1ec0fb85d299a192a447"
            self.assertEqual(compute_sha256(temp_path), expected)
        finally:
            temp_path.unlink()

    def test_prune_runs_csv_idempotency(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runs_path = Path(temp_dir) / "runs.csv"
            header = ["timestamp_utc", "run_id", "git_commit", "method"]
            rows = [
                ["20261121T000000Z", "run1", "commit_a", "majority"],
                ["20261121T000001Z", "run2", "commit_b", "majority"],
                ["20261121T000002Z", "run3", "commit_a", "rainfall"],
            ]
            with runs_path.open("w", newline="") as f:
                writer = csv.writer(f, lineterminator="\n")
                writer.writerow(header)
                writer.writerows(rows)

            # First prune for commit_a: removes 2 rows
            pruned = prune_runs_csv(runs_path, "commit_a")
            self.assertEqual(pruned, 2)

            with runs_path.open("r") as f:
                reader = list(csv.reader(f))
            self.assertEqual(len(reader), 2)  # header + 1 row
            self.assertEqual(reader[1][2], "commit_b")

            # Second prune for commit_a: 0 rows to prune (idempotent)
            pruned_second = prune_runs_csv(runs_path, "commit_a")
            self.assertEqual(pruned_second, 0)

            # Pruning unknown or empty commit does nothing
            self.assertEqual(prune_runs_csv(runs_path, "unknown"), 0)
            self.assertEqual(prune_runs_csv(runs_path, None), 0)

    def test_verify_checksums_detects_mismatch_and_success(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root_dir = Path(temp_dir)
            target = root_dir / "test.csv"
            target.write_text("sample,content\n", encoding="utf-8")
            target_hash = compute_sha256(target)

            manifest = root_dir / "sha256.txt"
            manifest.write_text(f"{target_hash}  test.csv\n", encoding="utf-8")

            # Matching checksum passes
            ok, mismatches = verify_checksums(manifest, root_dir)
            self.assertTrue(ok)
            self.assertEqual(mismatches, [])

            # Corrupted content fails
            target.write_text("modified,content\n", encoding="utf-8")
            ok, mismatches = verify_checksums(manifest, root_dir)
            self.assertFalse(ok)
            self.assertEqual(len(mismatches), 1)

            # Missing file fails
            target.unlink()
            ok, mismatches = verify_checksums(manifest, root_dir)
            self.assertFalse(ok)
            self.assertEqual(len(mismatches), 1)

    def test_main_verify_only_direct_call(self):
        # Directly invoke main(["--verify-only", "--skip-tests"]) to test verify execution cleanly
        exit_code = main(["--verify-only", "--skip-tests"])
        self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()
