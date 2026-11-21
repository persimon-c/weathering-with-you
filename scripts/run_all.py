#!/usr/bin/env python3
"""Run the end-to-end validation benchmark pipeline and verify reproducible outputs.

This script executes the reproducible evaluation pipeline for Milestone 4:
1. Environment and dependency verification:
   - Verifies Python 3 runtime.
   - Confirms required standard packages (numpy, matplotlib).
   - AST-scans src/ and scripts/ to enforce that no unauthorized third-party ML
     libraries (e.g. scikit-learn, pandas, scipy, torch) are imported.
   - Sets single-thread execution (OMP_NUM_THREADS=1) and fixed seed (PYTHONHASHSEED=0)
     to prevent BLAS/hash non-determinism across platforms.
2. Full unit test suite discovery and execution.
3. Idempotency management:
   - Prunes any existing run records for the current HEAD commit from results/runs.csv
     before appending, ensuring reproducible re-runs without compare.py row collisions.
4. Step-by-step pipeline execution:
   - scripts/run_baselines.py
   - scripts/run_models.py
   - scripts/compare.py --commit <commit>
   - scripts/run_thresholds.py
   - scripts/run_feature_groups.py
   - scripts/run_error_analysis.py
   - scripts/run_per_cluster.py
   - scripts/export_validation_scores.py
   - scripts/run_threshold_tuning.py
   - scripts/plot_results.py
5. Cryptographic checksum validation against results/tables/sha256.txt:
   - Verifies the 7 deterministic, machine-readable validation CSV tables.
   - (Note: results/runs.csv and results/tables/comparison_main.md are excluded from
     checksumming by design because they log run timestamps and the current commit.)

Usage:
    python scripts/run_all.py                # Full reproduction pipeline execution
    python scripts/run_all.py --verify-only  # Verify checked-in tables and run tests only
    python scripts/run_all.py --skip-tests   # Skip unit tests during reproduction
    python scripts/run_all.py --clean        # Purge __pycache__ before executing
"""

import argparse
import ast
import csv
import hashlib
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import unittest

# Ensure reproducible numeric and hashing behavior across platforms
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("PYTHONHASHSEED", "0")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MANIFEST_PATH = ROOT / "results" / "tables" / "sha256.txt"
RUNS_CSV_PATH = ROOT / "results" / "runs.csv"

DISALLOWED_LIBRARIES = {
    "sklearn",
    "scipy",
    "pandas",
    "torch",
    "statsmodels",
    "seaborn",
}

PIPELINE_STEPS = [
    ("1/10 Baselines", ["scripts/run_baselines.py"]),
    ("2/10 Models (NB, LR, KNN)", ["scripts/run_models.py"]),
    ("3/10 Comparison Table", ["scripts/compare.py", "--commit", "{commit}"]),
    ("4/10 Threshold Sweep", ["scripts/run_thresholds.py"]),
    ("5/10 Feature Group Ablation", ["scripts/run_feature_groups.py"]),
    ("6/10 Error Analysis", ["scripts/run_error_analysis.py"]),
    ("7/10 Per-Cluster Evaluation", ["scripts/run_per_cluster.py"]),
    ("8/10 Export Validation Scores", ["scripts/export_validation_scores.py"]),
    ("9/10 Threshold Tuning", ["scripts/run_threshold_tuning.py"]),
    ("10/10 Plot Results", ["scripts/plot_results.py"]),
]


def scan_for_disallowed_imports(root=ROOT):
    """Enforce own-code contract by scanning src/ and scripts/ for unauthorized ML packages."""
    violations = []
    for folder in ("src", "scripts"):
        target_dir = root / folder
        if not target_dir.exists():
            continue
        for py_file in target_dir.rglob("*.py"):
            try:
                tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
            except SyntaxError as e:
                raise RuntimeError(f"Syntax error parsing {py_file}: {e}") from e

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        root_mod = alias.name.split(".")[0]
                        if root_mod in DISALLOWED_LIBRARIES:
                            violations.append((str(py_file.relative_to(root)), root_mod))
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        root_mod = node.module.split(".")[0]
                        if root_mod in DISALLOWED_LIBRARIES:
                            violations.append((str(py_file.relative_to(root)), root_mod))
    return violations


def check_environment(root=ROOT):
    """Verify runtime environment, required dependencies, and own-code library constraints."""
    if sys.version_info[0] < 3:
        raise RuntimeError("Python 3 is required.")

    # Verify presence of authorized dependencies
    missing = []
    for package in ("numpy", "matplotlib"):
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        raise RuntimeError(
            f"Missing required dependencies: {', '.join(missing)}. "
            "Please install requirements.txt."
        )

    # Enforce own-code metric and model integrity
    violations = scan_for_disallowed_imports(root)
    if violations:
        details = "; ".join(f"{file} imports {mod}" for file, mod in violations)
        raise RuntimeError(
            f"Disallowed third-party ML/stats library detected ({details}). "
            "Course protocol mandates own-code algorithms and metrics."
        )


def clean_cache(root=ROOT):
    """Recursively remove __pycache__ directories and .pyc files."""
    cleaned = 0
    for path in root.rglob("__pycache__"):
        if path.is_dir() and ".git" not in path.parts:
            shutil.rmtree(path, ignore_errors=True)
            cleaned += 1
    for pyc in root.rglob("*.pyc"):
        if pyc.is_file() and ".git" not in pyc.parts:
            try:
                pyc.unlink()
                cleaned += 1
            except OSError:
                pass
    return cleaned


def run_unit_tests(root=ROOT):
    """Discover and run all unit tests in tests/."""
    print("=" * 60)
    print("Running unit test suite...")
    print("=" * 60)
    os.environ["RUN_ALL_ACTIVE"] = "1"
    try:
        loader = unittest.defaultTestLoader
        suite = loader.discover(str(root / "tests"), pattern="test_*.py")
        runner = unittest.TextTestRunner(verbosity=1)
        result = runner.run(suite)
        return result.wasSuccessful()
    finally:
        os.environ.pop("RUN_ALL_ACTIVE", None)


def get_git_commit(root=ROOT):
    """Retrieve the current short git commit hash."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def prune_runs_csv(runs_path=RUNS_CSV_PATH, commit=None):
    """Prune any runs from runs.csv matching commit to ensure idempotent re-runs."""
    if not runs_path.exists() or not commit or commit == "unknown":
        return 0

    with runs_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = list(csv.reader(f))

    if not reader:
        return 0

    header = reader[0]
    if "git_commit" not in header:
        return 0

    commit_idx = header.index("git_commit")
    filtered = [header]
    pruned = 0

    for row in reader[1:]:
        if len(row) > commit_idx and row[commit_idx] == commit:
            pruned += 1
        else:
            filtered.append(row)

    if pruned > 0:
        with runs_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerows(filtered)
        print(f"Idempotency reset: pruned {pruned} existing row(s) for commit {commit} from {runs_path.name}")

    return pruned


def compute_sha256(file_path):
    """Compute the SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_checksums(manifest_path=MANIFEST_PATH, root=ROOT):
    """Verify generated tables against the committed checksum manifest."""
    if not manifest_path.exists():
        print(f"Error: Manifest file {manifest_path} not found.", file=sys.stderr)
        return False, ["Manifest file missing"]

    mismatches = []
    print("\n" + "=" * 60)
    print("Verifying table checksums against results/tables/sha256.txt...")
    print("=" * 60)

    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split(None, 1)
            if len(parts) != 2:
                continue

            expected_hash, rel_path = parts[0], parts[1].strip()
            target_path = root / rel_path

            if not target_path.exists():
                print(f"  [MISSING] {rel_path}", file=sys.stderr)
                mismatches.append(f"{rel_path}: file missing")
                continue

            actual_hash = compute_sha256(target_path)
            if actual_hash == expected_hash:
                print(f"  [OK]      {rel_path}")
            else:
                print(
                    f"  [FAILED]  {rel_path}\n"
                    f"            expected: {expected_hash}\n"
                    f"            actual:   {actual_hash}",
                    file=sys.stderr,
                )
                mismatches.append(f"{rel_path}: hash mismatch")

    return len(mismatches) == 0, mismatches


def run_pipeline(root=ROOT, commit=None):
    """Execute all benchmark scripts in order."""
    if not commit:
        commit = get_git_commit(root)

    print("\n" + "=" * 60)
    print(f"Executing validation reproduction pipeline (commit: {commit})...")
    print("=" * 60)

    for step_name, cmd_args in PIPELINE_STEPS:
        formatted_args = [
            arg.format(commit=commit) for arg in cmd_args
        ]
        full_cmd = [sys.executable] + formatted_args
        print(f"\n>>> Running step: {step_name} ({' '.join(formatted_args)})")
        t0 = time.perf_counter()
        result = subprocess.run(full_cmd, cwd=root)
        elapsed = time.perf_counter() - t0

        if result.returncode != 0:
            print(
                f"\nPipeline step failed: {step_name} exited with code {result.returncode}",
                file=sys.stderr,
            )
            return False

        print(f"--- Finished {step_name} in {elapsed:.2f}s ---")

    return True


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run end-to-end reproducible validation pipeline and verify outputs."
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify checked-in artifacts against checksum manifest and run unit tests without re-running the pipeline.",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip unit test suite execution.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove __pycache__ directories and .pyc files before execution.",
    )

    args = parser.parse_args(argv)

    t_start = time.perf_counter()

    if args.clean:
        cleaned = clean_cache(ROOT)
        print(f"Cleaned {cleaned} cache directories / files.")

    try:
        check_environment(ROOT)
    except RuntimeError as err:
        print(f"Environment check failed: {err}", file=sys.stderr)
        return 1

    # Print environment diagnostic details
    try:
        import numpy as np
        import matplotlib as mpl
        print(
            f"Environment: Python {platform.python_version()} on "
            f"{platform.system()} {platform.machine()} | "
            f"NumPy {np.__version__} | Matplotlib {mpl.__version__}"
        )
    except Exception:
        pass

    if not args.skip_tests:
        tests_ok = run_unit_tests(ROOT)
        if not tests_ok:
            print("Unit test suite failed. Aborting reproduction run.", file=sys.stderr)
            return 1
        print("All unit tests passed.")

    if not args.verify_only:
        commit = get_git_commit(ROOT)
        prune_runs_csv(RUNS_CSV_PATH, commit)
        pipeline_ok = run_pipeline(ROOT, commit)
        if not pipeline_ok:
            return 1

    checksums_ok, mismatches = verify_checksums(MANIFEST_PATH, ROOT)
    if not checksums_ok:
        print(
            f"\nChecksum verification failed ({len(mismatches)} mismatch(es)).",
            file=sys.stderr,
        )
        return 1

    total_time = time.perf_counter() - t_start
    print("\n" + "=" * 60)
    if args.verify_only:
        print(f"MANIFEST VERIFIED: All tables match committed SHA-256 hashes ({total_time:.2f}s total)")
    else:
        print(f"REPRODUCTION SUCCESSFUL: All tables reproduced and verified exactly ({total_time:.2f}s total)")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
