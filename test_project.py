from __future__ import annotations

import argparse
import os
import py_compile
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"


def print_header(title: str) -> None:
    print()
    print("=" * 80)
    print(f"[TEST] {title}")
    print("=" * 80)


def run_step(title: str, fn) -> None:
    print_header(title)
    try:
        fn()
        print(f"[OK] {title}")
    except Exception as exc:
        print(f"[FAIL] {title}")
        print(exc)
        sys.exit(1)


def run_command(command: list[str]) -> None:
    print("> " + " ".join(command))

    env = os.environ.copy()

    # Ensure imports such as "from config import ..." and "from models..." work.
    pythonpath_items = [str(SRC_DIR)]
    if env.get("PYTHONPATH"):
        pythonpath_items.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_items)

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}")


def check_structure() -> None:
    required_paths = [
        SRC_DIR,
        SRC_DIR / "config.py",
        SRC_DIR / "run.py",
        SRC_DIR / "utils",
        SRC_DIR / "models",
        PROJECT_ROOT / "evaluate_experiments.py",
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")

    print("Repository structure looks correct.")


def compile_python_files() -> None:
    py_files = sorted(PROJECT_ROOT.rglob("*.py"))

    if not py_files:
        raise RuntimeError("No Python files found.")

    for file_path in py_files:
        relative = file_path.relative_to(PROJECT_ROOT)
        print(f"Compiling {relative}")
        py_compile.compile(str(file_path), doraise=True)

    print(f"Compiled {len(py_files)} Python files.")


def import_core_modules() -> None:
    sys.path.insert(0, str(SRC_DIR))

    from config import MODEL_NAME, TRAIN_CSV, TEST_CSV
    from utils.scoring import (
        ranking_from_scores,
        rank_tokens_from_scores,
        minmax_01,
        validate_scores,
        order_to_scores,
    )
    from utils.inference import score_dataframe_with_ranker
    from models.factory import (
        build_ranker,
        build_base_ranker,
        build_named_ranker,
        build_crossencoder_ranker,
        build_crossencoder_rank10_ranker,
        build_ensemble_from_spec,
        parse_weight_spec,
    )

    print("MODEL_NAME:", MODEL_NAME)
    print("TRAIN_CSV:", TRAIN_CSV)
    print("TEST_CSV:", TEST_CSV)
    print("ranking_from_scores:", ranking_from_scores([0.1, 0.5, 0.2]))

    # Avoid unused-variable warnings in editors and make sure imports exist.
    _ = (
        rank_tokens_from_scores,
        minmax_01,
        validate_scores,
        order_to_scores,
        score_dataframe_with_ranker,
        build_ranker,
        build_base_ranker,
        build_named_ranker,
        build_crossencoder_ranker,
        build_crossencoder_rank10_ranker,
        build_ensemble_from_spec,
        parse_weight_spec,
    )

    print("Core imports OK.")


def import_model_modules() -> None:
    sys.path.insert(0, str(SRC_DIR))

    import models.tfidf_ranker
    import models.bm25_ranker
    import models.semantic_ranker
    import models.cross_encoder_ranker
    import models.cross_encoder_rank10_ranker
    import models.cross_encoder_ensemble_ranker
    import models.tail_reranker
    import models.modern_reranker
    import models.vlm_ranker
    import models.llm_ranker

    print("Model imports OK.")


def import_training_scripts() -> None:
    sys.path.insert(0, str(SRC_DIR))

    import training.train_crossencoder
    import training.train_crossencoder_rank10

    print("Training imports OK.")


def search_old_references() -> None:
    patterns = [
        "bert_headtail",
        "bert_rank10",
        'MODEL_NAME = "bert"',
        "evaluate_experiments_vlm_tail",
        "train_bert.py",
        "train_bertin.py",
        "train_mdeberta.py",
        "CHANGE",
        "REMOVE",
    ]

    files = [
        *PROJECT_ROOT.glob("*.py"),
        *SRC_DIR.rglob("*.py"),
    ]

    readme = PROJECT_ROOT / "README.md"
    if readme.exists():
        files.append(readme)

    found_any = False

    for pattern in patterns:
        matches = []

        for file_path in files:
            try:
                text = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = file_path.read_text(encoding="latin-1")

            for line_number, line in enumerate(text.splitlines(), start=1):
                if pattern in line:
                    matches.append((file_path, line_number, line))

        if matches:
            found_any = True
            print()
            print(f"[WARN] Found pattern: {pattern}")
            for file_path, line_number, line in matches:
                relative = file_path.relative_to(PROJECT_ROOT)
                print(f"  {relative}:{line_number}: {line.strip()}")

    if not found_any:
        print("No suspicious old references found.")
    else:
        print()
        print("Warnings are not necessarily fatal.")
        print("Do not remove valid names such as:")
        print("- dccuchile/bert-base-spanish-wwm-cased")
        print("- bertin")
        print("- mdeberta")


def test_evaluate_light() -> None:
    run_command([
        sys.executable,
        "evaluate_experiments.py",
        "--stage",
        "individual",
        "--models",
        "tfidf,bm25",
        "--limit",
        "5",
    ])


def test_evaluate_ensemble() -> None:
    run_command([
        sys.executable,
        "evaluate_experiments.py",
        "--stage",
        "ensembles",
        "--ensemble-specs",
        "beto_headtail:0.40,beto:0.45,mdeberta:0.15",
        "--limit",
        "5",
    ])


def test_evaluate_rerankers() -> None:
    run_command([
        sys.executable,
        "evaluate_experiments.py",
        "--stage",
        "rerankers",
        "--base-ensemble",
        "beto_headtail:0.40,beto:0.45,mdeberta:0.15",
        "--limit",
        "5",
    ])


def test_evaluate_vlm() -> None:
    run_command([
        sys.executable,
        "evaluate_experiments.py",
        "--stage",
        "vlm",
        "--base-ensemble",
        "beto_headtail:0.40,beto:0.45,mdeberta:0.15",
        "--vlm-text-base",
        "tail_rank10",
        "--limit",
        "5",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run compilation, import and smoke tests for the PoliticHeadlinES project."
    )
    parser.add_argument(
        "--heavy",
        action="store_true",
        help="Run checkpoint-based tests such as ensemble and rerankers.",
    )
    parser.add_argument(
        "--vlm",
        action="store_true",
        help="Run VLM test. This also requires checkpoints and images.",
    )
    args = parser.parse_args()

    print("PoliticHeadlinES project test script")
    print("Project root:", PROJECT_ROOT)
    print("Python:", sys.executable)
    print("Heavy tests:", args.heavy)
    print("VLM test:", args.vlm)

    run_step("Repository structure", check_structure)
    run_step("Compile all Python files", compile_python_files)
    run_step("Import core modules", import_core_modules)
    run_step("Import model modules", import_model_modules)
    run_step("Import training scripts", import_training_scripts)
    run_step("Search old references", search_old_references)
    run_step("Lightweight evaluate_experiments test", test_evaluate_light)

    if args.heavy:
        run_step("Ensemble evaluate_experiments test", test_evaluate_ensemble)
        run_step("Reranker evaluate_experiments test", test_evaluate_rerankers)
    else:
        print()
        print("[SKIP] Heavy tests skipped. Use --heavy to run checkpoint-based tests.")

    if args.vlm:
        run_step("VLM evaluate_experiments test", test_evaluate_vlm)
    else:
        print()
        print("[SKIP] VLM test skipped. Use --vlm to run the multimodal test.")

    print()
    print("=" * 80)
    print("All selected tests passed.")
    print("=" * 80)


if __name__ == "__main__":
    main()