#!/usr/bin/env python3
"""Reproduce sequence generation and analyses used in the master thesis.

Run from anywhere:

    python /path/to/repository/master_workflow.py

The default ``all`` workflow creates the three statistical baselines and the
TFBS-guided sequences, analyzes their composition, analyzes existing TACO
results, and performs TFBS rescans when FIMO is installed. External sequences
can be added to composition analyses and a separate TFBS scan with
``--custom-inputs label=path``; the fixed final thesis figures remain unchanged.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

from taco_final_pool import FinalPoolValidationError, load_validated_final_pool


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR if (SCRIPT_DIR / "TACO_MA").is_dir() else SCRIPT_DIR.parent
TACO_DIR = PROJECT_DIR / "TACO_MA"
DEFAULT_TACO_DATA = TACO_DIR / "data"
DEFAULT_TACO_RESULTS = TACO_DIR / "results"
DEFAULT_MOTIFS = (
    TACO_DIR
    / "local_assets/tfbs/human/"
    "20240913075738_JASPAR2024_combined_matrices_1210274_meme.txt"
)
DEFAULT_CELLS = ("hepg2", "k562")
DEFAULT_TACO_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_TACO_ROUND = 100
DEFAULT_TACO_PATTERN = (
    "full_runs/{cell}_seed{seed}_b64_ga4/analysis/"
    "{cell}_seed{seed}_all_candidates_annotated.csv"
)


@dataclass
class StepResult:
    name: str
    status: str
    seconds: float
    command: list[str]
    message: str = ""


class WorkflowError(RuntimeError):
    """A workflow step could not be completed."""


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Master workflow for generating and analyzing real, baseline, "
            "TFBS-guided, and TACO DNA sequences."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "workflow",
        nargs="?",
        choices=("all", "generate", "analyze", "tfbs"),
        default="all",
        help="Part of the workflow to run.",
    )
    parser.add_argument("--cell-types", nargs="+", choices=DEFAULT_CELLS, default=list(DEFAULT_CELLS))
    parser.add_argument(
        "--taco-seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_TACO_SEEDS),
        help="TACO seeds to analyze.",
    )
    parser.add_argument(
        "--taco-round",
        type=int,
        default=DEFAULT_TACO_ROUND,
        help="Effective TACO optimization round used for composition and TFBS analysis.",
    )
    parser.add_argument("--seed", type=int, default=23, help="Seed for baseline and TFBS-guided generation.")
    parser.add_argument("--taco-data-dir", type=Path, default=DEFAULT_TACO_DATA)
    parser.add_argument("--taco-results-dir", type=Path, default=DEFAULT_TACO_RESULTS)
    parser.add_argument("--motif-file", type=Path, default=DEFAULT_MOTIFS)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=SCRIPT_DIR / "results",
        help="Root directory for every generated sequence, analysis, plot, and report.",
    )
    parser.add_argument(
        "--taco-pattern",
        default=DEFAULT_TACO_PATTERN,
        help="TACO result filename pattern; supports {cell} and {seed}.",
    )
    parser.add_argument(
        "--custom-inputs",
        nargs="+",
        default=[],
        metavar="LABEL=PATH",
        help=(
            "Own FASTA/CSV datasets to include in composition analyses and a "
            "separate FIMO scan (not the fixed final thesis figures). CSV files "
            "must contain a 'sequence' column. Labels must be unique."
        ),
    )
    parser.add_argument(
        "--fimo",
        choices=("auto", "yes", "no"),
        default="auto",
        help="Run TFBS rescans: auto runs them only when FIMO is available.",
    )
    parser.add_argument("--fimo-pvalue", type=float, default=1e-4)
    parser.add_argument("--fimo-max-stored-scores", type=int, default=10_000_000)
    parser.add_argument("--max-sequences", type=int, help="Limit sequences per FIMO dataset for testing.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue independent steps after a failed command.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional report path; defaults to OUTPUT_ROOT/master_workflow_report.json.",
    )
    return parser.parse_args(argv)


class Runner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.results: list[StepResult] = []

    def run(
        self,
        name: str,
        command: Sequence[object],
        *,
        cwd: Path = SCRIPT_DIR,
        required_inputs: Iterable[Path] = (),
    ) -> bool:
        cmd = [str(value) for value in command]
        if self.args.dry_run:
            print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
            print("$ " + " ".join(cmd))
            self.results.append(StepResult(name, "dry-run", 0.0, cmd))
            return True

        missing = [path for path in required_inputs if not path.exists()]
        if missing:
            message = "Missing input(s): " + ", ".join(str(path) for path in missing)
            self.results.append(StepResult(name, "failed", 0.0, cmd, message))
            print(f"\n[FAILED] {name}: {message}", file=sys.stderr)
            if not self.args.continue_on_error:
                raise WorkflowError(message)
            return False

        print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
        print("$ " + " ".join(cmd))

        started = time.monotonic()
        completed = subprocess.run(cmd, cwd=cwd, check=False)
        elapsed = time.monotonic() - started
        if completed.returncode:
            message = f"Exit status {completed.returncode}"
            self.results.append(StepResult(name, "failed", elapsed, cmd, message))
            if not self.args.continue_on_error:
                raise WorkflowError(f"{name}: {message}")
            return False
        self.results.append(StepResult(name, "completed", elapsed, cmd))
        return True

    def skip(self, name: str, message: str) -> None:
        print(f"\n[SKIPPED] {name}: {message}")
        self.results.append(StepResult(name, "skipped", 0.0, [], message))


def py(script: str, *args: object) -> list[str]:
    return [sys.executable, str(SCRIPT_DIR / script), *(str(arg) for arg in args)]


def real_csv(args: argparse.Namespace, cell: str) -> Path:
    return args.taco_data_dir.resolve() / cell / "mbo.csv"


def taco_csv(args: argparse.Namespace, cell: str, seed: int) -> Path:
    return args.taco_results_dir.resolve() / args.taco_pattern.format(cell=cell, seed=seed)


def validate_taco_inputs(args: argparse.Namespace) -> None:
    """Require complete B64/GA4 runs and their validated effective-round pools."""
    if len(set(args.taco_seeds)) != len(args.taco_seeds):
        raise WorkflowError("--taco-seeds must not contain duplicate values.")
    if args.dry_run:
        return
    for cell in args.cell_types:
        for seed in args.taco_seeds:
            result = taco_csv(args, cell, seed)
            try:
                pool = load_validated_final_pool(
                    result,
                    expected_cell_line=cell,
                    expected_seed=seed,
                    effective_round=args.taco_round,
                    require_complete_run=True,
                )
            except FinalPoolValidationError as exc:
                raise WorkflowError(str(exc)) from exc
            print(
                f"[TACO {cell} seed {seed}] validated Effective Round "
                f"{args.taco_round}: {len(pool.sequences)} unique sequences, "
                f"physical batches {pool.physical_batches[0]}-{pool.physical_batches[-1]}"
            )


def generate_sequences(runner: Runner) -> None:
    args = runner.args
    baseline_dir = args.output_root / "statistical_baselines"
    common = (
        "--taco_data_dir", args.taco_data_dir.resolve(),
        "--output_dir", baseline_dir,
        "--cell_types", *args.cell_types,
    )
    runner.run(
        "Generate baseline A (uniform)",
        py("baseline_A_uniform_random.py", *common, "--seed", args.seed),
        required_inputs=[real_csv(args, cell) for cell in args.cell_types],
    )
    runner.run(
        "Generate baseline B (GC-matched)",
        py(
            "generate_gc_matched_random.py",
            "--taco_data_dir", args.taco_data_dir.resolve(),
            "--output_dir", baseline_dir,
            "--cell_types", *args.cell_types,
            "--seed", args.seed,
        ),
        required_inputs=[real_csv(args, cell) for cell in args.cell_types],
    )
    runner.run(
        "Generate baseline C (first-order Markov)",
        py("baseline_C_markov_random.py", *common, "--seed", args.seed),
        required_inputs=[real_csv(args, cell) for cell in args.cell_types],
    )
    runner.run(
        "Generate TFBS-guided sequences",
        py(
            "tfbs_guided_generation.py",
            "--taco_data_dir", args.taco_data_dir.resolve(),
            "--meme_file", args.motif_file.resolve(),
            "--output_dir", args.output_root / "tfbs_guided_generation",
            "--cell_types", *args.cell_types,
            "--seed", args.seed,
        ),
        required_inputs=[
            args.motif_file.resolve(),
            *(real_csv(args, cell) for cell in args.cell_types),
        ],
    )


def parse_custom_inputs(values: Sequence[str]) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    labels: set[str] = set()
    for value in values:
        if "=" not in value:
            raise WorkflowError(f"Invalid custom input '{value}'; expected LABEL=PATH.")
        label, raw_path = value.split("=", 1)
        label = label.strip()
        if not label or not re.fullmatch(r"[A-Za-z0-9_.-]+", label):
            raise WorkflowError(
                f"Invalid label '{label}'. Use only letters, numbers, '.', '_' or '-'."
            )
        if label in labels:
            raise WorkflowError(f"Duplicate custom input label: {label}")
        path = Path(raw_path).expanduser().resolve()
        if path.suffix.lower() not in {".fa", ".fasta", ".fna", ".csv"}:
            raise WorkflowError(
                f"Unsupported custom input format for {path}; use FASTA or CSV."
            )
        labels.add(label)
        parsed.append((label, path))
    return parsed


def analyze_composition(runner: Runner) -> None:
    args = runner.args
    custom = args.custom_datasets
    baseline_dir = args.output_root / "statistical_baselines"
    analysis_root = args.output_root / "master_analysis"
    generated = {
        "uniform": "baseline_A_uniform",
        "gc_matched": "baseline_B_gc_matched",
        "markov": "baseline_C_markov",
    }
    for label, prefix in generated.items():
        inputs: list[object] = []
        paths: list[Path] = []
        for cell in args.cell_types:
            paths.extend([real_csv(args, cell), baseline_dir / f"{prefix}_{cell}.fasta"])
            inputs.extend([
                f"real_{cell}={paths[-2]}",
                f"{label}_{cell}={paths[-1]}",
            ])
        paths.extend(path for _, path in custom)
        inputs.extend(f"{name}={path}" for name, path in custom)
        runner.run(
            f"Analyze composition: {label}",
            py(
                "analyze_statistical_baseline_same_heatmap_scale.py",
                "--inputs", *inputs,
                "--output_dir", analysis_root / label,
                "--dinuc_heatmap_scale", "fixed",
                "--dinuc_heatmap_fixed_vmin", 0.0,
                "--dinuc_heatmap_fixed_vmax", 0.08,
            ),
            required_inputs=paths,
        )

    tfbs_paths: list[Path] = []
    tfbs_inputs: list[object] = []
    for cell in args.cell_types:
        tfbs_paths.extend([
            real_csv(args, cell),
            args.output_root / f"tfbs_guided_generation/tfbs_guided_{cell}.fasta",
        ])
        tfbs_inputs.extend([f"real_{cell}={tfbs_paths[-2]}", f"tfbs_{cell}={tfbs_paths[-1]}"])
    tfbs_paths.extend(path for _, path in custom)
    tfbs_inputs.extend(f"{name}={path}" for name, path in custom)
    runner.run(
        "Analyze composition: TFBS-guided",
        py(
            "analyze_sequence_generation_outputs.py",
            "--inputs", *tfbs_inputs,
            "--output_dir", analysis_root / "tfbs_guided",
        ),
        required_inputs=tfbs_paths,
    )


def analyze_taco(runner: Runner) -> None:
    args = runner.args
    custom = args.custom_datasets
    validate_taco_inputs(args)
    for cell in args.cell_types:
        heatmap_vmax = 0.20 if cell == "hepg2" else 0.66
        for seed in args.taco_seeds:
            result = taco_csv(args, cell, seed)
            runner.run(
                f"Analyze TACO: {cell}, seed {seed}, Effective Round {args.taco_round}",
                py(
                    "analyze_taco_generation_outputs.py",
                    "--inputs",
                    f"real_{cell}={real_csv(args, cell)}",
                    f"taco_{cell}_{seed}={result}",
                    *(f"{name}={path}" for name, path in custom),
                    "--output_dir",
                    args.output_root / f"master_analysis/taco/{cell}_seed_{seed}",
                    "--round_value", args.taco_round,
                    "--strict_taco_final_pool",
                    "--dinuc_heatmap_scale", "fixed",
                    "--dinuc_heatmap_fixed_vmin", 0.0,
                    "--dinuc_heatmap_fixed_vmax", heatmap_vmax,
                ),
                required_inputs=[
                    real_csv(args, cell), result,
                    *(path for _, path in custom),
                ],
            )


def create_thesis_specific_composition_plots(runner: Runner) -> None:
    """Create the final TACO heatmaps and detailed composition figures."""
    for cell in runner.args.cell_types:
        cell_label = "HepG2" if cell == "hepg2" else "K562"
        output_dir = runner.args.output_root / f"taco_{cell}_compositional_bias"
        runner.run(
            f"Create {cell_label} TACO six-panel dinucleotide heatmap",
            py(
                "plot_taco_thesis_heatmaps.py",
                "--cell-type", cell,
                "--taco-data-dir", runner.args.taco_data_dir,
                "--taco-results-dir", runner.args.taco_results_dir,
                "--taco-pattern", runner.args.taco_pattern,
                "--taco-seeds", *runner.args.taco_seeds,
                "--taco-round", runner.args.taco_round,
                "--output-dir", runner.args.output_root / "taco_thesis_heatmaps",
            ),
            required_inputs=[
                real_csv(runner.args, cell),
                *(taco_csv(runner.args, cell, seed) for seed in runner.args.taco_seeds),
            ],
        )
        runner.run(
            f"Create {cell_label} TACO compositional-bias figures",
            py(
                "plot_taco_compositional_bias.py",
                "--project-root", PROJECT_DIR,
                "--cell-type", cell,
                "--taco-data-dir", runner.args.taco_data_dir,
                "--taco-results-dir", runner.args.taco_results_dir,
                "--taco-pattern", runner.args.taco_pattern,
                "--taco-seeds", *runner.args.taco_seeds,
                "--taco-round", runner.args.taco_round,
                "--results-dir", output_dir,
                "--figure-dir", output_dir / "figures",
            ),
            required_inputs=[
                real_csv(runner.args, cell),
                *(taco_csv(runner.args, cell, seed) for seed in runner.args.taco_seeds),
            ],
        )


def fimo_enabled(args: argparse.Namespace) -> bool:
    available = shutil.which("fimo") is not None
    if args.dry_run and args.fimo == "yes":
        return True
    if args.fimo == "yes" and not available:
        raise WorkflowError("--fimo yes requested, but the 'fimo' executable is not available.")
    return args.fimo == "yes" or (args.fimo == "auto" and available)


def run_tfbs_scans(runner: Runner) -> None:
    args = runner.args
    if not fimo_enabled(args):
        runner.skip("TFBS/FIMO rescans", "FIMO not found or disabled with --fimo no.")
        return
    validate_taco_inputs(args)
    scan_root = args.output_root / "tfbs_analysis/rescan_maxstored10m"
    baseline_dir = args.output_root / "statistical_baselines"
    for cell in args.cell_types:
        real = (f"real_{cell}", real_csv(args, cell))
        groups: list[tuple[str, list[tuple[str, Path]]]] = [
            (
                "baseline",
                [
                    real,
                    (f"uniform_{cell}", baseline_dir / f"baseline_A_uniform_{cell}.fasta"),
                    (f"gc_matched_{cell}", baseline_dir / f"baseline_B_gc_matched_{cell}.fasta"),
                    (f"markov_{cell}", baseline_dir / f"baseline_C_markov_{cell}.fasta"),
                ],
            ),
            (
                "taco",
                [
                    real,
                    *[
                        (f"taco_{cell}_{seed}", taco_csv(args, cell, seed))
                        for seed in args.taco_seeds
                    ],
                ],
            ),
            (
                "tfbs_guided",
                [
                    real,
                    (
                        f"tfbs_guided_{cell}",
                        args.output_root / f"tfbs_guided_generation/tfbs_guided_{cell}.fasta",
                    ),
                ],
            ),
        ]
        if args.custom_datasets:
            groups.append(("custom", [real, *args.custom_datasets]))

        for group_name, datasets in groups:
            command: list[object] = [
                *py(
                    "scan_selected_tfbs.py",
                    "--cell_line", cell,
                    "--motifs", args.motif_file.resolve(),
                    "--inputs",
                ),
                *(f"{name}={path}" for name, path in datasets),
                "--output_dir", scan_root / f"{cell}_{group_name}_fimo",
                "--pvalue_threshold", args.fimo_pvalue,
                "--max_stored_scores", args.fimo_max_stored_scores,
            ]
            if args.max_sequences:
                command.extend(["--max_sequences", args.max_sequences])
            if group_name == "taco":
                command.extend([
                    "--round_value", args.taco_round,
                    "--strict_taco_final_pool",
                ])
            runner.run(
                f"TFBS/FIMO rescan: {cell} {group_name}",
                command,
                required_inputs=[args.motif_file.resolve(), *(path for _, path in datasets)],
            )

    if args.max_sequences is not None:
        runner.skip(
            "Aggregate and plot TFBS/FIMO results",
            "--max-sequences creates non-final scan populations; final thesis "
            "aggregation requires the complete datasets and 256 TACO sequences per seed.",
        )
        return

    runner.run(
        "Aggregate and plot TFBS/FIMO results",
        py(
            "plot_tfbs_rescan_results.py",
            "--input_dir", scan_root,
            "--output_dir", scan_root,
            "--figure_dir", args.output_root / "tfbs_analysis/figures",
            "--taco-seeds", *args.taco_seeds,
            "--taco-round", args.taco_round,
        ),
        required_inputs=[
            scan_root / f"{cell}_{group}_fimo/tfbs_summary.csv"
            for cell in args.cell_types
            for group in ("baseline", "taco", "tfbs_guided")
        ],
    )


def write_report(runner: Runner, exit_status: int) -> None:
    report = runner.args.report
    report.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "workflow": runner.args.workflow,
        "project_directory": str(PROJECT_DIR),
        "python": sys.version,
        "taco_seeds": runner.args.taco_seeds,
        "taco_round": runner.args.taco_round,
        "exit_status": exit_status,
        "steps": [asdict(result) for result in runner.results],
    }
    report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWorkflow report: {report}")


def main() -> int:
    args = parse_args()
    args.taco_data_dir = args.taco_data_dir.resolve()
    args.taco_results_dir = args.taco_results_dir.resolve()
    args.motif_file = args.motif_file.resolve()
    args.output_root = args.output_root.resolve()
    args.report = (
        args.report.resolve()
        if args.report is not None
        else args.output_root / "master_workflow_report.json"
    )
    args.custom_datasets = []
    runner = Runner(args)
    status = 0
    try:
        args.custom_datasets = parse_custom_inputs(args.custom_inputs)
        if args.workflow in ("all", "generate"):
            generate_sequences(runner)
        if args.workflow in ("all", "analyze"):
            analyze_composition(runner)
            analyze_taco(runner)
            create_thesis_specific_composition_plots(runner)
        if args.workflow in ("all", "tfbs"):
            run_tfbs_scans(runner)
    except (WorkflowError, OSError, ValueError) as exc:
        status = 1
        print(f"\nWORKFLOW FAILED: {exc}", file=sys.stderr)
    finally:
        if not args.dry_run:
            write_report(runner, status)

    completed = sum(result.status == "completed" for result in runner.results)
    skipped = sum(result.status == "skipped" for result in runner.results)
    failed = sum(result.status == "failed" for result in runner.results)
    print(f"\nSummary: {completed} completed, {skipped} skipped, {failed} failed.")
    return status or (1 if failed else 0)


if __name__ == "__main__":
    raise SystemExit(main())
