#!/usr/bin/env python3
"""Create the final six-panel TACO dinucleotide heatmaps used in the thesis."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

from taco_final_pool import load_validated_final_pool


BASES = ("A", "C", "G", "T")
DEFAULT_TACO_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_TACO_ROUND = 100
DEFAULT_TACO_PATTERN = (
    "full_runs/{cell}_seed{seed}_b64_ga4/analysis/"
    "{cell}_seed{seed}_all_candidates_annotated.csv"
)
CELL_VMAX = {"hepg2": 0.20, "k562": 0.66}
ANNOTATION_THRESHOLD_FRACTION = 0.52


def configure_plot_style() -> None:
    """Use the plotting style of the final thesis-ready reference figures."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.transparent": False,
        }
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir if (script_dir / "TACO_MA").is_dir() else script_dir.parent
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--cell-type", choices=tuple(CELL_VMAX), required=True)
    parser.add_argument(
        "--taco-data-dir",
        type=Path,
        default=project_root / "TACO_MA/data",
        help="TACO data root containing CELL/mbo.csv.",
    )
    parser.add_argument(
        "--taco-results-dir",
        type=Path,
        default=project_root / "TACO_MA/results",
        help="Root of the protocol-aligned TACO result package.",
    )
    parser.add_argument(
        "--taco-pattern",
        default=DEFAULT_TACO_PATTERN,
        help="Annotated result path relative to --taco-results-dir.",
    )
    parser.add_argument(
        "--taco-seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_TACO_SEEDS),
        help="TACO seeds shown after the Original panel.",
    )
    parser.add_argument(
        "--taco-round",
        type=int,
        default=DEFAULT_TACO_ROUND,
        help="Effective TACO round forming the final same-policy pool.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for the final PDF and PNG files.",
    )
    return parser.parse_args(argv)


def load_original_sequences(path: Path) -> list[str]:
    frame = pd.read_csv(path, usecols=["sequence"])
    if frame["sequence"].isna().any():
        raise ValueError(f"Missing sequence in {path}")
    sequences = frame["sequence"].astype(str).str.upper().tolist()
    validate_sequences(sequences, path)
    return sequences


def validate_sequences(sequences: Sequence[str], source: Union[Path, str]) -> None:
    if not sequences:
        raise ValueError(f"No sequences found in {source}")
    lengths = {len(sequence) for sequence in sequences}
    if len(lengths) != 1 or 0 in lengths:
        raise ValueError(f"Sequences in {source} must have one positive fixed length")
    invalid = [sequence for sequence in sequences if set(sequence) - set(BASES)]
    if invalid:
        raise ValueError(f"Non-ACGT sequence found in {source}")


def pooled_dinucleotide_matrix(sequences: Sequence[str]) -> np.ndarray:
    """Return pooled adjacent-pair frequencies in A, C, G, T order."""
    validate_sequences(sequences, "sequence collection")
    sequence_length = len(sequences[0])
    if sequence_length < 2:
        raise ValueError("Sequences must contain at least two nucleotides")

    encoded = np.frombuffer("".join(sequences).encode("ascii"), dtype=np.uint8)
    encoded = encoded.reshape(len(sequences), sequence_length)
    left = encoded[:, :-1]
    right = encoded[:, 1:]
    matrix = np.empty((len(BASES), len(BASES)), dtype=float)
    for row, first in enumerate(BASES):
        for column, second in enumerate(BASES):
            per_sequence = (
                (left == ord(first)) & (right == ord(second))
            ).sum(axis=1) / (sequence_length - 1)
            matrix[row, column] = per_sequence.mean()
    return matrix


def load_matrices(args: argparse.Namespace) -> list[tuple[str, np.ndarray]]:
    cell = args.cell_type
    original_path = args.taco_data_dir / cell / "mbo.csv"
    matrices = [("Original", pooled_dinucleotide_matrix(load_original_sequences(original_path)))]

    for seed in args.taco_seeds:
        result_path = args.taco_results_dir / args.taco_pattern.format(cell=cell, seed=seed)
        pool = load_validated_final_pool(
            result_path,
            expected_cell_line=cell,
            expected_seed=seed,
            effective_round=args.taco_round,
        )
        matrices.append((f"seed{seed}", pooled_dinucleotide_matrix(pool.sequences)))
    return matrices


def annotation_color(value: float, vmax: float) -> str:
    """Select contrasting text exactly as in the final thesis figures."""
    return "black" if value > vmax * ANNOTATION_THRESHOLD_FRACTION else "white"


def create_heatmap_figure(
    matrices: Sequence[tuple[str, np.ndarray]], cell_type: str
) -> tuple[plt.Figure, np.ndarray]:
    if cell_type not in CELL_VMAX:
        raise ValueError(f"Unsupported cell type: {cell_type}")
    if not matrices:
        raise ValueError("At least one heatmap matrix is required")
    for title, matrix in matrices:
        if np.shape(matrix) != (4, 4):
            raise ValueError(f"Heatmap matrix for {title} must have shape (4, 4)")

    vmax = CELL_VMAX[cell_type]
    ncols = 3 if len(matrices) > 4 else len(matrices)
    nrows = int(np.ceil(len(matrices) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.0 * ncols, 2.7 * nrows),
        squeeze=False,
    )
    norm = Normalize(vmin=0.0, vmax=vmax)
    image = None
    for index, axis in enumerate(axes.flat):
        if index >= len(matrices):
            axis.axis("off")
            continue
        title, matrix = matrices[index]
        image = axis.imshow(matrix, cmap="viridis", norm=norm, aspect="equal")
        axis.set_xticks(range(4))
        axis.set_xticklabels(BASES)
        axis.set_yticks(range(4))
        axis.set_yticklabels(BASES)
        axis.set_xlabel("Second nucleotide")
        axis.set_ylabel("First nucleotide")
        axis.set_title(f"{chr(65 + index)}  {title}", loc="left")
        for row in range(4):
            for column in range(4):
                value = float(matrix[row, column])
                axis.text(
                    column,
                    row,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    fontsize=7.2,
                    color=annotation_color(value, vmax),
                )

    fig.subplots_adjust(
        left=0.08,
        right=0.88,
        bottom=0.10,
        top=0.94,
        wspace=0.36,
        hspace=0.38,
    )
    if image is not None:
        colorbar_axis = fig.add_axes([0.915, 0.20, 0.018, 0.60])
        colorbar = fig.colorbar(image, cax=colorbar_axis)
        colorbar.set_label("Frequency")
    return fig, axes


def output_stem(output_dir: Path, cell_type: str) -> Path:
    return output_dir / f"taco_{cell_type}_dinucleotide_heatmaps_five_seed"


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if len(set(args.taco_seeds)) != len(args.taco_seeds):
        raise ValueError("--taco-seeds must not contain duplicates")
    configure_plot_style()
    matrices = load_matrices(args)
    figure, _ = create_heatmap_figure(matrices, args.cell_type)
    stem = output_stem(args.output_dir, args.cell_type)
    save_figure(figure, stem)
    print(f"Saved {stem.with_suffix('.pdf')}")
    print(f"Saved {stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()
