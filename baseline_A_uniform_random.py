#!/usr/bin/env python3

import argparse
import random
from pathlib import Path

import pandas as pd


DNA_ALPHABET = ["A", "C", "G", "T"]
DEFAULT_CELL_TYPES = ["hepg2", "k562"]


def generate_uniform_sequence(length):
    return "".join(random.choices(DNA_ALPHABET, k=length))


def write_fasta(sequences, output_path, cell_type):
    with open(output_path, "w") as f:
        for i, seq in enumerate(sequences, start=1):
            f.write(f">uniform_random_{cell_type}_{i}\n")
            f.write(seq + "\n")


def load_sequences_from_mbo(csv_path, sequence_column="sequence"):
    df = pd.read_csv(csv_path)

    if sequence_column not in df.columns:
        raise ValueError(
            f"Column '{sequence_column}' not found in {csv_path}. "
            f"Available columns: {list(df.columns)}"
        )

    sequences = (
        df[sequence_column]
        .dropna()
        .astype(str)
        .str.upper()
        .tolist()
    )

    sequences = [
        seq for seq in sequences
        if set(seq).issubset(set(DNA_ALPHABET))
    ]

    if not sequences:
        raise ValueError(f"No valid DNA sequences found in {csv_path}")

    return sequences


def generate_for_cell_type(cell_type, taco_data_dir, output_dir, seed):
    csv_path = taco_data_dir / cell_type / "mbo.csv"
    output_path = output_dir / f"baseline_A_uniform_{cell_type}.fasta"

    if not csv_path.exists():
        raise FileNotFoundError(f"Input file not found: {csv_path}")

    random.seed(seed)

    real_sequences = load_sequences_from_mbo(csv_path)
    lengths = [len(seq) for seq in real_sequences]

    generated_sequences = [
        generate_uniform_sequence(length)
        for length in lengths
    ]

    write_fasta(generated_sequences, output_path, cell_type)

    print(f"[{cell_type}] Input: {csv_path}")
    print(f"[{cell_type}] Real sequences: {len(real_sequences)}")
    print(f"[{cell_type}] Generated sequences: {len(generated_sequences)}")
    print(f"[{cell_type}] Lengths: min={min(lengths)}, max={max(lengths)}")
    print(f"[{cell_type}] Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate Baseline A uniform random DNA sequences directly "
            "from TACO mbo.csv files."
        )
    )

    parser.add_argument(
        "--taco_data_dir",
        default="../TACO_MA/data",
        help="Path to TACO data directory containing <cell_type>/mbo.csv"
    )

    parser.add_argument(
        "--output_dir",
        default="results/statistical_baselines",
        help="Directory for generated baseline FASTA files"
    )

    parser.add_argument(
        "--cell_types",
        nargs="+",
        default=DEFAULT_CELL_TYPES,
        help="Cell types to process. Default: hepg2 k562"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=23,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    taco_data_dir = Path(args.taco_data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for cell_type in args.cell_types:
        generate_for_cell_type(
            cell_type=cell_type.lower(),
            taco_data_dir=taco_data_dir,
            output_dir=output_dir,
            seed=args.seed
        )


if __name__ == "__main__":
    main()
