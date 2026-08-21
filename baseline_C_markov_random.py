#!/usr/bin/env python3

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


DNA_ALPHABET = ["A", "C", "G", "T"]
DEFAULT_CELL_TYPES = ["hepg2", "k562"]


def clean_sequences(sequences):
    cleaned = []

    for seq in sequences:
        seq = str(seq).upper().replace("N", "")
        if seq and set(seq).issubset(set(DNA_ALPHABET)):
            cleaned.append(seq)

    if not cleaned:
        raise ValueError("No valid DNA sequences found.")

    return cleaned


def load_sequences_from_mbo(csv_path, sequence_column="sequence"):
    df = pd.read_csv(csv_path)

    if sequence_column not in df.columns:
        raise ValueError(
            f"Column '{sequence_column}' not found in {csv_path}. "
            f"Available columns: {list(df.columns)}"
        )

    return clean_sequences(df[sequence_column].dropna().astype(str).tolist())


def estimate_initial_probabilities(sequences, pseudocount=1.0):
    counts = Counter({base: pseudocount for base in DNA_ALPHABET})

    for seq in sequences:
        counts[seq[0]] += 1

    total = sum(counts.values())
    return {base: counts[base] / total for base in DNA_ALPHABET}


def estimate_transition_probabilities(sequences, pseudocount=1.0):
    transition_counts = {
        base: Counter({next_base: pseudocount for next_base in DNA_ALPHABET})
        for base in DNA_ALPHABET
    }

    for seq in sequences:
        for i in range(len(seq) - 1):
            current_base = seq[i]
            next_base = seq[i + 1]
            transition_counts[current_base][next_base] += 1

    transition_probabilities = {}

    for base in DNA_ALPHABET:
        total = sum(transition_counts[base].values())
        transition_probabilities[base] = {
            next_base: transition_counts[base][next_base] / total
            for next_base in DNA_ALPHABET
        }

    return transition_probabilities


def weighted_choice(probabilities):
    bases = list(probabilities.keys())
    weights = list(probabilities.values())
    return random.choices(bases, weights=weights, k=1)[0]


def generate_markov_sequence(length, initial_probabilities, transition_probabilities):
    if length <= 0:
        raise ValueError("Sequence length must be greater than zero.")

    sequence = [weighted_choice(initial_probabilities)]

    while len(sequence) < length:
        previous_base = sequence[-1]
        next_base = weighted_choice(transition_probabilities[previous_base])
        sequence.append(next_base)

    return "".join(sequence)


def write_fasta(sequences, output_path, cell_type):
    with open(output_path, "w") as f:
        for i, seq in enumerate(sequences, start=1):
            f.write(f">markov_random_{cell_type}_{i}\n")
            for j in range(0, len(seq), 80):
                f.write(seq[j:j + 80] + "\n")


def print_transition_matrix(transition_probabilities, cell_type):
    print(f"[{cell_type}] First-order transition probabilities:")
    header = "      " + "  ".join(DNA_ALPHABET)
    print(header)

    for base in DNA_ALPHABET:
        values = "  ".join(
            f"{transition_probabilities[base][next_base]:.4f}"
            for next_base in DNA_ALPHABET
        )
        print(f"  {base}: {values}")


def generate_for_cell_type(cell_type, taco_data_dir, output_dir, seed, pseudocount):
    csv_path = taco_data_dir / cell_type / "mbo.csv"
    output_path = output_dir / f"baseline_C_markov_{cell_type}.fasta"

    if not csv_path.exists():
        raise FileNotFoundError(f"Input file not found: {csv_path}")

    random.seed(seed)

    real_sequences = load_sequences_from_mbo(csv_path)
    lengths = [len(seq) for seq in real_sequences]

    initial_probabilities = estimate_initial_probabilities(
        sequences=real_sequences,
        pseudocount=pseudocount
    )
    transition_probabilities = estimate_transition_probabilities(
        sequences=real_sequences,
        pseudocount=pseudocount
    )

    generated_sequences = [
        generate_markov_sequence(
            length=length,
            initial_probabilities=initial_probabilities,
            transition_probabilities=transition_probabilities
        )
        for length in lengths
    ]

    write_fasta(generated_sequences, output_path, cell_type)

    print(f"[{cell_type}] Input: {csv_path}")
    print(f"[{cell_type}] Real sequences: {len(real_sequences)}")
    print(f"[{cell_type}] Generated sequences: {len(generated_sequences)}")
    print(f"[{cell_type}] Lengths: min={min(lengths)}, max={max(lengths)}")
    print(f"[{cell_type}] Output: {output_path}")
    print_transition_matrix(transition_probabilities, cell_type)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate Baseline C Markov-based random DNA sequences directly "
            "from TACO mbo.csv files. The model is a first-order Markov chain "
            "trained separately for each cell type."
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

    parser.add_argument(
        "--pseudocount",
        type=float,
        default=1.0,
        help=(
            "Laplace smoothing pseudocount for initial and transition "
            "probabilities. Default: 1.0"
        )
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
            seed=args.seed,
            pseudocount=args.pseudocount
        )


if __name__ == "__main__":
    main()
