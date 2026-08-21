#!/usr/bin/env python3

import argparse
from pathlib import Path
import random
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent

INPUT_FILES = {
    "hepg2": SCRIPT_DIR / "../TACO_MA/data/hepg2/mbo.csv",
    "k562": SCRIPT_DIR / "../TACO_MA/data/k562/mbo.csv",
}

OUTPUT_DIR = SCRIPT_DIR / "results/statistical_baselines"


def gc_content(seq: str) -> float:
    seq = seq.upper()
    return (seq.count("G") + seq.count("C")) / len(seq)


def generate_gc_matched_sequence(length: int, gc: float) -> str:
    bases = ["A", "T", "G", "C"]
    weights = [
        (1 - gc) / 2,
        (1 - gc) / 2,
        gc / 2,
        gc / 2,
    ]
    return "".join(random.choices(bases, weights=weights, k=length))


def write_fasta(headers, sequences, output_path: Path) -> None:
    with open(output_path, "w") as f:
        for header, seq in zip(headers, sequences):
            f.write(f">{header}\n")
            for i in range(0, len(seq), 80):
                f.write(seq[i:i + 80] + "\n")


def process_dataset(cell_type: str, input_path: Path, output_dir: Path = OUTPUT_DIR) -> None:
    df = pd.read_csv(input_path)

    if "sequence" not in df.columns:
        raise ValueError(f"Missing required 'sequence' column in {input_path}")
    id_column = next(
        (
            candidate
            for candidate in ("Unnamed: 0", "id", "ID", "sequence_id", "seq_id")
            if candidate in df.columns
        ),
        None,
    )

    fasta_headers = []
    random_sequences = []
    original_gc = []
    generated_gc = []

    for row_number, (_, row) in enumerate(df.iterrows(), start=1):
        original_id = (
            str(row[id_column])
            if id_column is not None
            else f"{cell_type}_{row_number}"
        )
        seq = str(row["sequence"]).upper()

        gc = gc_content(seq)
        generated_seq = generate_gc_matched_sequence(len(seq), gc)

        fasta_headers.append(f"{original_id}|baseline_B_gc_matched|{cell_type}")
        random_sequences.append(generated_seq)

        original_gc.append(gc)
        generated_gc.append(gc_content(generated_seq))

    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"baseline_B_gc_matched_{cell_type}.fasta"
    write_fasta(fasta_headers, random_sequences, output_path)

    print(f"{cell_type}: saved {len(random_sequences)} sequences to {output_path}")
    print(f"Mean original GC:  {sum(original_gc) / len(original_gc):.4f}")
    print(f"Mean generated GC: {sum(generated_gc) / len(generated_gc):.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate GC-matched random DNA sequences from TACO mbo.csv files."
    )
    parser.add_argument(
        "--taco_data_dir",
        type=Path,
        default=SCRIPT_DIR / "../TACO_MA/data",
        help="Directory containing <cell_type>/mbo.csv.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for generated FASTA files.",
    )
    parser.add_argument(
        "--cell_types",
        nargs="+",
        choices=sorted(INPUT_FILES),
        default=sorted(INPUT_FILES),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    for cell_type in args.cell_types:
        process_dataset(
            cell_type,
            args.taco_data_dir / cell_type / "mbo.csv",
            args.output_dir,
        )


if __name__ == "__main__":
    main()
