#!/usr/bin/env python3
"""
Scan original and generated DNA sequences for selected cell-line-specific TFBS motifs
using FIMO from the MEME Suite.

The script:
  - reads a local JASPAR2024 MEME motif file
  - extracts exactly the predefined motif matrix IDs for HepG2 or K562
  - converts CSV/TSV/FASTA input files to FASTA if needed
  - runs FIMO for each input dataset
  - summarizes motif hits per dataset, sequence, and motif

Selected motifs:
  HepG2:
    CEBPA  -> MA0102.5
    FOXA2  -> MA0047.4
    HNF4A  -> MA1494.2

  K562:
    GATA1        -> MA0035.5
    GATA1::TAL1  -> MA0140.3
    KLF1         -> MA0493.3
"""

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from taco_final_pool import load_validated_final_pool


CELL_LINE_MOTIFS = {
    "hepg2": {
        "CEBPA": "MA0102.5",
        "FOXA2": "MA0047.4",
        "HNF4A": "MA1494.2",
    },
    "k562": {
        "GATA1": "MA0035.5",
        "GATA1::TAL1": "MA0140.3",
        "KLF1": "MA0493.3",
    },
}
EXPECTED_FIMO_VERSION = "5.5.9"
TACO_DATASET_PATTERN = re.compile(r"^taco_(hepg2|k562)_(\d+)$", re.IGNORECASE)


def check_fimo_available():
    if shutil.which("fimo") is None:
        raise RuntimeError(
            "FIMO was not found in PATH. Install MEME Suite or activate the environment "
            "where FIMO is available, then test with: fimo --version"
        )
    try:
        completed = subprocess.run(
            ["fimo", "--version"],
            check=False,
            capture_output=True,
            text=True,
        )
        version_text = (completed.stdout or completed.stderr).strip()
    except OSError as exc:
        raise RuntimeError(f"Could not execute 'fimo --version': {exc}") from exc
    if EXPECTED_FIMO_VERSION not in version_text:
        print(
            "WARNING: Final thesis results were generated with FIMO "
            f"{EXPECTED_FIMO_VERSION}, but this executable reports "
            f"{version_text or 'an unknown version'}.",
            file=sys.stderr,
        )
    return version_text


def get_matrix_id_from_motif_line(line: str) -> str:
    """
    Example:
      MOTIF MA0102.5 CEBPA -> MA0102.5
    """
    parts = line.strip().split()
    if len(parts) < 2 or parts[0] != "MOTIF":
        raise ValueError(f"Invalid MOTIF line: {line}")
    return parts[1]


def extract_selected_motifs_from_meme(input_meme: str, output_meme: str, selected_ids: set):
    """
    Extract selected MOTIF blocks from a MEME file while preserving the MEME header.
    """
    with open(input_meme, "r", encoding="utf-8") as handle:
        lines = handle.readlines()

    motif_start_indices = [
        i for i, line in enumerate(lines)
        if line.startswith("MOTIF ")
    ]

    if not motif_start_indices:
        raise ValueError(f"No MOTIF entries found in MEME file: {input_meme}")

    first_motif_index = motif_start_indices[0]
    header_lines = lines[:first_motif_index]

    selected_blocks = []
    found_ids = set()
    selected_raw_names = {}

    for idx, start in enumerate(motif_start_indices):
        end = motif_start_indices[idx + 1] if idx + 1 < len(motif_start_indices) else len(lines)
        block = lines[start:end]
        motif_line = block[0]
        matrix_id = get_matrix_id_from_motif_line(motif_line)

        if matrix_id in selected_ids:
            selected_blocks.extend(block)
            found_ids.add(matrix_id)
            selected_raw_names[matrix_id] = motif_line.strip().replace("MOTIF ", "")

    missing = selected_ids - found_ids
    if missing:
        available_ids = [
            get_matrix_id_from_motif_line(lines[i])
            for i in motif_start_indices
        ]
        raise ValueError(
            "Missing selected motif matrix IDs in MEME file: "
            + ", ".join(sorted(missing))
            + "\nAvailable matrix IDs include:\n"
            + ", ".join(available_ids[:100])
            + ("\n..." if len(available_ids) > 100 else "")
        )

    with open(output_meme, "w", encoding="utf-8") as handle:
        handle.writelines(header_lines)
        if not header_lines[-1].endswith("\n"):
            handle.write("\n")
        handle.write("\n")
        handle.writelines(selected_blocks)

    return selected_raw_names


def detect_sequence_column(fieldnames):
    candidates = [
        "sequence",
        "seq",
        "dna_sequence",
        "enhancer_sequence",
        "Sequence",
        "SEQ",
        "mbo",
        "MBO",
    ]

    for candidate in candidates:
        if candidate in fieldnames:
            return candidate

    for fieldname in fieldnames:
        lowered = fieldname.lower()
        if "sequence" in lowered or lowered == "seq":
            return fieldname

    raise ValueError(
        "Could not automatically detect sequence column. "
        f"Available columns: {', '.join(fieldnames)}. "
        "Use --sequence_column to specify it."
    )


def read_fasta(path: str, max_sequences=None):
    sequences = []
    current_id = None
    current_chunks = []

    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()

            if not stripped:
                continue

            if stripped.startswith(">"):
                if current_id is not None:
                    sequences.append((current_id, "".join(current_chunks).upper()))
                    if max_sequences is not None and len(sequences) >= max_sequences:
                        return sequences

                current_id = stripped[1:].split()[0]
                current_chunks = []
            else:
                current_chunks.append(stripped)

        if current_id is not None:
            sequences.append((current_id, "".join(current_chunks).upper()))

    if max_sequences is not None:
        sequences = sequences[:max_sequences]

    return sequences


def read_delimited_sequences(
    path: str,
    delimiter=",",
    sequence_column=None,
    max_sequences=None,
):
    sequences = []

    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)

        if reader.fieldnames is None:
            raise ValueError(f"Input file has no header: {path}")

        if sequence_column is None:
            sequence_column = detect_sequence_column(reader.fieldnames)

        id_column = None
        for candidate in ["id", "ID", "name", "Name", "sequence_id", "seq_id"]:
            if candidate in reader.fieldnames:
                id_column = candidate
                break

        for row_index, row in enumerate(reader):
            sequence = row[sequence_column].strip().upper()
            sequence_id = row[id_column].strip() if id_column else f"seq_{row_index}"

            sequences.append((sequence_id, sequence))

            if max_sequences is not None and len(sequences) >= max_sequences:
                break

    return sequences


def read_sequences(path: str, sequence_column=None, max_sequences=None):
    suffix = Path(path).suffix.lower()

    if suffix in [".fa", ".fasta", ".fna"]:
        return read_fasta(path, max_sequences=max_sequences)

    if suffix == ".csv":
        return read_delimited_sequences(
            path,
            delimiter=",",
            sequence_column=sequence_column,
            max_sequences=max_sequences,
        )

    if suffix == ".tsv":
        return read_delimited_sequences(
            path,
            delimiter="\t",
            sequence_column=sequence_column,
            max_sequences=max_sequences,
        )

    raise ValueError(
        f"Unsupported input format: {path}. "
        "Supported formats: .fasta, .fa, .fna, .csv, .tsv"
    )


def read_validated_taco_sequences(
    path: str,
    dataset_label: str,
    round_value: int,
    max_sequences=None,
):
    """Read a TACO pool through the validator shared with composition."""
    match = TACO_DATASET_PATTERN.fullmatch(dataset_label)
    if match is None:
        raise ValueError(
            "Strict TACO final-pool validation requires a dataset label such as "
            "taco_hepg2_0 or taco_k562_4."
        )
    cell_line, raw_seed = match.groups()
    pool = load_validated_final_pool(
        path,
        expected_cell_line=cell_line.lower(),
        expected_seed=int(raw_seed),
        effective_round=round_value,
    )
    sequences = [
        (f"seq_{index}", sequence)
        for index, sequence in enumerate(pool.sequences, start=1)
    ]
    return sequences if max_sequences is None else sequences[:max_sequences]


def write_fasta(sequences, output_path: str):
    with open(output_path, "w", encoding="utf-8") as handle:
        for sequence_id, sequence in sequences:
            clean_id = re.sub(r"\s+", "_", sequence_id)
            handle.write(f">{clean_id}\n")
            handle.write(f"{sequence}\n")


def parse_labeled_inputs(input_arguments):
    labeled_inputs = []

    for argument in input_arguments:
        if "=" not in argument:
            raise ValueError(
                f"Invalid input format: {argument}. "
                "Use label=path, for example real_hepg2=data.csv"
            )

        label, path = argument.split("=", 1)

        if not label:
            raise ValueError(f"Missing label in input argument: {argument}")

        if not path:
            raise ValueError(f"Missing path in input argument: {argument}")

        labeled_inputs.append((label, path))

    return labeled_inputs


def run_fimo(
    selected_meme: str,
    fasta_path: str,
    output_dir: str,
    pvalue_threshold: float,
    max_stored_scores: int,
):
    os.makedirs(output_dir, exist_ok=True)

    command = [
        "fimo",
        "--oc",
        output_dir,
        "--thresh",
        str(pvalue_threshold),
        "--max-stored-scores",
        str(max_stored_scores),
        selected_meme,
        fasta_path,
    ]

    subprocess.run(command, check=True)


def parse_fimo_tsv(fimo_tsv_path: str, dataset_label: str, matrix_id_to_label: dict):
    hits = []

    if not os.path.exists(fimo_tsv_path):
        return hits

    with open(fimo_tsv_path, "r", encoding="utf-8") as handle:
        non_comment_lines = [
            line for line in handle
            if line.strip() and not line.startswith("#")
        ]

    if not non_comment_lines:
        return hits

    reader = csv.DictReader(non_comment_lines, delimiter="\t")

    for row in reader:
        matrix_id = row.get("motif_id", "")
        motif_label = matrix_id_to_label.get(matrix_id, row.get("motif_alt_id", matrix_id))

        sequence_id = row.get("sequence_name", "")

        hits.append({
            "dataset": dataset_label,
            "sequence_id": sequence_id,
            "motif": motif_label,
            "matrix_id": matrix_id,
            "motif_alt_id": row.get("motif_alt_id", ""),
            "start_1_based": row.get("start", ""),
            "stop_1_based": row.get("stop", ""),
            "strand": row.get("strand", ""),
            "score": row.get("score", ""),
            "p_value": row.get("p-value", ""),
            "q_value": row.get("q-value", ""),
            "matched_sequence": row.get("matched_sequence", ""),
        })

    return hits


def write_csv(path: str, rows: list, fieldnames: list):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Scan DNA sequences for selected cell-line-specific TFBS motifs "
            "using FIMO and local JASPAR2024 MEME motifs."
        )
    )

    parser.add_argument(
        "--cell_line",
        required=True,
        choices=sorted(CELL_LINE_MOTIFS.keys()),
        help="Cell line whose selected TFs should be scanned: hepg2 or k562.",
    )

    parser.add_argument(
        "--motifs",
        required=True,
        help="Path to local JASPAR2024 MEME-format motif file.",
    )

    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help=(
            "Input sequence datasets in label=path format. "
            "Supported formats: FASTA, CSV, TSV."
        ),
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where outputs will be written.",
    )

    parser.add_argument(
        "--pvalue_threshold",
        type=float,
        default=1e-4,
        help="FIMO p-value threshold. Default: 1e-4.",
    )

    parser.add_argument(
        "--max_stored_scores",
        type=int,
        default=10000000,
        help="FIMO max stored scores. Default: 10000000.",
    )

    parser.add_argument(
        "--sequence_column",
        default=None,
        help=(
            "Optional sequence column name for CSV/TSV inputs. "
            "If omitted, the script tries to infer it."
        ),
    )

    parser.add_argument(
        "--max_sequences",
        type=int,
        default=None,
        help="Optional maximum number of sequences per dataset for testing.",
    )

    parser.add_argument(
        "--round_value",
        type=int,
        default=None,
        help=(
            "Effective round required for protocol-aligned TACO inputs."
        ),
    )

    parser.add_argument(
        "--strict_taco_final_pool",
        action="store_true",
        help=(
            "Validate TACO-labeled CSV inputs as 256-sequence B64/GA4 "
            "same-policy pools before FASTA conversion."
        ),
    )

    args = parser.parse_args()

    check_fimo_available()

    os.makedirs(args.output_dir, exist_ok=True)

    prepared_dir = os.path.join(args.output_dir, "prepared_fastas")
    fimo_runs_dir = os.path.join(args.output_dir, "fimo_runs")
    os.makedirs(prepared_dir, exist_ok=True)
    os.makedirs(fimo_runs_dir, exist_ok=True)

    selected_meme_path = os.path.join(args.output_dir, f"selected_motifs_{args.cell_line}.meme")

    motif_label_to_matrix_id = CELL_LINE_MOTIFS[args.cell_line]
    matrix_id_to_label = {
        matrix_id: motif_label
        for motif_label, matrix_id in motif_label_to_matrix_id.items()
    }

    selected_ids = set(motif_label_to_matrix_id.values())

    print(f"Cell line: {args.cell_line}")
    print("Selected motifs:")
    for motif_label, matrix_id in motif_label_to_matrix_id.items():
        print(f"  {motif_label}: {matrix_id}")
    print(f"FIMO p-value threshold: {args.pvalue_threshold}")

    selected_raw_names = extract_selected_motifs_from_meme(
        input_meme=args.motifs,
        output_meme=selected_meme_path,
        selected_ids=selected_ids,
    )

    labeled_inputs = parse_labeled_inputs(args.inputs)

    all_hits = []
    dataset_sequence_ids = {}

    for dataset_label, input_path in labeled_inputs:
        print(f"[{dataset_label}] reading sequences from {input_path}", flush=True)

        if dataset_label.lower().startswith("taco_"):
            if not args.strict_taco_final_pool:
                raise ValueError(
                    "TACO inputs require --strict_taco_final_pool and --round_value."
                )
            if args.round_value is None:
                raise ValueError(
                    "--round_value is required with --strict_taco_final_pool."
                )
            sequences = read_validated_taco_sequences(
                input_path,
                dataset_label,
                args.round_value,
                max_sequences=args.max_sequences,
            )
        else:
            sequences = read_sequences(
                input_path,
                sequence_column=args.sequence_column,
                max_sequences=args.max_sequences,
            )

        dataset_sequence_ids[dataset_label] = [
            re.sub(r"\s+", "_", sequence_id)
            for sequence_id, _ in sequences
        ]

        fasta_path = os.path.join(prepared_dir, f"{dataset_label}.fasta")
        write_fasta(sequences, fasta_path)

        print(f"[{dataset_label}] wrote {len(sequences)} sequences to {fasta_path}", flush=True)

        dataset_fimo_dir = os.path.join(fimo_runs_dir, dataset_label)
        print(f"[{dataset_label}] running FIMO", flush=True)

        run_fimo(
            selected_meme=selected_meme_path,
            fasta_path=fasta_path,
            output_dir=dataset_fimo_dir,
            pvalue_threshold=args.pvalue_threshold,
            max_stored_scores=args.max_stored_scores,
        )

        fimo_tsv_path = os.path.join(dataset_fimo_dir, "fimo.tsv")
        hits = parse_fimo_tsv(
            fimo_tsv_path=fimo_tsv_path,
            dataset_label=dataset_label,
            matrix_id_to_label=matrix_id_to_label,
        )

        print(f"[{dataset_label}] FIMO hits: {len(hits)}", flush=True)
        all_hits.extend(hits)

    per_sequence_counts = defaultdict(int)

    for hit in all_hits:
        key = (hit["dataset"], hit["sequence_id"], hit["motif"], hit["matrix_id"])
        per_sequence_counts[key] += 1

    # Ensure zero-hit combinations are present.
    for dataset_label, sequence_ids in dataset_sequence_ids.items():
        for sequence_id in sequence_ids:
            for motif_label, matrix_id in motif_label_to_matrix_id.items():
                _ = per_sequence_counts[(dataset_label, sequence_id, motif_label, matrix_id)]

    per_sequence_rows = []
    for (dataset_label, sequence_id, motif_label, matrix_id), hit_count in sorted(per_sequence_counts.items()):
        per_sequence_rows.append({
            "dataset": dataset_label,
            "sequence_id": sequence_id,
            "motif": motif_label,
            "matrix_id": matrix_id,
            "hit_count": hit_count,
        })

    summary_rows = []
    for dataset_label, sequence_ids in dataset_sequence_ids.items():
        num_sequences = len(sequence_ids)

        for motif_label, matrix_id in motif_label_to_matrix_id.items():
            counts = [
                per_sequence_counts[(dataset_label, sequence_id, motif_label, matrix_id)]
                for sequence_id in sequence_ids
            ]

            total_hits = sum(counts)
            sequences_with_hit = sum(1 for count in counts if count > 0)
            mean_hits = total_hits / num_sequences if num_sequences else 0.0
            fraction_with_hit = sequences_with_hit / num_sequences if num_sequences else 0.0

            summary_rows.append({
                "dataset": dataset_label,
                "motif": motif_label,
                "matrix_id": matrix_id,
                "num_sequences": num_sequences,
                "total_hits": total_hits,
                "mean_hits_per_sequence": mean_hits,
                "sequences_with_at_least_one_hit": sequences_with_hit,
                "fraction_sequences_with_at_least_one_hit": fraction_with_hit,
            })

    selected_motif_rows = []
    for motif_label, matrix_id in motif_label_to_matrix_id.items():
        selected_motif_rows.append({
            "cell_line": args.cell_line,
            "motif": motif_label,
            "matrix_id": matrix_id,
            "raw_meme_motif_name": selected_raw_names.get(matrix_id, ""),
        })

    hits_output = os.path.join(args.output_dir, "tfbs_hits.csv")
    per_sequence_output = os.path.join(args.output_dir, "tfbs_per_sequence_counts.csv")
    summary_output = os.path.join(args.output_dir, "tfbs_summary.csv")
    selected_motifs_output = os.path.join(args.output_dir, "selected_motifs.csv")

    write_csv(
        hits_output,
        all_hits,
        [
            "dataset",
            "sequence_id",
            "motif",
            "matrix_id",
            "motif_alt_id",
            "start_1_based",
            "stop_1_based",
            "strand",
            "score",
            "p_value",
            "q_value",
            "matched_sequence",
        ],
    )

    write_csv(
        per_sequence_output,
        per_sequence_rows,
        [
            "dataset",
            "sequence_id",
            "motif",
            "matrix_id",
            "hit_count",
        ],
    )

    write_csv(
        summary_output,
        summary_rows,
        [
            "dataset",
            "motif",
            "matrix_id",
            "num_sequences",
            "total_hits",
            "mean_hits_per_sequence",
            "sequences_with_at_least_one_hit",
            "fraction_sequences_with_at_least_one_hit",
        ],
    )

    write_csv(
        selected_motifs_output,
        selected_motif_rows,
        [
            "cell_line",
            "motif",
            "matrix_id",
            "raw_meme_motif_name",
        ],
    )

    print(f"Saved selected MEME file: {selected_meme_path}")
    print(f"Saved hit table: {hits_output}")
    print(f"Saved per-sequence counts: {per_sequence_output}")
    print(f"Saved summary table: {summary_output}")
    print(f"Saved selected motif information: {selected_motifs_output}")


if __name__ == "__main__":
    main()
