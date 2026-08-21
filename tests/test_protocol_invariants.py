import csv
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import analyze_taco_generation_outputs as composition
import generate_gc_matched_random as gc_matched
import plot_taco_compositional_bias as composition_plot
import scan_selected_tfbs as fimo_scan
from taco_final_pool import FinalPoolValidationError, load_validated_final_pool


FIELDS = [
    "cell_line",
    "seed",
    "round",
    "effective_round",
    "physical_batch_size",
    "grad_accum_steps",
    "sequence",
    "surrogate_score",
    "oracle_score",
]


def unique_dna(index, length=200):
    bases = "ACGT"
    sequence = ["A"] * length
    position = length - 1
    value = index
    while value:
        sequence[position] = bases[value % 4]
        value //= 4
        position -= 1
    return "".join(sequence)


def valid_pool_rows(cell="hepg2", seed=0, effective_round=100):
    first_batch = effective_round * 4 - 3
    rows = []
    for index in range(256):
        rows.append(
            {
                "cell_line": cell,
                "seed": seed,
                "round": first_batch + index // 64,
                "effective_round": effective_round,
                "physical_batch_size": 64,
                "grad_accum_steps": 4,
                "sequence": unique_dna(index),
                "surrogate_score": 0.1,
                "oracle_score": 0.2,
            }
        )
    return rows


def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


class FinalPoolInvariantTests(unittest.TestCase):
    def validate(self, path, cell="hepg2", seed=0, effective_round=100):
        return load_validated_final_pool(
            path,
            expected_cell_line=cell,
            expected_seed=seed,
            effective_round=effective_round,
        )

    def test_accepts_exact_protocol_pool(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pool.csv"
            write_rows(path, valid_pool_rows())
            pool = self.validate(path)
            self.assertEqual(len(pool.sequences), 256)
            self.assertEqual(len(set(pool.sequences)), 256)
            self.assertEqual(pool.physical_batches, (397, 398, 399, 400))

    def test_rejects_wrong_size_and_duplicate_sequences(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pool.csv"
            rows = valid_pool_rows()
            write_rows(path, rows[:-1])
            with self.assertRaisesRegex(FinalPoolValidationError, "255 rows"):
                self.validate(path)

            rows[-1]["sequence"] = rows[0]["sequence"]
            write_rows(path, rows)
            with self.assertRaisesRegex(FinalPoolValidationError, "duplicate"):
                self.validate(path)

    def test_rejects_wrong_round_seed_and_cell_line(self):
        cases = [
            ("effective_round", 99, "0 rows"),
            ("seed", 1, "Wrong seed"),
            ("cell_line", "k562", "Wrong cell_line"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pool.csv"
            for column, value, message in cases:
                with self.subTest(column=column):
                    rows = valid_pool_rows()
                    for row in rows:
                        row[column] = value
                    write_rows(path, rows)
                    with self.assertRaisesRegex(FinalPoolValidationError, message):
                        self.validate(path)

    def test_rejects_wrong_batches_batch_size_and_gradient_accumulation(self):
        cases = [
            ("round", 396, "physical batches"),
            ("round", 397.5, "Invalid 'round'"),
            ("physical_batch_size", 32, "physical_batch_size"),
            ("grad_accum_steps", 2, "grad_accum_steps"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pool.csv"
            for column, value, message in cases:
                with self.subTest(column=column):
                    rows = valid_pool_rows()
                    rows[0][column] = value
                    write_rows(path, rows)
                    with self.assertRaisesRegex(FinalPoolValidationError, message):
                        self.validate(path)

    def test_rejects_missing_invalid_or_wrong_length_sequence(self):
        cases = [
            ("", "Missing DNA sequence"),
            ("N" + "A" * 199, "Non-ACGT"),
            ("A" * 199, "length 199"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pool.csv"
            for value, message in cases:
                with self.subTest(value=value[:4]):
                    rows = valid_pool_rows()
                    rows[0]["sequence"] = value
                    write_rows(path, rows)
                    with self.assertRaisesRegex(FinalPoolValidationError, message):
                        self.validate(path)

    def test_composition_and_fimo_receive_identical_sequence_set(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pool.csv"
            write_rows(path, valid_pool_rows())
            composition_sequences, raw_df = composition.load_strict_taco_dataset(
                path, "taco_hepg2_0", 100
            )
            fimo_sequences = fimo_scan.read_validated_taco_sequences(
                str(path), "taco_hepg2_0", 100
            )
            limited_fimo_sequences = fimo_scan.read_validated_taco_sequences(
                str(path), "taco_hepg2_0", 100, max_sequences=1
            )
            self.assertEqual(len(composition_sequences), 256)
            self.assertEqual(
                set(composition_sequences),
                {sequence for _, sequence in fimo_sequences},
            )
            self.assertEqual(
                limited_fimo_sequences,
                [("seq_1", composition_sequences[0])],
            )
            score_summary = composition.summarize_scores("taco_hepg2_0", raw_df)
            self.assertEqual(score_summary["min_round"], 397)
            self.assertEqual(score_summary["max_round"], 400)
            self.assertAlmostEqual(score_summary["mean_surrogate_score"], 0.1)
            self.assertAlmostEqual(score_summary["mean_oracle_score"], 0.2)


class FimoConfigurationTests(unittest.TestCase):
    def test_primary_command_uses_pvalue_both_strands_and_default_background(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(fimo_scan.subprocess, "run") as run:
                fimo_scan.run_fimo(
                    selected_meme="selected.meme",
                    fasta_path="sequences.fasta",
                    output_dir=str(Path(directory) / "fimo"),
                    pvalue_threshold=1e-4,
                    max_stored_scores=10_000_000,
                )
            command = run.call_args.args[0]
            self.assertEqual(command[command.index("--thresh") + 1], "0.0001")
            self.assertEqual(
                command[command.index("--max-stored-scores") + 1], "10000000"
            )
            self.assertNotIn("--bgfile", command)
            self.assertNotIn("--norc", command)
            self.assertNotIn("--qv-thresh", command)


class CompositionFormulaTests(unittest.TestCase):
    def test_shannon_entropy_known_examples_and_zero_probabilities(self):
        self.assertEqual(composition.shannon_entropy("AAAA"), 0.0)
        self.assertEqual(composition.shannon_entropy("ACGT"), 2.0)

    def test_known_dinucleotide_frequencies(self):
        _, _, rows, _ = composition.summarize_dataset("known", ["AACA"])
        lookup = {row["dinucleotide"]: row["frequency"] for row in rows}
        self.assertAlmostEqual(lookup["AA"], 1 / 3)
        self.assertAlmostEqual(lookup["AC"], 1 / 3)
        self.assertAlmostEqual(lookup["CA"], 1 / 3)
        self.assertEqual(lookup["GG"], 0.0)

    def test_taco_seed_summary_uses_sample_sd_and_range(self):
        datasets = [("Real HepG2", Path("real.csv"), None, None)] + [
            (f"TACO seed{seed}", Path(f"seed{seed}.csv"), 100, seed)
            for seed in range(5)
        ]
        per_sequence = []
        aggregate = []
        for seed in range(5):
            dataset = f"TACO seed{seed}"
            per_sequence.append(
                {
                    "dataset": dataset,
                    "gc_content": seed / 10,
                    "mononucleotide_shannon_entropy": 2 - seed / 10,
                    **{dinucleotide: seed / 100 for dinucleotide in composition_plot.SELECTED_DINUCS},
                }
            )
            aggregate.extend(
                {
                    "dataset": dataset,
                    "dinucleotide": dinucleotide,
                    "frequency": seed / 100,
                }
                for dinucleotide in composition_plot.SELECTED_DINUCS
            )
        seed_rows, summary_rows = composition_plot.make_taco_seed_composition_summaries(
            per_sequence, aggregate, datasets, "hepg2"
        )
        self.assertEqual([row["seed"] for row in seed_rows], [0, 1, 2, 3, 4])
        gc_summary = next(row for row in summary_rows if row["metric"] == "mean_gc_content")
        self.assertAlmostEqual(gc_summary["mean_across_seeds"], 0.2)
        self.assertAlmostEqual(gc_summary["sample_sd_across_seeds"], 0.15811388300841897)
        self.assertEqual(gc_summary["min_across_seeds"], 0.0)
        self.assertEqual(gc_summary["max_across_seeds"], 0.4)


class GcMatchedSchemaTests(unittest.TestCase):
    def test_export_index_column_is_optional(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "mbo.csv"
            with input_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["sequence", "target"])
                writer.writeheader()
                writer.writerow({"sequence": "ACGT", "target": 1.0})
            random.seed(23)
            gc_matched.process_dataset("hepg2", input_path, root / "out")
            fasta = (root / "out/baseline_B_gc_matched_hepg2.fasta").read_text()
            self.assertTrue(fasta.startswith(">hepg2_1|baseline_B_gc_matched|hepg2\n"))


if __name__ == "__main__":
    unittest.main()
