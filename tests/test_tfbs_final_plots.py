import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import plot_tfbs_rescan_results as tfbs_plot


def synthetic_percentage_table():
    approaches = [
        "Real",
        "Uniform baseline",
        "GC-matched baseline",
        "Markov baseline",
        *[f"TACO seed {seed}" for seed in range(5)],
        "TFBS-guided",
    ]
    rows = []
    for index, approach in enumerate(approaches):
        if approach.startswith("TACO seed "):
            seed = int(approach.rsplit(" ", 1)[1])
            values = (0.0, float(seed), 1.0 + seed)
            n = 256
        else:
            values = (1.0 + index, 2.0 + index, 3.0 + index)
            n = 100
        rows.append(
            {
                "approach": approach,
                "n": n,
                "CEBPA": values[0],
                "FOXA2": values[1],
                "HNF4A": values[2],
            }
        )
    return pd.DataFrame(rows)


def synthetic_metric_rows():
    motifs = ["CEBPA", "FOXA2", "HNF4A", "ANY_SELECTED", "ALL_3_SELECTED"]
    approaches = [
        "Real",
        "Uniform baseline",
        "GC-matched baseline",
        "Markov baseline",
        *[f"TACO seed {seed}" for seed in range(5)],
        "TFBS-guided",
    ]
    rows = []
    for approach_index, approach in enumerate(approaches):
        is_taco = approach.startswith("TACO seed ")
        seed = int(approach.rsplit(" ", 1)[1]) if is_taco else None
        for motif_index, motif in enumerate(motifs):
            hit_fraction = (
                (seed + motif_index) / 100.0
                if is_taco
                else (approach_index + motif_index + 1) / 100.0
            )
            n = 256 if is_taco else 100
            total_hits = math.nan if motif == "ALL_3_SELECTED" else hit_fraction * n
            rows.append(
                {
                    "dataset": (
                        f"taco_hepg2_{seed}"
                        if is_taco
                        else approach.lower().replace(" ", "_") + "_hepg2"
                    ),
                    "approach": approach,
                    "motif": motif,
                    "matrix_id": motif,
                    "total_sequences": n,
                    "sequences_with_hit": hit_fraction * n,
                    "hit_fraction": hit_fraction,
                    "hit_fraction_percent": hit_fraction * 100.0,
                    "total_hits": total_hits,
                    "mean_hits_per_sequence": (
                        math.nan if motif == "ALL_3_SELECTED" else hit_fraction
                    ),
                }
            )
    return pd.DataFrame(rows)


class FinalTfbsFigureTests(unittest.TestCase):
    def test_summary_reader_accepts_current_scanner_schema(self):
        summary = pd.DataFrame(
            [
                {
                    "dataset": "real_hepg2",
                    "motif": "CEBPA",
                    "num_sequences": 100,
                    "sequences_with_at_least_one_hit": 4,
                    "fraction_sequences_with_at_least_one_hit": 0.04,
                    "total_hits": 5,
                    "mean_hits_per_sequence": 0.05,
                }
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            scan_dir = Path(tmp) / "hepg2_baseline_fimo"
            scan_dir.mkdir()
            summary.to_csv(scan_dir / "tfbs_summary.csv", index=False)
            observed = tfbs_plot.read_all_summaries(Path(tmp))

        self.assertEqual(observed["total_sequences"].tolist(), [100])
        self.assertEqual(observed["sequences_with_hit"].tolist(), [4])
        self.assertEqual(observed["hit_fraction"].tolist(), [0.04])

    def test_summary_reader_requires_current_scanner_schema(self):
        summary = pd.DataFrame(
            [
                {
                    "dataset": "real_hepg2",
                    "motif": "CEBPA",
                    "n_sequences": 100,
                    "sequences_with_at_least_one_hit": 4,
                    "fraction_sequences_with_at_least_one_hit": 0.04,
                    "total_hits": 5,
                    "mean_hits_per_sequence": 0.05,
                }
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            scan_dir = Path(tmp) / "hepg2_baseline_fimo"
            scan_dir.mkdir()
            summary.to_csv(scan_dir / "tfbs_summary.csv", index=False)
            with self.assertRaisesRegex(ValueError, "missing columns"):
                tfbs_plot.read_all_summaries(Path(tmp))

    def test_plotter_cli_defaults_and_custom_seed_subset(self):
        defaults = tfbs_plot.parse_args([])
        self.assertEqual(defaults.taco_seeds, [0, 1, 2, 3, 4])
        self.assertEqual(defaults.taco_round, 100)
        custom = tfbs_plot.parse_args(
            ["--taco-seeds", "1", "3", "4", "--taco-round", "99"]
        )
        self.assertEqual(custom.taco_seeds, [1, 3, 4])
        self.assertEqual(custom.taco_round, 99)

    def test_taco_is_one_mean_category_with_sample_sd_and_seed_points(self):
        table = synthetic_percentage_table()
        figure_data, seed_points, order = tfbs_plot.make_final_figure_data(
            table, "hepg2"
        )

        self.assertEqual(order, tfbs_plot.FINAL_APPROACH_ORDER)
        self.assertEqual(order.count("TACO"), 1)
        taco_foxa2 = figure_data[
            (figure_data["approach"] == "TACO")
            & (figure_data["motif"] == "FOXA2")
        ].iloc[0]
        self.assertAlmostEqual(taco_foxa2["mean_percent"], 2.0)
        self.assertAlmostEqual(
            taco_foxa2["sd_percent"],
            pd.Series([0.0, 1.0, 2.0, 3.0, 4.0]).std(ddof=1),
        )
        self.assertEqual(len(seed_points), 15)
        self.assertEqual(
            seed_points[seed_points["motif"] == "FOXA2"]["seed"].tolist(),
            [0, 1, 2, 3, 4],
        )
        output = tfbs_plot.make_figure_output_data(
            figure_data, seed_points, "hepg2", taco_round=100
        )
        taco_output = output[
            (output["approach"] == "TACO") & (output["motif"] == "FOXA2")
        ].iloc[0]
        self.assertEqual(taco_output["effective_round"], 100)
        self.assertEqual(taco_output["individual_seed_values_percent"], "0;1;2;3;4")

    def test_log_plot_keeps_true_zeros_without_pseudocount(self):
        table = synthetic_percentage_table()
        with tempfile.TemporaryDirectory() as tmp:
            with (
                mock.patch.object(tfbs_plot.plt, "savefig"),
                mock.patch.object(tfbs_plot.plt, "close"),
            ):
                figure_data, seed_points = tfbs_plot.plot_motif_percentages(
                    table,
                    "hepg2",
                    Path(tmp),
                )
                fig = tfbs_plot.plt.gcf()

        ax = fig.axes[0]
        self.assertEqual(ax.get_yscale(), "log")
        self.assertEqual(
            [label.get_text() for label in ax.get_xticklabels()],
            tfbs_plot.FINAL_APPROACH_ORDER,
        )
        taco_cebpa = figure_data[
            (figure_data["approach"] == "TACO")
            & (figure_data["motif"] == "CEBPA")
        ].iloc[0]
        self.assertEqual(taco_cebpa["mean_percent"], 0.0)
        self.assertTrue(
            (seed_points[seed_points["motif"] == "CEBPA"]["hit_fraction_percent"] == 0).all()
        )
        self.assertIn(0.0, [patch.get_height() for patch in ax.patches])
        plt.close(fig)

    def test_final_csv_summaries_retain_seed_rows_and_taco_statistics(self):
        metric_rows = synthetic_metric_rows()
        taco = tfbs_plot.make_taco_seed_summary(
            metric_rows, "hepg2", taco_round=100
        )
        final = tfbs_plot.make_dataset_summary(
            metric_rows,
            "hepg2",
            selected_seeds=[0, 1, 2, 3, 4],
            taco_round=100,
        )

        self.assertEqual(sorted(taco["seed"].unique().tolist()), [0, 1, 2, 3, 4])
        self.assertEqual(taco["effective_round"].unique().tolist(), [100])
        self.assertEqual(final["taco_effective_round"].unique().tolist(), [100])
        self.assertEqual(
            set(taco["motif"]),
            {"CEBPA", "FOXA2", "HNF4A", "ANY_SELECTED", "ALL_3_SELECTED"},
        )
        self.assertEqual(set(final["approach"].astype(str)), set(tfbs_plot.FINAL_APPROACH_ORDER))

        taco_cebpa = final[
            (final["approach"].astype(str) == "TACO")
            & (final["motif"].astype(str) == "CEBPA")
        ].iloc[0]
        self.assertEqual(taco_cebpa["n_seeds"], 5)
        self.assertAlmostEqual(taco_cebpa["hit_fraction_percent"], 2.0)
        self.assertAlmostEqual(
            taco_cebpa["hit_fraction_percent_sd_across_seeds"],
            pd.Series([0.0, 1.0, 2.0, 3.0, 4.0]).std(ddof=1),
        )

    def test_final_summary_rejects_incomplete_taco_scan_population(self):
        metric_rows = synthetic_metric_rows()
        metric_rows.loc[
            metric_rows["approach"] == "TACO seed 3", "total_sequences"
        ] = 32
        with self.assertRaisesRegex(ValueError, "exactly 256 sequences"):
            tfbs_plot.make_dataset_summary(
                metric_rows,
                "hepg2",
                selected_seeds=[0, 1, 2, 3, 4],
                taco_round=100,
            )


if __name__ == "__main__":
    unittest.main()
