import tempfile
import unittest
from pathlib import Path
from unittest import mock

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch

import master_workflow
import plot_taco_compositional_bias as composition_plot
import plot_tfbs_rescan_results as tfbs_plot


class TacoCliAndRoundTests(unittest.TestCase):
    def test_default_and_custom_taco_selection(self):
        defaults = master_workflow.parse_args(["analyze", "--fimo", "no"])
        self.assertEqual(defaults.taco_seeds, [0, 1, 2, 3, 4])
        self.assertEqual(defaults.taco_round, 100)

        custom = master_workflow.parse_args(
            ["analyze", "--taco-seeds", "1", "3", "4", "--taco-round", "87"]
        )
        self.assertEqual(custom.taco_seeds, [1, 3, 4])
        self.assertEqual(custom.taco_round, 87)

    def test_requested_effective_round_reaches_strict_composition_validator(self):
        args = master_workflow.parse_args(
            [
                "analyze",
                "--dry-run",
                "--cell-types", "hepg2",
                "--taco-data-dir", "/data",
                "--taco-results-dir", "/results",
                "--taco-seeds", "1", "3",
                "--taco-round", "99",
                "--output-root", "/output",
            ]
        )
        args.custom_datasets = []
        runner = master_workflow.Runner(args)
        with mock.patch("builtins.print"):
            master_workflow.analyze_taco(runner)

        self.assertEqual(len(runner.results), 2)
        for result in runner.results:
            self.assertIn("--strict_taco_final_pool", result.command)
            self.assertNotIn("--round_mode", result.command)
            self.assertEqual(
                result.command[result.command.index("--round_value") + 1], "99"
            )
        self.assertFalse(hasattr(master_workflow, "best_fitness_round"))

    def test_compositional_bias_plots_follow_selected_cell_types(self):
        args = master_workflow.parse_args(
            [
                "analyze",
                "--dry-run",
                "--cell-types",
                "hepg2",
                "--taco-data-dir",
                "/data",
                "--taco-results-dir",
                "/results",
                "--taco-seeds",
                "1",
                "3",
                "--taco-round",
                "99",
                "--output-root",
                "/output",
            ]
        )
        args.custom_datasets = []
        runner = master_workflow.Runner(args)
        with mock.patch("builtins.print"):
            master_workflow.create_thesis_specific_composition_plots(runner)

        self.assertEqual(
            [result.name for result in runner.results],
            [
                "Create HepG2 TACO six-panel dinucleotide heatmap",
                "Create HepG2 TACO compositional-bias figures",
            ],
        )
        heatmap_command = runner.results[0].command
        self.assertEqual(
            heatmap_command[heatmap_command.index("--taco-round") + 1], "99"
        )
        heatmap_seed_start = heatmap_command.index("--taco-seeds") + 1
        self.assertEqual(
            heatmap_command[heatmap_seed_start : heatmap_seed_start + 2],
            ["1", "3"],
        )

        command = runner.results[1].command
        self.assertEqual(command[command.index("--cell-type") + 1], "hepg2")
        self.assertEqual(command[command.index("--taco-round") + 1], "99")
        seed_start = command.index("--taco-seeds") + 1
        self.assertEqual(command[seed_start : seed_start + 2], ["1", "3"])
        self.assertIn(
            "taco_hepg2_compositional_bias",
            command[command.index("--results-dir") + 1],
        )

    def test_custom_seeds_and_round_reach_fimo_and_final_aggregation(self):
        args = master_workflow.parse_args(
            [
                "tfbs",
                "--dry-run",
                "--fimo", "yes",
                "--cell-types", "hepg2",
                "--taco-data-dir", "/data",
                "--taco-results-dir", "/results",
                "--motif-file", "/motifs.meme",
                "--taco-seeds", "1", "3", "4",
                "--taco-round", "99",
                "--output-root", "/output",
            ]
        )
        args.custom_datasets = []
        runner = master_workflow.Runner(args)
        with mock.patch("builtins.print"):
            master_workflow.run_tfbs_scans(runner)

        taco_step = next(
            result for result in runner.results if result.name.endswith("hepg2 taco")
        )
        self.assertIn("--strict_taco_final_pool", taco_step.command)
        self.assertEqual(
            taco_step.command[taco_step.command.index("--round_value") + 1], "99"
        )
        taco_inputs = " ".join(taco_step.command)
        self.assertIn("taco_hepg2_1=", taco_inputs)
        self.assertIn("taco_hepg2_3=", taco_inputs)
        self.assertIn("taco_hepg2_4=", taco_inputs)
        self.assertNotIn("taco_hepg2_0=", taco_inputs)

        aggregate = runner.results[-1].command
        seed_start = aggregate.index("--taco-seeds") + 1
        self.assertEqual(aggregate[seed_start : seed_start + 3], ["1", "3", "4"])
        self.assertEqual(
            aggregate[aggregate.index("--taco-round") + 1], "99"
        )

    def test_limited_fimo_dry_run_does_not_create_final_aggregation(self):
        args = master_workflow.parse_args(
            [
                "tfbs",
                "--dry-run",
                "--fimo", "yes",
                "--cell-types", "hepg2",
                "--taco-data-dir", "/data",
                "--taco-results-dir", "/results",
                "--motif-file", "/motifs.meme",
                "--max-sequences", "10",
                "--output-root", "/output",
            ]
        )
        args.custom_datasets = []
        runner = master_workflow.Runner(args)
        with mock.patch("builtins.print"):
            master_workflow.run_tfbs_scans(runner)

        final_step = runner.results[-1]
        self.assertEqual(final_step.name, "Aggregate and plot TFBS/FIMO results")
        self.assertEqual(final_step.status, "skipped")
        self.assertIn("non-final", final_step.message)
        self.assertEqual(final_step.command, [])


class TacoPlotTests(unittest.TestCase):
    def setUp(self):
        self.datasets = [
            ("Real K562", Path("real.csv"), None, None),
            ("TACO seed1", Path("seed1.csv"), 100, 1),
            ("TACO seed3", Path("seed3.csv"), 100, 3),
            ("TACO seed4", Path("seed4.csv"), 100, 4),
        ]
        self.colors = composition_plot.colors_for_datasets(self.datasets)

    def test_seed_color_contract(self):
        for original_label in ("Real HepG2", "Real K562"):
            all_seeds = [(original_label, Path("real.csv"), None, None)] + [
                (f"TACO seed{seed}", Path(f"seed{seed}.csv"), 100, seed)
                for seed in range(5)
            ]
            colors = composition_plot.colors_for_datasets(all_seeds)
            self.assertEqual(colors[0], composition_plot.ORIGINAL_COLOR)
            self.assertEqual(
                colors[1:],
                [composition_plot.SEED_GREEN_COLORS[i] for i in range(5)],
            )

    def test_hepg2_dataset_paths_and_output_defaults(self):
        args = composition_plot.parse_args(
            [
                "--cell-type", "hepg2",
                "--project-root", "/project",
                "--taco-data-dir", "/data",
                "--taco-results-dir", "/results",
                "--taco-seeds", "1", "3",
                "--taco-round", "99",
            ]
        )
        datasets = composition_plot.build_datasets(args)
        self.assertEqual(datasets[0], ("Real HepG2", Path("/data/hepg2/mbo.csv"), None, None))
        self.assertEqual(
            datasets[1],
            (
                "TACO seed1",
                Path("/results/full_runs/hepg2_seed1_b64_ga4/analysis/hepg2_seed1_all_candidates_annotated.csv"),
                99,
                1,
            ),
        )
        self.assertEqual([dataset[3] for dataset in datasets[1:]], [1, 3])
        self.assertEqual(args.results_dir.name, "taco_hepg2_compositional_bias")
        self.assertEqual(args.figure_dir, args.results_dir / "figures")

    def test_variable_seed_barplot_uses_dataset_colors(self):
        aggregate = [
            {"dataset": dataset, "dinucleotide": dinuc, "frequency": 0.1}
            for dataset, _, _, _ in self.datasets
            for dinuc in composition_plot.SELECTED_DINUCS
        ]
        captured = []
        with mock.patch.object(composition_plot, "save_figure", side_effect=lambda fig, _: captured.append(fig)):
            composition_plot.make_barplot(aggregate, Path("unused"), self.datasets)
        fig = captured[0]
        bars = fig.axes[0].patches
        self.assertEqual(len(bars), len(self.datasets) * len(composition_plot.SELECTED_DINUCS))
        for index, expected in enumerate(self.colors):
            group = bars[index * len(composition_plot.SELECTED_DINUCS) : (index + 1) * len(composition_plot.SELECTED_DINUCS)]
            self.assertTrue(all(mcolors.to_hex(bar.get_facecolor()) == expected for bar in group))
            self.assertTrue(all(mcolors.to_hex(bar.get_edgecolor()) == expected for bar in group))
        plt.close(fig)

    def test_variable_seed_boxplot_has_no_black_taco_artists(self):
        per_sequence = [
            {"dataset": dataset, "GG": value}
            for dataset, _, _, _ in self.datasets
            for value in (0.1, 0.2, 0.3)
        ]
        captured = []
        with mock.patch.object(composition_plot, "save_figure", side_effect=lambda fig, _: captured.append(fig)):
            composition_plot.make_dinucleotide_boxplot(
                per_sequence, Path("unused"), self.datasets
            )
        fig = captured[0]
        ax = fig.axes[0]
        boxes = [artist for artist in ax.get_children() if isinstance(artist, PathPatch)]
        self.assertEqual(
            [mcolors.to_hex(patch.get_facecolor()) for patch in boxes],
            self.colors,
        )
        allowed = set(self.colors)
        self.assertTrue(ax.lines)
        self.assertTrue(all(mcolors.to_hex(line.get_color()) in allowed for line in ax.lines))
        plt.close(fig)

    def test_tfbs_labels_and_order_accept_seed_subset(self):
        approaches = ["TFBS-guided", "TACO seed 4", "Real", "TACO seed 1"]
        self.assertEqual(tfbs_plot.dataset_label("taco_k562_3"), "TACO seed 3")
        self.assertEqual(
            tfbs_plot.ordered_approaches(approaches),
            ["Real", "TACO seed 1", "TACO seed 4", "TFBS-guided"],
        )

    def test_full_tfbs_occurrence_plot_uses_log_scale_and_muted_colors(self):
        table = tfbs_plot.pd.DataFrame(
            {
                "approach": ["Real", "TACO seed 0", "TFBS-guided"],
                "n": [100, 100, 100],
                "CEBPA": [4.0, 0.0, 69.0],
                "FOXA2": [3.5, 0.4, 52.0],
                "HNF4A": [3.8, 2.0, 100.0],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            with (
                mock.patch.object(tfbs_plot.plt, "savefig"),
                mock.patch.object(tfbs_plot.plt, "close"),
            ):
                tfbs_plot.plot_motif_percentages(
                    table,
                    "hepg2",
                    Path(tmp),
                )
                fig = tfbs_plot.plt.gcf()

        ax = fig.axes[0]
        self.assertEqual(ax.get_yscale(), "log")
        self.assertAlmostEqual(ax.get_ylim()[0], 0.01)
        self.assertEqual(
            {mcolors.to_hex(patch.get_facecolor()) for patch in ax.patches},
            {color.lower() for color in tfbs_plot.FULL_PLOT_COLORS},
        )
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
