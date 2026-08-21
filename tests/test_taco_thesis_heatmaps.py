import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_taco_thesis_heatmaps import (
    CELL_VMAX,
    annotation_color,
    create_heatmap_figure,
    output_stem,
    parse_args,
    pooled_dinucleotide_matrix,
)


class ThesisHeatmapTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_cli_defaults_to_five_seeds_and_round_100(self):
        args = parse_args(["--cell-type", "hepg2", "--output-dir", "/tmp/out"])
        self.assertEqual(args.taco_seeds, [0, 1, 2, 3, 4])
        self.assertEqual(args.taco_round, 100)

    def test_known_dinucleotide_matrix(self):
        matrix = pooled_dinucleotide_matrix(["ACGT", "ACGT"])
        expected = np.zeros((4, 4))
        expected[0, 1] = 1 / 3
        expected[1, 2] = 1 / 3
        expected[2, 3] = 1 / 3
        np.testing.assert_allclose(matrix, expected)

    def test_final_figure_has_six_panels_and_fixed_hepg2_scale(self):
        matrices = [("Original", np.zeros((4, 4)))] + [
            (f"seed{seed}", np.full((4, 4), seed / 100)) for seed in range(5)
        ]
        figure, axes = create_heatmap_figure(matrices, "hepg2")

        self.assertEqual(axes.shape, (2, 3))
        self.assertEqual(len(figure.axes), 7)  # six panels plus one colorbar
        for axis in axes.flat:
            self.assertEqual(axis.images[0].get_clim(), (0.0, 0.20))

    def test_k562_scale_and_annotation_threshold(self):
        vmax = CELL_VMAX["k562"]
        threshold = vmax * 0.52
        self.assertEqual(annotation_color(threshold, vmax), "white")
        self.assertEqual(annotation_color(threshold + 1e-12, vmax), "black")

        matrix = np.zeros((4, 4))
        matrix[0, 0] = threshold
        matrix[0, 1] = threshold + 0.001
        _, axes = create_heatmap_figure([("Original", matrix)], "k562")
        text_by_position = {
            tuple(text.get_position()): text.get_color() for text in axes[0, 0].texts
        }
        self.assertEqual(axes[0, 0].images[0].get_clim(), (0.0, 0.66))
        self.assertEqual(text_by_position[(0, 0)], "white")
        self.assertEqual(text_by_position[(1, 0)], "black")

    def test_established_output_names(self):
        output_dir = Path("figures")
        self.assertEqual(
            output_stem(output_dir, "hepg2"),
            output_dir / "taco_hepg2_dinucleotide_heatmaps_five_seed",
        )
        self.assertEqual(
            output_stem(output_dir, "k562"),
            output_dir / "taco_k562_dinucleotide_heatmaps_five_seed",
        )


if __name__ == "__main__":
    unittest.main()
