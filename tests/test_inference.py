"""Regression checks for map geometry and inference correctness (CPU only)."""
import contextlib
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import mrcfile
import numpy as np
import torch

from cryomni import SwinTransformer_MAE3D_New
from cryomni.checkpoint import load_model
from cryomni.protein_swin_mae3d import ShiftedWindowAttention
from data_processing.Gen_Box import generate_data_pair
from data_processing.Resize_Map import my_reform_1a
from data_processing.map_utils import segment_map
from scripts import infer


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.quiet = contextlib.redirect_stdout(io.StringIO())
        self.quiet.__enter__()
        self.addCleanup(self.quiet.__exit__, None, None, None)

    def write_input(self, data, voxel=(1, 1, 1), origin=(11, 22, 33)):
        path = self.root / 'input.mrc'
        with mrcfile.new(path, overwrite=True) as m:
            m.set_data(np.asarray(data, dtype=np.float32))
            m.voxel_size = voxel
            m.header.origin = origin
        return path

    def test_crop_keeps_last_foreground_voxel_and_origin(self):
        data = np.zeros((7, 8, 9), dtype=np.float32)
        data[1:4, 2:6, 3:8] = 1
        src = self.write_input(data)
        dest = self.root / 'crop.mrc'
        segment_map(str(src), str(dest), 0.5)
        with mrcfile.open(dest) as m:
            self.assertEqual(m.data.shape, (3, 4, 5))
            np.testing.assert_allclose(m.header.origin.tolist(), (14, 24, 34))

    def test_empty_crop_reports_threshold(self):
        src = self.write_input(np.zeros((4, 4, 4)))
        with self.assertRaisesRegex(ValueError, 'contour|threshold'):
            segment_map(str(src), str(self.root / 'crop.mrc'), 1)

    def test_relative_resize_link_and_rerun(self):
        src = self.write_input(np.ones((3, 4, 5))).relative_to(Path.cwd())
        dest = (self.root / 'resize.mrc').relative_to(Path.cwd())
        for _ in range(2):
            my_reform_1a(str(src), str(dest), use_gpu=False)
            self.assertTrue(dest.exists(), 'Relative-path resize produced a dangling link')
            with mrcfile.open(dest) as m:
                self.assertEqual(m.data.shape, (3, 4, 5))

    def test_normalization_keeps_origin(self):
        data = np.arange(8**3, dtype=np.float32).reshape(8, 8, 8)
        src = self.write_input(data)
        generate_data_pair(str(src), None, None, str(self.root), 1, 8, 8, 'test')
        with mrcfile.open(self.root / 'test_normal.mrc') as m:
            np.testing.assert_allclose(m.header.origin.tolist(), (11, 22, 33))
            self.assertTrue(np.isfinite(m.data).all())

    def test_constant_map_rejected_without_nan(self):
        for value in (0, 1):
            src = self.write_input(np.full((4, 4, 4), value))
            with self.assertRaisesRegex(ValueError, 'density|constant|normaliz'):
                generate_data_pair(str(src), None, None, str(self.root), -1, 4, 4, 'test')

    def test_singleton_spatial_axis_is_preserved(self):
        src = self.write_input(np.ones((1, 4, 5)))
        dest = self.root / 'output.mrc'
        infer.write_map(torch.ones(1, 1, 1, 4, 5), dest, src)
        with mrcfile.open(dest) as m:
            self.assertEqual(m.data.shape, (1, 4, 5))

    def test_attention_eval_disables_dropout(self):
        layer = ShiftedWindowAttention(12, [2, 2, 2], [0, 0, 0], 3,
                                       attention_dropout=0.5, dropout=0.5).eval()
        x = torch.randn(1, 4, 4, 4, 12)
        with torch.inference_mode():
            torch.testing.assert_close(layer(x), layer(x), rtol=0, atol=0)

    def test_seed_controls_model_mask_rng(self):
        src = self.write_input(np.ones((4, 4, 4)))
        model = MagicMock(resolution=64)
        values = []
        argv = ['infer.py', str(src), '--device', 'cpu', '--seed', '17']
        with patch('sys.argv', argv), patch.object(infer, 'load_model', return_value=model), \
             patch.object(infer, 'run_single_map', side_effect=lambda **kw: values.append(random.random())):
            infer.main()
            infer.main()
        self.assertEqual(values[0], values[1])

    def test_empty_checkpoint_is_rejected(self):
        path = self.root / 'empty.pt'
        torch.save({}, path)
        with patch('cryomni.checkpoint.SwinTransformer_MAE3D_New', return_value=torch.nn.Linear(2, 2)):
            with self.assertRaises(RuntimeError):
                load_model(path)

    def test_small_model_forward_shape_and_finiteness(self):
        model = SwinTransformer_MAE3D_New(
            patch_size=[4, 4, 4], embed_dim=96, depths=[1, 1, 1, 1],
            num_heads=[3, 6, 12, 24], window_size=[2, 2, 2],
            resolution=32, stochastic_depth_prob=0,
        ).eval()
        x = torch.rand(1, 1, 32, 32, 32)
        with torch.inference_mode():
            removed, predicted, visible = model.forward_pred(x)
        for value in (removed, predicted, visible):
            self.assertEqual(value.shape, x.shape)
            self.assertTrue(torch.isfinite(value).all())
        torch.testing.assert_close(removed + visible, x)

    def test_preprocess_and_output_geometry_end_to_end(self):
        data = np.zeros((9, 10, 11), dtype=np.float32)
        data[2:7, 3:8, 4:9] = np.arange(1, 126).reshape(5, 5, 5)
        src = self.write_input(data)

        class IdentityModel:
            def forward_pred(self, x):
                return torch.zeros_like(x), torch.zeros_like(x), x

        output = self.root / 'result'
        infer.run_single_map(IdentityModel(), src, output, 0.5, 8, 8,
                             torch.device('cpu'), False)
        with mrcfile.open(output / 'recon_input.mrc') as result, \
             mrcfile.open(output / 'preprocessed/input/input_normal.mrc') as normalized:
            self.assertEqual(result.data.shape, (5, 5, 5))
            np.testing.assert_allclose(result.header.origin.tolist(), (15, 25, 35))
            np.testing.assert_array_equal(result.data, normalized.data)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
