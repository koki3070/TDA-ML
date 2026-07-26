import torch
import unittest
from tda_ml.models import AnisotropicOutlierClassifier

class TestAnisotropicOutlierClassifier(unittest.TestCase):
    def test_forward_pass(self):
        """
        Tests if the AnisotropicOutlierClassifier model can be initialized
        and can perform a forward pass with dummy data.
        """
        batch_size = 4
        num_points = 100
        point_dim = 2

        model = AnisotropicOutlierClassifier(point_dim=point_dim)

        dummy_input = torch.randn(batch_size, num_points, point_dim)

        outlier_logits, ellipse_params = model(dummy_input)

        self.assertEqual(outlier_logits.shape, (batch_size, num_points, 1))
        self.assertEqual(ellipse_params.shape, (batch_size, num_points, 3))

        self.assertIsInstance(outlier_logits, torch.Tensor)
        self.assertIsInstance(ellipse_params, torch.Tensor)

    def test_legacy_axes_positive_without_modeling_floor(self):
        """Legacy forward keeps a,b > 0 without encoder clamp or +1e-4 offset."""
        torch.manual_seed(3)
        model = AnisotropicOutlierClassifier()
        x = torch.rand(1, 24, 2) * 0.8 - 0.4
        with torch.no_grad():
            _, params = model(x)
        axes = params[0, :, 0:2]
        self.assertTrue((axes > 0).all())

    def test_axes_use_exp_delta_times_base(self):
        """a_i = exp(Delta a_i) * a_base (no sigmoid clip on axis scale)."""
        torch.manual_seed(7)
        model = AnisotropicOutlierClassifier()
        x = torch.rand(1, 20, 2)
        with torch.no_grad():
            _, topo_feats, _, base_axes = model.encoder(x)
            raw = model.topology_head(topo_feats)
            expected_axes = torch.exp(raw[:, :, 0:2]) * base_axes
            _, params = model(x)
        torch.testing.assert_close(params[:, :, 0:2], expected_axes)

if __name__ == '__main__':
    unittest.main()
