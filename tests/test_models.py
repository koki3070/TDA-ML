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
        ellipse_param_dim = 5

        # Instantiate the model
        model = AnisotropicOutlierClassifier(point_dim=point_dim)

        # Create dummy input data
        dummy_input = torch.randn(batch_size, num_points, point_dim)

        # Forward pass
        outlier_logits, ellipse_params = model(dummy_input)

        # Check the shapes of the outputs
        self.assertEqual(outlier_logits.shape, (batch_size, num_points, 1))
        self.assertEqual(ellipse_params.shape, (batch_size, num_points, ellipse_param_dim))

        # Check the types of the outputs
        self.assertIsInstance(outlier_logits, torch.Tensor)
        self.assertIsInstance(ellipse_params, torch.Tensor)

    def test_forward_ellipse_param_dim_three(self):
        model = AnisotropicOutlierClassifier(ellipse_param_dim=3)
        x = torch.randn(2, 32, 2)
        logits, params = model(x)
        self.assertEqual(logits.shape, (2, 32, 1))
        self.assertEqual(params.shape, (2, 32, 5))

    def test_invalid_ellipse_param_dim_raises(self):
        with self.assertRaises(ValueError):
            AnisotropicOutlierClassifier(ellipse_param_dim=4)

if __name__ == '__main__':
    unittest.main()