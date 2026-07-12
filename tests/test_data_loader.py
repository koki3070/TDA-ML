import torch
import unittest
from tda_ml.data_loader import get_dataset, PreloadedOutlierMNIST, MockOutlierDataset, NoisyMNISTDataset

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        """Set up common parameters for tests."""
        self.mock_config = {
            "data": {
                "dataset_type": "mock",
                "num_samples": 10,
                "max_points": 50,
                "num_outliers": 10,
                "batch_size": 2,
                "num_workers": 0,
                "pin_memory": False,
            }
        }
        self.mnist_config = {
            "data": {
                "dataset_type": "mnist",
                "num_samples": 5,
                "max_points": 100,
                "num_outliers": 25,
                "batch_size": 1,
                "num_workers": 0,
                "pin_memory": False,
            }
        }
        self.invalid_config = {"data": {"dataset_type": "unknown"}}

    def test_get_dataset_factory(self):
        """Test if get_dataset returns the correct dataset instance."""
        mock_dataset = get_dataset(self.mock_config)
        self.assertIsInstance(mock_dataset, MockOutlierDataset)

        try:
            mnist_dataset = get_dataset(self.mnist_config)
            self.assertIsInstance(mnist_dataset, PreloadedOutlierMNIST)
        except Exception as e:
            self.skipTest(f"Skipping MNIST test due to potential download issue: {e}")

        with self.assertRaises(ValueError):
            get_dataset(self.invalid_config)

    def test_mock_outlier_dataset(self):
        """Test if MockOutlierDataset works correctly."""
        max_points = 60
        num_outliers = 15
        dataset = MockOutlierDataset(num_samples=20, max_points=max_points, num_outliers=num_outliers)
        self.assertEqual(len(dataset), 20)

        noisy_pc, labels, clean_pc = dataset[0]
        self.assertEqual(noisy_pc.shape, (max_points, 2))
        self.assertEqual(clean_pc.shape, (max_points, 2))
        self.assertEqual(labels.shape, (max_points,))
        self.assertIsInstance(noisy_pc, torch.Tensor)
        self.assertIsInstance(clean_pc, torch.Tensor)
        self.assertIsInstance(labels, torch.Tensor)

        # Check number of inliers and outliers
        self.assertEqual(torch.sum(labels == 1), num_outliers)
        self.assertEqual(torch.sum(labels == 0), max_points - num_outliers)

        # Check that clean_pc contains only the inlier points (padded)
        self.assertEqual(torch.sum(torch.abs(clean_pc).sum(dim=-1) > 1e-6), max_points - num_outliers)



    def test_empty_foreground_strict_raises(self):
        """Empty MNIST foreground must hard-fail unless opt-in fallback is enabled."""
        dataset = NoisyMNISTDataset(
            train=False,
            num_samples=1,
            max_points=20,
            num_outliers=5,
            preload=False,
            deterministic=True,
            noise_seed=42,
            allow_empty_cloud_fallback=False,
        )
        dataset.images = torch.zeros((1, 28, 28), dtype=torch.uint8)
        with self.assertRaises(RuntimeError):
            dataset[0]

    def test_empty_foreground_opt_in_fallback(self):
        """Opt-in fallback reproduces legacy random-point behavior."""
        dataset = NoisyMNISTDataset(
            train=False,
            num_samples=1,
            max_points=20,
            num_outliers=5,
            preload=False,
            deterministic=True,
            noise_seed=42,
            allow_empty_cloud_fallback=True,
        )
        dataset.images = torch.zeros((1, 28, 28), dtype=torch.uint8)
        data, labels, clean_pc = dataset[0]
        self.assertEqual(data.shape, (25, 2))
        self.assertEqual(labels.shape, (25,))
        self.assertEqual(clean_pc.shape, (20, 2))

    def test_mnist_outlier_dataset_loading(self):
        """Test if PreloadedOutlierMNIST initializes and returns correct data."""
        num_samples = 2
        max_points = 100
        num_outliers = 20
        try:
            dataset = PreloadedOutlierMNIST(
                train=False, num_samples=num_samples, max_points=max_points, num_outliers=num_outliers
            )
        except Exception as e:
            self.skipTest(f"Skipping MNIST test due to potential download issue: {e}")
            return

        self.assertEqual(len(dataset), num_samples)

        noisy_pc, labels, clean_pc = dataset[0]

        self.assertEqual(noisy_pc.shape, (max_points, 2))
        self.assertEqual(clean_pc.shape, (max_points, 2))
        self.assertEqual(labels.shape, (max_points,))
        self.assertIsInstance(noisy_pc, torch.Tensor)
        self.assertIsInstance(clean_pc, torch.Tensor)
        self.assertIsInstance(labels, torch.Tensor)

        # Check number of inliers and outliers
        self.assertEqual(torch.sum(labels == 1), num_outliers)
        self.assertEqual(torch.sum(labels == 0), max_points - num_outliers)

if __name__ == "__main__":
    unittest.main()