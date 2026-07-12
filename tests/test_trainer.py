import torch
from torch.utils.data import DataLoader
import unittest
import os
import shutil

from tda_ml.models import AnisotropicOutlierClassifier
from tda_ml.trainer import Trainer
from torch.utils.data import Dataset

class MockOutlierDataset(Dataset):
    def __init__(self, num_samples=10, max_points=30, num_outliers=5):
        self.num_samples = num_samples
        self.max_points = max_points
        self.num_outliers = num_outliers

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Create random data
        total_points = self.max_points + self.num_outliers
        data = torch.randn(total_points, 2)
        labels = torch.zeros(total_points, dtype=torch.long)
        labels[self.max_points:] = 1 # Outliers
        clean_pc = torch.randn(self.max_points, 2)
        return data, labels, clean_pc

class TestTrainer(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.model = AnisotropicOutlierClassifier().to(self.device)
        self.config = {
            'training': {
                'lr': 0.001,
                'lambda_class': 1.0,
                'lambda_topo': 0.1,
                'lambda_aniso': 0.01,
                'grad_clip_value': 1.0,
                'visualize_every': 10
            },
            'loss': {
                'w_class': 1.0,
                'w_topo': 0.1,
                'w_aniso': 0.01,
                'w_size': 0.1,
            },
            'model': {
                'topology_loss': {
                    'distance_backend': 'mahalanobis',
                }
            },
            'outputs': {
                'image_dir': 'test_outputs/images',
                'log_dir': 'test_outputs/logs'
            },
            'meta': {
                'config_id': 'test'
            }
        }
        
        dataset = MockOutlierDataset(num_samples=5, max_points=30, num_outliers=5)
        self.train_loader = DataLoader(dataset, batch_size=1)
        self.trainer = Trainer(self.model, self.config, self.device)

    def tearDown(self):
        if os.path.exists('test_outputs'):
            shutil.rmtree('test_outputs')

    def test_train_epoch(self):
        """
        Tests if the train_epoch function runs for one epoch and updates weights.
        """
        initial_params = [p.clone() for p in self.model.parameters()]
        
        avg_loss = self.trainer.train_epoch(self.train_loader, epoch=1)[0]
        
        params_updated = False
        for initial_p, final_p in zip(initial_params, self.model.parameters()):
            if not torch.equal(initial_p, final_p):
                params_updated = True
                break
        
        self.assertTrue(params_updated, "Model weights were not updated after a training step.")
        self.assertGreater(avg_loss, 0.0)

    def test_validate_reports_val_topo_loss(self):
        val_res = self.trainer.validate(self.train_loader)
        self.assertEqual(len(val_res), 8)
        self.assertGreater(val_res[7], 0.0)

    def test_w_class_zero_skips_classification_gradient(self):
        """w_class=0 and prob_weighting=false must not update the classification head."""
        self.config["loss"] = {
            "w_class": 0.0,
            "w_topo": 0.1,
            "w_aniso": 0.01,
            "w_size": 0.1,
        }
        self.config["model"]["topology_loss"]["prob_weighting"] = False
        trainer = Trainer(self.model, self.config, self.device)
        self.assertEqual(trainer.lambda_class, 0.0)

        cls_head_params = list(self.model.classification_head.parameters())
        cls_before = [p.clone() for p in cls_head_params]
        topo_before = [p.clone() for p in self.model.topology_head.parameters()]

        trainer.train_epoch(self.train_loader, epoch=1)

        cls_changed = any(
            not torch.equal(b, p) for b, p in zip(cls_before, cls_head_params)
        )
        topo_changed = any(
            not torch.equal(b, p) for b, p in zip(topo_before, self.model.topology_head.parameters())
        )
        self.assertFalse(cls_changed, "classification head should be frozen when w_class=0")
        self.assertTrue(topo_changed, "topology head should still update when w_class=0")

if __name__ == '__main__':
    unittest.main()