import logging
import os

import torch
from torch.optim import Adam
from torch_topological.nn import VietorisRipsComplex
from tda_ml.teacher_pd import compute_clean_teacher_batch
from tda_ml.reproducibility import (
    assert_ellphi_differentiable_available,
    assert_loss_config_explicit,
    record_fallback,
    reproducibility_settings,
)
from tda_ml.losses import (
    ClassificationLoss, 
    TopologicalLoss, 
    SizeRegularizationLoss, 
    AnisotropyPenaltyLoss,
    MinBRegularizationLoss,
)
from tda_ml.metrics import compute_recall_specificity_gmean_mcc
from tda_ml.visualization import visualize
import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, model, config, device=None, trial=None):
        self.model = model
        self.config = config
        self.trial = trial
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        self.model.to(self.device)

        self.optimizer = Adam(model.parameters(), lr=config['training']['lr'])
        self._init_amp(config)
        self._init_losses(config)

        self.visualize_every = config['training'].get('visualize_every', 5)
        self.output_dir = config['outputs']['image_dir']
        self.log_dir = config['outputs']['log_dir']

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        self.fixed_indices = None
        self.threshold = config['model'].get('threshold', 0.5)

        self.vr_complex = VietorisRipsComplex(dim=1)

        self.warmup_epochs = config.get('training', {}).get('warmup_epochs', 0)

        self._val_aniso_accum = 0.0
        self._val_size_accum = 0.0

    def _init_amp(self, config):
        """Mixed-precision settings (AMP autocast / GradScaler)."""
        training_cfg = config.get('training', {})
        perf_cfg = config.get('performance', {})
        self.use_amp = (
            self.device.type == "cuda"
            and bool(training_cfg.get("use_amp", perf_cfg.get("use_amp", True)))
        )
        self.autocast_device_type = "cuda" if self.device.type == "cuda" else "cpu"
        amp_dtype_name = str(training_cfg.get("amp_dtype", perf_cfg.get("amp_dtype", "float16"))).lower()
        self.amp_dtype = torch.float16 if amp_dtype_name == "float16" else torch.bfloat16
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    def _init_losses(self, config):
        """Parse loss weights from ``loss.*``; legacy ``training.lambda_*`` only when opted in."""
        assert_loss_config_explicit(config)
        training_cfg = config.get('training', {})
        loss_cfg = config.get('loss', {})
        allow_legacy = reproducibility_settings(config)["allow_legacy_loss_keys"]

        def _loss_or_legacy(key: str, legacy_key: str, default: float) -> float:
            if key in loss_cfg:
                return loss_cfg[key]
            if allow_legacy and legacy_key in training_cfg:
                return training_cfg[legacy_key]
            raise ValueError(
                f"loss.{key} must be set explicitly; legacy training.{legacy_key} fallback "
                "is disabled. Set reproducibility.allow_legacy_loss_keys=true to opt in."
            )

        self._repro = reproducibility_settings(config)
        self._manifest_ref = config.setdefault("_manifest", {})

        self.lambda_class = _loss_or_legacy("w_class", "lambda_class", 1.0)
        self.lambda_topo = _loss_or_legacy("w_topo", "lambda_topo", 0.1)
        self.lambda_aniso = _loss_or_legacy("w_aniso", "lambda_aniso", 0.01)
        if "w_min_b" in loss_cfg:
            self.lambda_min_b = float(loss_cfg["w_min_b"])
        elif allow_legacy and "lambda_min_b" in training_cfg:
            self.lambda_min_b = float(training_cfg["lambda_min_b"])
        else:
            self.lambda_min_b = 0.0
        self.min_b_target = float(
            loss_cfg.get('min_b_target', training_cfg.get('min_b_target', 0.2))
        )
        self.topo_loss_max_points = training_cfg.get(
            'topo_loss_max_points', loss_cfg.get('topo_loss_max_points')
        )
        if self.topo_loss_max_points is not None:
            self.topo_loss_max_points = int(self.topo_loss_max_points)
        
        size_default = _loss_or_legacy("w_size", "lambda_size", 0.1)
        self.lambda_major = training_cfg.get("lambda_major", size_default)
        self.lambda_minor = training_cfg.get("lambda_minor", size_default)

        self.aniso_mode = loss_cfg.get("aniso_mode", training_cfg.get("aniso_mode", "linear"))
        self.aniso_barrier_threshold = float(
            loss_cfg.get(
                "aniso_barrier_threshold",
                training_cfg.get("barrier_threshold", 6.0),
            )
        )
        self.size_mode = str(
            loss_cfg.get("size_mode", training_cfg.get("size_mode", "quadratic"))
        ).strip().lower()
        self.size_barrier_radius = float(
            loss_cfg.get(
                "size_barrier_radius",
                training_cfg.get("size_barrier_radius", 1.5),
            )
        )
        self.size_ref = float(
            loss_cfg.get("size_ref", training_cfg.get("size_ref", 1.34))
        )
        self.size_power = float(
            loss_cfg.get("size_power", training_cfg.get("size_power", 1.5))
        )
        self.size_softplus_beta = float(
            loss_cfg.get(
                "size_softplus_beta",
                training_cfg.get("size_softplus_beta", 8.0),
            )
        )
        logger.info(
            "Anisotropy penalty mode: %s (barrier_threshold=%s)",
            self.aniso_mode,
            self.aniso_barrier_threshold,
        )
        if self.size_mode == "barrier":
            logger.info(
                "Size penalty mode: barrier (radius=%s, w_size=%s)",
                self.size_barrier_radius,
                size_default,
            )
        elif self.size_mode == "power":
            logger.info(
                "Size penalty mode: power (ref=%s, gamma=%s, w_size=%s)",
                self.size_ref,
                self.size_power,
                size_default,
            )
        elif self.size_mode == "softplus":
            logger.info(
                "Size penalty mode: softplus (ref=%s, beta=%s, w_size=%s)",
                self.size_ref,
                self.size_softplus_beta,
                size_default,
            )

        pos_weight_val = config.get('loss', {}).get('pos_weight', 1.0)
        pos_weight = torch.tensor([pos_weight_val], device=self.device) if pos_weight_val != 1.0 else None

        # Initialize Losses
        self.class_loss_fn = ClassificationLoss(pos_weight=pos_weight)
        _topo = config.get("model", {}).get("topology_loss", {})
        self.distance_backend = _topo.get("distance_backend", "mahalanobis")
        self.ellphi_differentiable = _topo.get("ellphi_differentiable", True)
        self.prob_weighting = bool(_topo.get("prob_weighting", True))
        # Filtration-unit alignment for the topology loss (see TopologicalLoss).
        # Legacy knob `training.topo_eps_scale` (v73=0.7022); also accept loss.topo_eps_scale.
        self.topo_eps_scale = float(
            loss_cfg.get("topo_eps_scale", training_cfg.get("topo_eps_scale", 1.0))
        )
        self.topo_scale_mode = str(
            loss_cfg.get("topo_scale_mode", training_cfg.get("topo_scale_mode", "fixed"))
        ).strip().lower()
        self.teacher_mode = str(
            loss_cfg.get("teacher_mode", training_cfg.get("teacher_mode", "euclidean"))
        ).strip().lower()
        self.teacher_local_pca_k = int(
            loss_cfg.get(
                "teacher_local_pca_k",
                training_cfg.get("teacher_local_pca_k", 10),
            )
        )
        self.teacher_local_pca_normalize_axes = bool(
            loss_cfg.get(
                "teacher_local_pca_normalize_axes",
                training_cfg.get("teacher_local_pca_normalize_axes", True),
            )
        )
        logger.info(
            "Topological distance backend: %s%s (prob_weighting=%s, scale_mode=%s, eps_scale=%s, teacher_mode=%s%s)",
            self.distance_backend,
            (
                f" (ellphi_differentiable={self.ellphi_differentiable})"
                if self.distance_backend == "ellphi"
                else ""
            ),
            self.prob_weighting,
            self.topo_scale_mode,
            self.topo_eps_scale,
            self.teacher_mode,
            (
                f", teacher_local_pca_normalize_axes={self.teacher_local_pca_normalize_axes}"
                if self.teacher_mode == "local_pca"
                else ""
            ),
        )
        if self.distance_backend == "ellphi":
            impl = assert_ellphi_differentiable_available(
                ellphi_differentiable=self.ellphi_differentiable
            )
            self._manifest_ref["distance_backend_impl"] = impl
        self.topo_loss_fn = TopologicalLoss(
            weight=self.lambda_topo,
            distance_backend=self.distance_backend,
            ellphi_differentiable=self.ellphi_differentiable,
            prob_weighting=self.prob_weighting,
            eps_scale=self.topo_eps_scale,
            scale_mode=self.topo_scale_mode,
            max_points=self.topo_loss_max_points,
            strict_topo_samples=self._repro["strict_topo_samples"],
            manifest_ref=self._manifest_ref,
        )
        self.size_loss_fn = SizeRegularizationLoss(
            w_major=self.lambda_major,
            w_minor=self.lambda_minor,
            mode=self.size_mode,
            barrier_radius=self.size_barrier_radius,
            size_ref=self.size_ref,
            size_power=self.size_power,
            size_softplus_beta=self.size_softplus_beta,
        )
        self.aniso_loss_fn = AnisotropyPenaltyLoss(
            weight=self.lambda_aniso,
            mode=self.aniso_mode,
            barrier_threshold=self.aniso_barrier_threshold,
        )
        self.min_b_loss_fn = MinBRegularizationLoss(
            weight=self.lambda_min_b,
            target=self.min_b_target,
        )
        if self.lambda_min_b > 0:
            logger.info(
                "Min-B regularization: lambda=%s target=%s",
                self.lambda_min_b,
                self.min_b_target,
            )
        if self.topo_loss_max_points is not None:
            logger.info("Topological loss subsampling: max_points=%s", self.topo_loss_max_points)

    def _compute_clean_pd_info(self, clean_pc: torch.Tensor):
        """Compute clean (teacher) persistence diagrams without gradient tracking."""
        return compute_clean_teacher_batch(
            clean_pc,
            self.vr_complex,
            teacher_mode=self.teacher_mode,
            distance_backend=self.distance_backend,
            ellphi_differentiable=self.ellphi_differentiable,
            local_pca_k=self.teacher_local_pca_k,
            local_pca_normalize_axes=self.teacher_local_pca_normalize_axes,
            max_points=self.topo_loss_max_points,
            need_clean_scales=(self.topo_scale_mode == "median"),
        )

    def train_epoch(self, data_loader, epoch):
        self.model.train()
        total_loss = 0
        total_class_loss = 0
        total_topo_loss = 0
        total_aniso_loss = 0
        total_size_loss = 0
        total_min_b_loss = 0
        steps_completed = 0

        all_train_preds = []
        all_train_labels = []

        pbar = tqdm.tqdm(data_loader, desc=f"Epoch {epoch}")

        if self.fixed_indices is None:
            import random
            self.fixed_indices = random.sample(range(len(data_loader.dataset)), 3)

        for i, (data, labels, clean_pc) in enumerate(pbar):
            data = data.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            clean_pc = clean_pc.to(self.device, non_blocking=True)

            if self.config.get('training', {}).get('rotation_augmentation', False):
                theta = torch.rand(1, device=self.device) * 2 * 3.141592653589793
                cos_t, sin_t = torch.cos(theta), torch.sin(theta)
                rotation_matrix = torch.stack([
                    torch.stack([cos_t, -sin_t], dim=-1),
                    torch.stack([sin_t, cos_t], dim=-1)
                ], dim=-2).squeeze(0)
                
                data = torch.matmul(data, rotation_matrix.T)
                clean_pc = torch.matmul(clean_pc, rotation_matrix.T)

            self.optimizer.zero_grad(set_to_none=True)

            clean_pd_info = None
            clean_scales = None
            if self.lambda_topo > 0 and epoch > self.warmup_epochs:
                # Topological target PD does not require autograd; keep it out of AMP/grad graph.
                clean_pd_info, clean_scales = self._compute_clean_pd_info(clean_pc)

            with torch.amp.autocast(
                device_type=self.autocast_device_type,
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                logits, params = self.model(data)
                class_loss = self.class_loss_fn(logits, labels)

                topo_loss = torch.tensor(0.0, device=self.device)
                if clean_pd_info is not None:
                    topo_loss = self.topo_loss_fn(
                        data, params, logits, clean_pd_info, clean_scales=clean_scales
                    )

                # Regularization Losses (Size and Anisotropy only, as per slides)
                if epoch > self.warmup_epochs:
                    size_loss = self.size_loss_fn(params)
                    aniso_loss = self.aniso_loss_fn(params)
                    min_b_loss = self.min_b_loss_fn(params)
                else:
                    size_loss = torch.tensor(0.0, device=self.device)
                    aniso_loss = torch.tensor(0.0, device=self.device)
                    min_b_loss = torch.tensor(0.0, device=self.device)

                loss = topo_loss + aniso_loss + size_loss + min_b_loss
                if self.lambda_class > 0:
                    loss = loss + self.lambda_class * class_loss

            if torch.isnan(loss):
                if self._repro["allow_nan_batch_skip"]:
                    record_fallback(
                        self._manifest_ref,
                        "nan_batch_skip",
                        f"epoch={epoch} batch_index={i}",
                    )
                    logger.warning(
                        "NaN loss at epoch=%s batch_index=%s; skipping step (opt-in fallback)",
                        epoch,
                        i,
                    )
                    continue
                raise RuntimeError(
                    f"NaN loss at epoch={epoch} batch_index={i}; "
                    "set reproducibility.allow_nan_batch_skip=true to opt in to skipping."
                )

            steps_completed += 1
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['grad_clip_value'])
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['grad_clip_value'])
                self.optimizer.step()

            total_loss += loss.item()
            total_class_loss += class_loss.item()
            total_topo_loss += topo_loss.item()
            total_aniso_loss += aniso_loss.item()
            total_size_loss += size_loss.item()
            total_min_b_loss += min_b_loss.item()
            
            probs = torch.sigmoid(logits).squeeze(-1)
            preds = (probs > self.threshold).long()
            all_train_preds.extend(preds.cpu().numpy().flatten())
            all_train_labels.extend(labels.cpu().numpy().flatten())

            pbar.set_postfix(loss=f"{loss.item():.4f}", cls=f"{class_loss.item():.4f}", topo=f"{topo_loss.item():.4f}", aniso=f"{aniso_loss.item():.4f}", size=f"{size_loss.item():.4f}", min_b=f"{min_b_loss.item():.4f}")
            if i % 10 == 0:
                logger.debug(
                    "Step %s: loss=%.4f class=%.4f topo=%.4f aniso=%.4f size=%.4f",
                    i,
                    loss.item(),
                    class_loss.item(),
                    topo_loss.item(),
                    aniso_loss.item(),
                    size_loss.item(),
                )

        if steps_completed == 0 or not all_train_labels:
            raise RuntimeError(
                f"All training batches were skipped at epoch={epoch}; loss was NaN for every batch."
            )

        denom = steps_completed if steps_completed > 0 else 1
        avg_loss = total_loss / denom
        avg_class_loss = total_class_loss / denom
        avg_topo_loss = total_topo_loss / denom
        avg_aniso_loss = total_aniso_loss / denom
        avg_size_loss = total_size_loss / denom

        train_f1 = f1_score(all_train_labels, all_train_preds, zero_division=0)
        train_precision = precision_score(all_train_labels, all_train_preds, zero_division=0)
        train_recall = recall_score(all_train_labels, all_train_preds, zero_division=0)
        
        _, train_specificity, train_gmean, train_mcc = compute_recall_specificity_gmean_mcc(
            all_train_labels, all_train_preds
        )
        
        logger.info(
            "Epoch %s avg loss=%.4f (class=%.4f topo=%.4f) train F1=%.4f spec=%.4f "
            "G-mean=%.4f MCC=%.4f",
            epoch,
            avg_loss,
            avg_class_loss,
            avg_topo_loss,
            train_f1,
            train_specificity,
            train_gmean,
            train_mcc,
        )

        if epoch % self.visualize_every == 0:
             visualize(self.model, self.device, data_loader.dataset, epoch, output_dir=self.output_dir, title_prefix=self.config['meta'].get('config_id', 'train'), sample_indices=self.fixed_indices, threshold=self.threshold, backend=self.distance_backend)

        return (
            avg_loss,
            avg_class_loss,
            avg_topo_loss,
            avg_aniso_loss,
            avg_size_loss,
            train_f1,
            train_precision,
            train_recall,
            train_specificity,
            train_gmean,
            train_mcc,
        )

    def validate(self, data_loader):
        self.model.eval()
        total_loss = 0
        all_labels = []
        all_preds = []
        self._val_aniso_accum = 0.0
        self._val_size_accum = 0.0
        total_topo_loss = 0.0
        topo_steps = 0

        with torch.no_grad():
            for data, labels, clean_pc in data_loader:
                data = data.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                clean_pc = clean_pc.to(self.device, non_blocking=True)

                clean_pd_info = None
                clean_scales = None
                if self.lambda_topo > 0:
                    clean_pd_info, clean_scales = self._compute_clean_pd_info(clean_pc)

                with torch.amp.autocast(
                    device_type=self.autocast_device_type,
                    dtype=self.amp_dtype,
                    enabled=self.use_amp,
                ):
                    logits, params = self.model(data)
                    class_loss = self.class_loss_fn(logits, labels)
                    topo_loss = torch.tensor(0.0, device=self.device)
                    if clean_pd_info is not None:
                        topo_loss = self.topo_loss_fn(
                            data,
                            params,
                            logits,
                            clean_pd_info,
                            clean_scales=clean_scales,
                        )
                total_loss += (self.lambda_class * class_loss).item()
                if clean_pd_info is not None:
                    total_topo_loss += topo_loss.item()
                    topo_steps += 1
                
                aniso_loss = self.aniso_loss_fn(params)
                size_loss = self.size_loss_fn(params)
                
                probs = torch.sigmoid(logits).squeeze(-1)
                preds = (probs > self.threshold).long()

                all_labels.extend(labels.cpu().numpy().flatten())
                all_preds.extend(preds.cpu().numpy().flatten())
                
                self._val_aniso_accum += aniso_loss.item()
                self._val_size_accum += size_loss.item()

        num_batches = len(data_loader) if len(data_loader) > 0 else 1
        avg_loss = total_loss / num_batches

        avg_aniso = self._val_aniso_accum / num_batches
        avg_size = self._val_size_accum / num_batches
        if self.lambda_topo > 0 and topo_steps == 0:
            raise RuntimeError(
                "validate: w_topo>0 but topological loss was not computed for any batch"
            )
        avg_topo_loss = total_topo_loss / topo_steps

        recall = recall_score(all_labels, all_preds, zero_division=0)
        
        _, specificity, gmean, mcc = compute_recall_specificity_gmean_mcc(
            all_labels, all_preds
        )
        
        return avg_loss, recall, specificity, gmean, mcc, avg_aniso, avg_size, avg_topo_loss

