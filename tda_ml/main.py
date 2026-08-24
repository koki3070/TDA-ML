import argparse
import csv
import json
import logging
import os

import torch

from tda_ml.checkpoint_io import extract_model_state_dict, load_torch_checkpoint
from tda_ml.config import deep_update, load_config, model_kwargs_from_config, default_project_root
from tda_ml.model_selection import (
    compute_epoch_selection,
    selection_settings_from_config,
)
from tda_ml.models import AnisotropicOutlierClassifier
from tda_ml.run_setup import (
    build_dataloaders,
    configure_torch_runtime,
    resolve_dataloader_settings,
    resolve_device,
)
from tda_ml.run_paths import build_run_dir
from tda_ml.runtime_profile import build_runtime_profile
from tda_ml.seed_utils import set_global_seed
from tda_ml.preflight import preflight_training_config
from tda_ml.reproducibility import (
    RUN_STATUS_COMPLETED,
    RUN_STATUS_FAILED,
    RUN_STATUS_NOT_RUN,
    RUN_STATUS_RUNNING,
    build_dbscan_eval_manifest_fields,
    build_reproducibility_manifest_fields,
)
from tda_ml.supervised_diagnostics import (
    git_revision,
    run_abort_diagnostics,
    should_early_abort,
    write_abort_report,
)
from tda_ml.trainer import Trainer

logger = logging.getLogger(__name__)


def main(config_name=None, config=None, trial=None, config_overrides=None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if config is None:
        if config_name is None:
            raise ValueError("Either config_name or config must be provided")
        config = load_config(config_name)
        
    if config_overrides:
        deep_update(config, config_overrides)
    
    logger.info("Loaded config: %s", config["meta"].get("config_id", "unknown"))

    import datetime

    device = resolve_device(config)
    config_id = config["meta"].get("config_id", "unknown")
    run_dir = run_slug = run_stamp = None
    log_dir = None
    manifest_path = None

    if "outputs" in config:
        run_dir, run_slug, run_stamp = build_run_dir(config)
        config["outputs"]["log_dir"] = os.path.join(run_dir, "logs")
        config["outputs"]["image_dir"] = os.path.join(run_dir, "images")
        log_dir = config["outputs"]["log_dir"]
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(config["outputs"]["image_dir"], exist_ok=True)
        logger.info("Project structure created at: %s", run_dir)

        data_cfg = config["data"]
        if "seed" not in data_cfg:
            raise ValueError("data.seed must be set explicitly; refusing silent seed=42")
        seed = int(data_cfg["seed"])
        topo_cfg = (config.get("model") or {}).get("topology_loss") or {}
        if "distance_backend" not in topo_cfg:
            raise ValueError(
                "model.topology_loss.distance_backend must be set before manifest write; "
                "refusing silent default (training PD requires ellphi)"
            )
        selection = (config.get("training", {}).get("selection") or {})
        if "metric" not in selection:
            raise ValueError(
                "training.selection.metric must be set explicitly; "
                "refusing silent val_topo default in run manifest"
            )
        init_checkpoint = config.get("init_checkpoint")
        if init_checkpoint and not os.path.exists(init_checkpoint):
            raise FileNotFoundError(f"Initial checkpoint not found: {init_checkpoint}")
        dataset_type = str(data_cfg.get("dataset_type", "")).strip().lower()
        if not dataset_type:
            raise ValueError(
                "data.dataset_type must be set explicitly (mnist|thin_rings); "
                "refusing silent MNIST default in run manifest"
            )
        if "noise_std" not in data_cfg:
            raise ValueError(
                "data.noise_std must be set explicitly; refusing silent default in run manifest"
            )
        if dataset_type == "thin_rings":
            from tda_ml.ring_dataset import ring_kwargs_from_config

            outlier_mode = "ring_radial"
            data_outliers = {
                "dataset_type": dataset_type,
                "outlier_mode": outlier_mode,
                "noise_std": float(data_cfg["noise_std"]),
                "num_outliers": int(data_cfg["num_outliers"]),
                **ring_kwargs_from_config(data_cfg),
            }
        else:
            if "outlier_mode" not in data_cfg:
                raise ValueError(
                    "data.outlier_mode must be set explicitly (uniform|local_pca_tangent); "
                    "refusing silent uniform default in run manifest"
                )
            outlier_mode = str(data_cfg["outlier_mode"]).strip().lower()
            data_outliers = {
                "dataset_type": dataset_type,
                "outlier_mode": outlier_mode,
                "noise_std": float(data_cfg["noise_std"]),
                "num_outliers": int(data_cfg["num_outliers"]),
            }
        if outlier_mode == "local_pca_tangent":
            for key in (
                "tangent_pca_k",
                "tangent_offset_min",
                "tangent_offset_max",
                "tangent_angle_jitter_deg",
                "tangent_stroke_clearance",
                "tangent_direction",
            ):
                if key not in data_cfg:
                    raise ValueError(
                        f"data.{key} must be set explicitly for outlier_mode=local_pca_tangent"
                    )
            data_outliers.update(
                tangent_pca_k=int(data_cfg["tangent_pca_k"]),
                tangent_offset_min=float(data_cfg["tangent_offset_min"]),
                tangent_offset_max=float(data_cfg["tangent_offset_max"]),
                tangent_angle_jitter_deg=float(data_cfg["tangent_angle_jitter_deg"]),
                tangent_stroke_clearance=float(data_cfg["tangent_stroke_clearance"]),
                tangent_direction=str(data_cfg["tangent_direction"]),
            )
        manifest = {
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "source_revision": git_revision(),
            "config_id": config_id,
            "run_slug": run_slug,
            "run_stamp": run_stamp,
            "command_entry": "tda_ml.main",
            "seed": seed,
            "epochs_planned": config["training"]["epochs"],
            "distance_backend": str(topo_cfg["distance_backend"]).lower().strip(),
            "data_outliers": data_outliers,
            "checkpoint_selection": selection["metric"],
            "early_abort": config.get("training", {}).get("early_abort"),
            "run_dir": run_dir,
            "run_status": "pending",
            "final_status": "pending",
            "preflight_status": "pending",
            "fallback_status": "none",
            "fallbacks": [],
            "reproducibility": build_reproducibility_manifest_fields(
                config, project_root=default_project_root()
            ),
        }
        if config.get("evaluation"):
            manifest["dbscan_eval"] = build_dbscan_eval_manifest_fields(config)
        manifest.update(config.get("_manifest_extras") or {})
        config.setdefault("_manifest", {})
        manifest_path = os.path.join(log_dir, "run_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=True, indent=2)
        logger.info("Run manifest (pending) saved: %s", manifest_path)

        try:
            preflight_training_config(config, project_root=default_project_root())
        except Exception as exc:
            manifest["run_status"] = RUN_STATUS_NOT_RUN
            manifest["final_status"] = RUN_STATUS_NOT_RUN
            manifest["preflight_status"] = "failed"
            manifest["preflight_error"] = str(exc)
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=True, indent=2)
            raise
        manifest["preflight_status"] = "passed"
        manifest["run_status"] = RUN_STATUS_RUNNING
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=True, indent=2)
    else:
        preflight_training_config(config, project_root=default_project_root())
        data_cfg = config["data"]
        if "seed" not in data_cfg:
            raise ValueError("data.seed must be set explicitly; refusing silent seed=42")
        seed = int(data_cfg["seed"])
        manifest = {"config_id": config_id, "seed": seed}
        config.setdefault("_manifest", manifest)

    def _mark_failed(exc: BaseException) -> None:
        if manifest_path is None:
            return
        manifest["final_status"] = RUN_STATUS_FAILED
        manifest["run_status"] = RUN_STATUS_FAILED
        manifest["error"] = str(exc)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=True, indent=2)

    try:
        data_cfg = config["data"]
        seed = int(data_cfg["seed"])
        deterministic_algorithms = bool(
            config.get("reproducibility", {}).get("deterministic_algorithms", False)
        )
        set_global_seed(seed, deterministic_algorithms=deterministic_algorithms)
        logger.info(
            "Global seed initialized: seed=%s, deterministic_algorithms=%s",
            seed,
            deterministic_algorithms,
        )
        loader_settings = resolve_dataloader_settings(config, device)
        configure_torch_runtime(config, device)
        logger.info(
            "DataLoader settings: workers=%s, pin_memory=%s, persistent_workers=%s, prefetch_factor=%s",
            loader_settings.num_workers,
            loader_settings.pin_memory,
            loader_settings.persistent_workers,
            loader_settings.prefetch_factor,
        )

        data_loader, val_loader, test_loader = build_dataloaders(
            config, seed, loader_settings
        )

        model = AnisotropicOutlierClassifier(**model_kwargs_from_config(config))
        model.to(device)

        trainer = Trainer(model, config, device=device)

        init_checkpoint = config.get("init_checkpoint")
        if init_checkpoint:
            logger.info("Loading initial weights from %s", init_checkpoint)
            checkpoint = load_torch_checkpoint(init_checkpoint, map_location=device)
            model.load_state_dict(extract_model_state_dict(checkpoint), strict=True)

        log_dir = config["outputs"]["log_dir"]
        metrics_path = os.path.join(log_dir, "metrics.csv")
        runtime_profile = build_runtime_profile(
            config=config,
            device=device,
            num_workers=loader_settings.num_workers,
            pin_memory=loader_settings.pin_memory,
            persistent_workers=loader_settings.persistent_workers,
            prefetch_factor=loader_settings.prefetch_factor,
            use_amp_effective=trainer.use_amp,
            amp_dtype_effective=str(trainer.amp_dtype).replace("torch.", ""),
        )
        runtime_profile_path = os.path.join(log_dir, "runtime_profile.json")
        with open(runtime_profile_path, "w", encoding="utf-8") as f:
            json.dump(runtime_profile, f, ensure_ascii=True, indent=2)
        logger.info("Runtime profile saved: %s", runtime_profile_path)

        if manifest_path is not None:
            impl = config.get("_manifest", {}).get("distance_backend_impl")
            if impl is not None:
                manifest["distance_backend_impl"] = impl
            manifest["final_status"] = "running"
            manifest["run_status"] = RUN_STATUS_RUNNING
            if config.get("_manifest", {}).get("fallbacks"):
                manifest["fallbacks"] = config["_manifest"]["fallbacks"]
                manifest["fallback_status"] = config["_manifest"].get(
                    "fallback_status", "recorded"
                )
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=True, indent=2)
    except Exception as exc:
        _mark_failed(exc)
        raise

    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            'epoch', 'train_loss', 'train_class_loss', 'train_topo_loss',
            'train_aniso_loss', 'train_size_loss', 'val_loss', 'val_recall',
            'val_mcc', 'val_aniso', 'val_size', 'val_topo_loss', 'val_wdist',
            'sel_eps', 'sel_min_samples',
        ])

    # --- Training Loop ---
    epochs = config['training']['epochs']
    save_every = config['outputs'].get('save_every', 1)
    best_val_mcc = -1.0
    early_abort_cfg = config.get("training", {}).get("early_abort", {})
    metrics_history: list[dict] = []
    final_status = "completed"
    abort_report_path = None

    # --- Model-selection (checkpoint) policy (see tda_ml.model_selection) ---
    sel_settings = selection_settings_from_config(config)
    sel_metric = sel_settings.metric
    best_sel_value = float("inf") if sel_settings.minimize else -1.0

    for epoch in range(1, epochs + 1):
        try:
            # res returns (avg_loss, class_loss, topo_loss, aniso_loss, size_loss, ...)
            res = trainer.train_epoch(data_loader, epoch)
        except Exception as exc:
            if manifest_path is not None:
                manifest["final_status"] = RUN_STATUS_FAILED
                manifest["run_status"] = RUN_STATUS_FAILED
                manifest["failure_type"] = type(exc).__name__
                manifest["failure_error"] = str(exc)
                if metrics_history:
                    manifest["last_completed_epoch"] = metrics_history[-1]["epoch"]
                with open(manifest_path, "w", encoding="utf-8") as f:
                    json.dump(manifest, f, ensure_ascii=True, indent=2)
            raise
        val_res = trainer.validate(val_loader)

        val_mcc = val_res[4] # MCC is at index 4
        val_topo_loss = val_res[7]
        train_mcc = res[10]
        val_recall = val_res[1]
        print(
            f"Epoch {epoch}: Val MCC={val_mcc:.4f}, Val topo={val_topo_loss:.4f}, "
            f"Aniso={val_res[5]:.4f}"
        )

        metrics_history.append(
            {
                "epoch": epoch,
                "val_mcc": float(val_mcc),
                "train_mcc": float(train_mcc),
                "val_recall": float(val_recall),
                "val_specificity": float(val_res[2]),
                "val_loss": float(val_res[0]),
                "train_loss": float(res[0]),
                "val_size": float(val_res[6]),
                "val_aniso": float(val_res[5]),
                "val_topo_loss": float(val_topo_loss),
            }
        )

        # Track best threshold MCC for early-abort diagnostics (independent of
        # the checkpoint-selection metric).
        if val_mcc > best_val_mcc:
            best_val_mcc = val_mcc

        # --- Checkpoint selection ---
        sel = compute_epoch_selection(
            sel_settings,
            epoch=epoch,
            epochs=epochs,
            val_mcc=val_mcc,
            val_loss=val_res[0],
            val_topo_loss=val_topo_loss,
            model=model,
            val_loader=val_loader,
            device=device,
            config=config,
        )
        sel_value = sel.value
        sel_eps = sel.eps
        sel_min_samples = sel.min_samples
        sel_wdist = sel.wdist

        if sel_value is not None:
            is_better = (
                sel_value < best_sel_value if sel_settings.minimize else sel_value > best_sel_value
            )
            if is_better:
                best_sel_value = sel_value
                best_model_path = os.path.join(run_dir, 'best_model.pth')
                ckpt = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_mcc': float(val_mcc),
                    'selection_metric': sel_metric,
                    'selection_value': float(sel_value),
                }
                if sel_metric == "val_topo":
                    ckpt['val_topo_loss'] = float(sel_value)
                if sel_eps is not None:
                    ckpt['dbscan_eps'] = float(sel_eps)
                    ckpt['dbscan_min_samples'] = int(sel_min_samples)
                    ckpt['val_wdist'] = float(sel_wdist)
                    ckpt['val_mcc_dbscan'] = float(sel.mcc_dbscan)
                    hp_path = os.path.join(log_dir, 'dbscan_hparams_train.json')
                    with open(hp_path, 'w', encoding='utf-8') as f:
                        json.dump(
                            {
                                'eps': float(sel_eps),
                                'min_samples': int(sel_min_samples),
                                'backend': sel_settings.backend,
                                'epoch': epoch,
                                'val_wdist': float(sel_wdist),
                                'val_mcc_dbscan': float(sel.mcc_dbscan),
                                'selection_metric': sel_metric,
                            },
                            f,
                            ensure_ascii=True,
                            indent=2,
                        )
                torch.save(ckpt, best_model_path)
                print(f"Saved best model ({sel_metric}={sel_value:.5f}) to {best_model_path}")

        if epoch % save_every == 0:
            checkpoint_path = os.path.join(run_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_mcc': val_mcc,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

        with open(metrics_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, res[0], res[1], res[2], res[3], res[4],
                val_res[0], val_res[1], val_res[4], val_res[5], val_res[6],
                val_res[7],
                '' if sel_wdist is None else sel_wdist,
                '' if sel_eps is None else sel_eps,
                '' if sel_min_samples is None else sel_min_samples,
            ])

        do_abort, abort_reason = should_early_abort(
            epoch=epoch,
            best_val_mcc=best_val_mcc,
            val_recall=val_recall,
            val_mcc=val_mcc,
            train_mcc=train_mcc,
            early_abort_cfg=early_abort_cfg,
        )
        if do_abort:
            abort_ckpt = os.path.join(run_dir, "abort_checkpoint.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_mcc": val_mcc,
                    "best_val_mcc": best_val_mcc,
                    "abort_reason": abort_reason,
                },
                abort_ckpt,
            )
            logger.warning("Early abort at epoch %s: %s", epoch, abort_reason)
            report = run_abort_diagnostics(
                trainer=trainer,
                model=model,
                val_loader=val_loader,
                device=device,
                epoch=epoch,
                metrics_history=metrics_history,
                abort_reason=abort_reason,
            )
            abort_report_path = write_abort_report(log_dir, report)
            logger.warning("Abort diagnostics: %s", abort_report_path)
            manifest["final_status"] = "early-aborted"
            manifest["run_status"] = RUN_STATUS_FAILED
            manifest["abort_epoch"] = epoch
            manifest["abort_reason"] = abort_reason
            manifest["abort_report"] = str(abort_report_path)
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=True, indent=2)
            final_status = "early-aborted"
            break

    if final_status == "completed":
        manifest["final_status"] = "completed"
        manifest["run_status"] = RUN_STATUS_COMPLETED
        if config.get("_manifest", {}).get("fallbacks"):
            manifest["fallback_status"] = "recorded"
            manifest["fallbacks"] = config["_manifest"]["fallbacks"]
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=True, indent=2)

    if final_status == "early-aborted":
        return {
            "val_gmean": val_res[3],
            "val_mcc": val_res[4],
            "run_dir": run_dir,
            "status": final_status,
            "best_val_mcc": best_val_mcc,
            "abort_report": str(abort_report_path) if abort_report_path else None,
        }

    final_model_path = os.path.join(run_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final model to {final_model_path}")

    test_res = trainer.validate(test_loader)
    (
        test_loss,
        test_recall,
        test_specificity,
        test_gmean,
        test_mcc,
        test_aniso,
        test_size,
        test_topo_loss,
    ) = test_res
    test_metrics = {
        "test_loss": test_loss,
        "test_recall": test_recall,
        "test_specificity": test_specificity,
        "test_gmean": test_gmean,
        "test_mcc": test_mcc,
        "test_aniso": test_aniso,
        "test_size": test_size,
        "test_topo_loss": test_topo_loss,
    }
    test_metrics_path = os.path.join(log_dir, "test_metrics.json")
    with open(test_metrics_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, ensure_ascii=True, indent=2)
    logger.info(
        "Test (held-out): MCC=%.4f, recall=%.4f, specificity=%.4f, G-mean=%.4f, loss=%.4f",
        test_mcc,
        test_recall,
        test_specificity,
        test_gmean,
        test_loss,
    )
    logger.info("Test metrics saved: %s", test_metrics_path)

    return {
        "val_gmean": val_res[3],
        "val_mcc": val_res[4],
        "run_dir": run_dir,
        "status": final_status,
        "best_val_mcc": best_val_mcc,
        "abort_report": None,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='dev', help='Config path')
    parser.add_argument('--init_checkpoint', type=str, default=None, help='Path to initial checkpoint')
    args = parser.parse_args()
    
    config_overrides = {}
    if args.init_checkpoint:
        config_overrides['init_checkpoint'] = args.init_checkpoint
        
    main(config_name=args.config, config_overrides=config_overrides) 
