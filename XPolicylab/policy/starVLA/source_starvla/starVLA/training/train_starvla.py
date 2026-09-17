# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""
StarVLA’s trainer is built directly on native PyTorch + Accelerate + DeepSpeed, keeping the loop explicit and easy to hack.
Conventions:
1. Store runtime state in dicts where possible (simplifies data info, procesing info, config, etc).
2. Use multiple dataloaders to adapt heterogeneous data types / task mixtures.
3. Put each training strategy in its own `trainer_*.py` file (avoid large if‑else chains).
"""

# Standard Library
import argparse
import json
import os
import time
from pathlib import Path
from typing import Tuple

# Third-Party Libraries
import numpy as np
import torch
import torch.distributed as dist

# NPU support: import torch_npu and enable automatic CUDA→NPU mapping.
# On GPU-only environments this is a no-op (ImportError is silently ignored).
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except ImportError:
    pass

import wandb
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, get_scheduler

# Local Modules
from starVLA.dataloader import build_dataloader
from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.training.trainer_utils.config_tracker import AccessTrackedConfig, wrap_config
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils, build_param_lr_groups, setup_optimizer_and_scheduler, normalize_dotlist_args

deepspeed_plugin = DeepSpeedPlugin()

# The default ZeRO-2 communication buckets are 500M elements each.  With the
# 3-view SParkArena batch at BS=8, those buffers consume enough device memory
# to push the model just over an 80 GiB A800.  Allow the launcher to use
# smaller buckets without replacing Accelerate's generated DeepSpeed config.
deepspeed_bucket_size = os.environ.get("STARVLA_DEEPSPEED_BUCKET_SIZE")
if deepspeed_bucket_size is not None:
    try:
        deepspeed_bucket_size = int(deepspeed_bucket_size)
    except ValueError as exc:
        raise ValueError(
            "STARVLA_DEEPSPEED_BUCKET_SIZE must be a positive integer"
        ) from exc
    if deepspeed_bucket_size <= 0:
        raise ValueError("STARVLA_DEEPSPEED_BUCKET_SIZE must be a positive integer")

    zero_config = deepspeed_plugin.deepspeed_config.get("zero_optimization", {})
    zero_stage = int(zero_config.get("stage", 0))
    if zero_stage != 2:
        raise RuntimeError(
            "STARVLA_DEEPSPEED_BUCKET_SIZE is only supported for ZeRO-2; "
            f"resolved ZeRO stage is {zero_stage}"
        )
    zero_config["reduce_bucket_size"] = deepspeed_bucket_size
    zero_config["allgather_bucket_size"] = deepspeed_bucket_size

accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)
accelerator.print(accelerator.state)
effective_zero_config = deepspeed_plugin.deepspeed_config.get("zero_optimization", {})
accelerator.print(
    "Effective DeepSpeed ZeRO config: "
    f"stage={effective_zero_config.get('stage')}, "
    f"reduce_bucket_size={effective_zero_config.get('reduce_bucket_size')}, "
    f"allgather_bucket_size={effective_zero_config.get('allgather_bucket_size')}"
)

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize logger
logger = get_logger(__name__)


def load_fast_tokenizer():
    return AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)


def setup_directories(cfg) -> Path:
    """Create output directory and checkpoint directory."""
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)

    if not dist.is_initialized() or dist.get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)

    return output_dir


def prepare_data(cfg, accelerator, output_dir) -> DataLoader:
    """Prepare VLA training data."""
    logger.info(f"Creating VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}`")
    vla_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)

    accelerator.dataloader_config.dispatch_batches = False
    if dist.is_initialized():
        dist.barrier()
    return vla_train_dataloader


def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Set optimizer and scheduler."""
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
        fused=True,
    )

    if dist.is_initialized() and dist.get_rank() == 0:
        for group in optimizer.param_groups:
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")

    # Strip keys unknown to transformers' get_scheduler before passing kwargs.
    sched_kwargs = {k: v for k, v in cfg.trainer.scheduler_specific_kwargs.items()}
    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=sched_kwargs,
    )

    return optimizer, lr_scheduler


class VLATrainer(TrainerUtils):
    def __init__(self, cfg, model, vla_train_dataloader, optimizer, lr_scheduler, accelerator):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator

        self.completed_steps = 0
        self.total_batch_size = self._calculate_total_batch_size()

    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        # Save config snapshots upfront so that even if a later setup step
        # (ckpt load / DeepSpeed init / dataloader build) crashes, the
        # produced run dir is still introspectable / from_pretrained-able.
        self._save_initial_configs()

        self._init_checkpointing()
        self._adjust_lr_scheduler_for_resume()

        freeze_modules = (
            self.config.trainer.freeze_modules
            if (self.config and hasattr(self.config.trainer, "freeze_modules"))
            else None
        )
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)
        self.print_trainable_parameters(self.model)

        self.model, self.optimizer, self.vla_train_dataloader = self.setup_distributed_training(
            self.accelerator,
            self.model,
            self.optimizer,
            self.vla_train_dataloader,
        )

        self._init_wandb()

    def _calculate_total_batch_size(self):
        """Calculate global batch size."""
        total_batch_size = (
            self.config.datasets.vla_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )
        expected = os.environ.get("STARVLA_EXPECTED_GLOBAL_BATCH_SIZE")
        if expected is not None and total_batch_size != int(expected):
            raise RuntimeError(
                "Global batch-size contract failed: "
                f"per_device={self.config.datasets.vla_data.per_device_batch_size}, "
                f"processes={self.accelerator.num_processes}, "
                f"gradient_accumulation={self.accelerator.gradient_accumulation_steps}, "
                f"resolved={total_batch_size}, expected={expected}"
            )
        return total_batch_size

    def _init_wandb(self):
        """Initialize Weights & Biases, optionally failing closed."""
        self._wandb_enabled = False
        self._wandb_required = os.environ.get("WANDB_REQUIRE_ONLINE", "").lower() in {
            "1",
            "true",
            "yes",
        }
        wandb_mode = os.environ.get("WANDB_MODE", "online").lower()
        wandb_disabled = wandb_mode == "disabled" or os.environ.get(
            "WANDB_DISABLED", ""
        ).lower() in {
            "1",
            "true",
            "yes",
        }
        if self._wandb_required and (wandb_disabled or wandb_mode != "online"):
            raise RuntimeError(
                "WANDB_REQUIRE_ONLINE=1 requires WANDB_MODE=online and WANDB_DISABLED unset."
            )
        if wandb_disabled:
            self.accelerator.wait_for_everyone()
            return

        init_failed = False
        init_error = None
        if self.accelerator.is_main_process:
            try:
                run = wandb.init(
                    name=self.config.run_id,
                    dir=os.path.join(self.config.output_dir, "wandb"),
                    project=self.config.wandb_project,
                    entity=self.config.wandb_entity,
                    group="vla-train",
                )
                if run is None:
                    raise RuntimeError("wandb.init returned no run")
                run_mode = str(getattr(getattr(run, "settings", None), "mode", "online")).lower()
                if self._wandb_required and run_mode != "online":
                    raise RuntimeError(f"W&B initialized in {run_mode!r} mode instead of online")
                self._wandb_enabled = True
            except Exception as exc:
                self._wandb_enabled = False
                init_error = exc
                init_failed = self._wandb_required
                if self._wandb_required:
                    logger.error(f"Required W&B init failed: {exc}")
                else:
                    logger.warning(f"W&B init failed; continuing without W&B: {exc}")

        failure = torch.tensor(
            [1 if init_failed else 0], dtype=torch.int32, device=self.accelerator.device
        )
        if dist.is_initialized():
            dist.broadcast(failure, src=0)
        if failure.item():
            raise RuntimeError(
                "W&B online initialization failed on rank 0; aborting because "
                "WANDB_REQUIRE_ONLINE=1."
            ) from init_error
        # Rendezvous after rank-0 W&B init. Otherwise a slow or failing init on
        # rank 0 lets the other ranks reach the first collective alone and
        # eventually hit an NCCL watchdog timeout.
        self.accelerator.wait_for_everyone()

    def _save_initial_configs(self):
        """Save full config and training script at the very start of training."""
        if not self.accelerator.is_main_process:
            return

        output_dir = Path(self.config.output_dir)

        # 1. Save config.full.yaml — the complete merged config (all parameters)
        if isinstance(self.config, AccessTrackedConfig):
            full_cfg = self.config.unwrap()
        else:
            full_cfg = self.config
        full_yaml_path = output_dir / "config.full.yaml"
        OmegaConf.save(full_cfg, full_yaml_path, resolve=True)
        logger.info(f"📝 Full config saved at {full_yaml_path}")

        # The model server must be able to reject an inference client whose
        # action/camera/prompt contract differs from training.  Keep this
        # small, human-readable sidecar next to the complete config rather
        # than relying on the compact accessed-only snapshot.
        plain_cfg = OmegaConf.to_container(full_cfg, resolve=True)
        vla_cfg = (plain_cfg or {}).get("datasets", {}).get("vla_data", {})
        schema = vla_cfg.get("xpolicylab_schema")
        if isinstance(schema, dict) and schema.get("robot_type") in {
            "xpolicylab_egovla",
            "xpolicylab_sparkarena",
        }:
            contract = {
                "contract_version": 2,
                "robot_type": schema["robot_type"],
                "action_dim": int(plain_cfg["framework"]["action_model"]["action_dim"]),
                "state_dim": int(plain_cfg["framework"]["action_model"]["state_dim"]),
                "action_mode": vla_cfg.get("action_mode"),
                "action_source": vla_cfg.get("action_source"),
                "action_temporal_offset": vla_cfg.get("action_temporal_offset"),
                "action_derived_from_state": vla_cfg.get("action_derived_from_state"),
                "camera_names": vla_cfg.get("camera_names"),
                "black_camera_names": vla_cfg.get("black_camera_names"),
                "image_size": vla_cfg.get("obs_image_size") or vla_cfg.get("image_size"),
                "include_state": vla_cfg.get("include_state"),
                "instruction_mapping_sha256": vla_cfg.get("instruction_mapping_sha256"),
                "conversion_manifest_sha256": vla_cfg.get("conversion_manifest_sha256"),
                "raw_dataset_manifest": vla_cfg.get("raw_dataset_manifest"),
                "xpolicylab_schema": schema,
            }
            contract_name = (
                "egovla_contract.json"
                if schema["robot_type"] == "xpolicylab_egovla"
                else "sparkarena_contract.json"
            )
            contract_path = output_dir / contract_name
            temporary = contract_path.with_name(f".{contract_path.name}.tmp")
            with temporary.open("w", encoding="utf-8") as stream:
                json.dump(contract, stream, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, contract_path)
            logger.info(f"📜 Training contract saved at {contract_path}")

        # 2. Save config.yaml — accessed-only snapshot (will be updated at checkpoints)
        if isinstance(self.config, AccessTrackedConfig):
            self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
            logger.info(f"📊 Accessed config snapshot saved at {output_dir / 'config.yaml'}")

    def _init_checkpointing(self):
        """Initialize checkpoint directory and handle checkpoint loading."""
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        pretrained_checkpoint = getattr(self.config.trainer, "pretrained_checkpoint", None)
        is_resume = getattr(self.config.trainer, "is_resume", False)
        self.resume_from_checkpoint = pretrained_checkpoint

        if is_resume:
            resume_from_checkpoint, self.completed_steps = self._get_latest_checkpoint(self.checkpoint_dir)
            if resume_from_checkpoint:
                self.resume_from_checkpoint = resume_from_checkpoint
                self.model = self.load_pretrained_backbones(self.model, self.resume_from_checkpoint, reload_modules=None)
                logger.info(
                    f"Resuming training from checkpoint: {self.resume_from_checkpoint}, steps: {self.completed_steps}"
                )
                return

            logger.warning(f"No valid checkpoint found in {self.checkpoint_dir}. Starting training from scratch.")
            self.completed_steps = 0

        if pretrained_checkpoint:
            reload_modules = getattr(self.config.trainer, "reload_modules", None)
            self.model = self.load_pretrained_backbones(self.model, pretrained_checkpoint, reload_modules=reload_modules)
            self.completed_steps = 0
            self.resume_from_checkpoint = pretrained_checkpoint
            logger.info(f"Loaded pretrained checkpoint: {pretrained_checkpoint}, steps: {self.completed_steps}")
        else:
            logger.info("No pretrained checkpoint provided. Starting training from scratch.")
            self.completed_steps = 0

    def _adjust_lr_scheduler_for_resume(self):
        """Adjust LR scheduler state after resuming from non-zero steps."""
        if self.completed_steps > 0:
            logger.info(f"Adjusting LR scheduler for resume from step {self.completed_steps}")
            for _ in range(self.completed_steps):
                self.lr_scheduler.step()
            logger.info(
                f"LR scheduler adjusted to step {self.completed_steps}, current LR: {self.lr_scheduler.get_last_lr()}"
            )

    def _load_checkpoint(self, checkpoint_path):
        """Load checkpoint."""
        self.accelerator.load_state(checkpoint_path)
        self.accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")

    def _save_checkpoint(self):
        """Save current training state."""
        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")
            checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")

            state_dict = self.accelerator.get_state_dict(self.model)
            if save_format == "safetensors":
                from safetensors.torch import save_file

                save_file(state_dict, checkpoint_path + "_model.safetensors")
            elif save_format == "pt":
                torch.save(state_dict, checkpoint_path + "_pytorch_model.pt")
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")

            summary_data = {"steps": self.completed_steps}
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")
            self.accelerator.print(f"✅ Checkpoint saved at {checkpoint_path}")

            if isinstance(self.config, AccessTrackedConfig):
                logger.info("📊 Saving accessed configuration...")
                output_dir = Path(self.config.output_dir)
                self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
                logger.info("✅ Configuration files saved")

        self.accelerator.wait_for_everyone()

    def _log_metrics(self, metrics):
        """Record training metrics."""
        rank = dist.get_rank() if dist.is_initialized() else 0
        if self.completed_steps % self.config.trainer.logging_frequency != 0:
            return

        log_failed = False
        log_error = None
        if rank == 0:
            last_lrs = self.lr_scheduler.get_last_lr()
            for i, group in enumerate(self.optimizer.param_groups):
                group_name = group.get("name", str(i))
                metrics[f"learning_rate/{group_name}"] = last_lrs[i] if i < len(last_lrs) else last_lrs[-1]
            metrics["epoch"] = round(self.completed_steps / len(self.vla_train_dataloader), 2)
            if getattr(self, "_wandb_enabled", False):
                try:
                    wandb.log(metrics, step=self.completed_steps)
                except Exception as exc:
                    log_error = exc
                    log_failed = getattr(self, "_wandb_required", False)
                    if log_failed:
                        logger.error(f"Required W&B log failed: {exc}")
                    else:
                        self._wandb_enabled = False
                        logger.warning(f"W&B log failed; disabling W&B: {exc}")
            elif getattr(self, "_wandb_required", False):
                log_failed = True
            logger.info(f"Step {self.completed_steps}, Loss: {metrics})")

        if getattr(self, "_wandb_required", False):
            failure = torch.tensor(
                [1 if log_failed else 0], dtype=torch.int32, device=self.accelerator.device
            )
            if dist.is_initialized():
                dist.broadcast(failure, src=0)
            if failure.item():
                raise RuntimeError(
                    "W&B metric upload failed on rank 0; aborting because "
                    "WANDB_REQUIRE_ONLINE=1."
                ) from log_error

    def _create_data_iterators(self):
        """Create data iterators."""
        self.vla_iter = iter(self.vla_train_dataloader)

    def _get_next_batch(self):
        """Get next batch (automatically handle data loop)."""
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            if not hasattr(self, "vla_epoch_count"):
                self.vla_epoch_count = 0
            self.vla_iter, self.vla_epoch_count = TrainerUtils._reset_dataloader(
                self.vla_train_dataloader, self.vla_epoch_count
            )
            batch_vla = next(self.vla_iter)

        return batch_vla

    def train(self):
        """Execute training loop."""
        self._log_training_config()
        self._create_data_iterators()
        progress_bar = tqdm(
            total=self.config.trainer.max_train_steps,
            initial=self.completed_steps,
            disable=not self.accelerator.is_local_main_process,
        )

        while self.completed_steps < self.config.trainer.max_train_steps:
            t_start_data = time.perf_counter()
            batch_vla = self._get_next_batch()
            t_end_data = time.perf_counter()

            t_start_model = time.perf_counter()
            step_metrics = self._train_step(batch_vla)
            t_end_model = time.perf_counter()

            optimizer_stepped = step_metrics.pop("_optimizer_step", self.accelerator.sync_gradients)
            if optimizer_stepped:
                progress_bar.update(1)
                self.completed_steps += 1

            if self.accelerator.is_local_main_process:
                progress_bar.set_postfix(
                    {
                        "data_times": f"{t_end_data - t_start_data:.3f}",
                        "model_times": f"{t_end_model - t_start_model:.3f}",
                    }
                )

            if optimizer_stepped and self.completed_steps % self.config.trainer.eval_interval == 0:
                step_metrics = self.eval_action_model(step_metrics)

            step_metrics["timing/data"] = t_end_data - t_start_data
            step_metrics["timing/model"] = t_end_model - t_start_model
            if optimizer_stepped:
                self._log_metrics(step_metrics)

            if optimizer_stepped and self.completed_steps % self.config.trainer.save_interval == 0 and self.completed_steps > 0:
                self._save_checkpoint()

            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        self._finalize_training()

    def eval_action_model(self, step_metrics: dict = None) -> float:
        """Run simple action-eval on current batch and attach score to metrics."""
        examples = self._get_next_batch()
        actions = [example["action"] for example in examples]
        output_dict = self.accelerator.unwrap_model(self.model).predict_action(
            examples=examples, use_ddim=True, num_ddim_steps=20
        )

        if self.accelerator.is_main_process:
            normalized_actions = output_dict["normalized_actions"]
            actions = np.array(actions)
            num_pots = np.prod(actions.shape)
            score = TrainerUtils.euclidean_distance(normalized_actions, actions)
            step_metrics["mse_score"] = score / num_pots

        del examples
        if dist.is_initialized():
            dist.barrier()
        return step_metrics

    def _log_training_config(self):
        """Record training config."""
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  Per device batch size = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Gradient accumulation steps = {self.accelerator.gradient_accumulation_steps}")
            logger.info(f"  Total batch size = {self.total_batch_size}")

    def _train_step(self, batch_vla, batch_vlm=None):
        """Execute single training step."""
        # Accelerate implements accumulation with ``no_sync()``, which
        # DeepSpeed ZeRO-2/3 forbids because gradients are partitioned.  Let
        # the prepared DeepSpeed engine own accumulation instead: ``backward``
        # accumulates and ``step`` updates only at its configured boundary.
        if hasattr(self.model, "is_gradient_accumulation_boundary") and hasattr(self.model, "backward"):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_dict = self.model.forward(batch_vla)
                action_loss = output_dict["action_loss"]

            self.model.backward(action_loss)
            optimizer_stepped = bool(self.model.is_gradient_accumulation_boundary())
            self.model.step()
            if optimizer_stepped:
                self.lr_scheduler.step()

            return {
                "action_dit_loss": action_loss.item(),
                "_optimizer_step": optimizer_stepped,
            }

        with self.accelerator.accumulate(self.model):
            self.optimizer.zero_grad()

            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_dict = self.model.forward(batch_vla)
                action_loss = output_dict["action_loss"]
                total_loss = action_loss

            self.accelerator.backward(total_loss)

            if self.config.trainer.gradient_clipping is not None:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)

            self.optimizer.step()
            # Only step the LR scheduler when gradients are actually synced
            # (i.e., not mid-accumulation). Without this guard the scheduler
            # runs gradient_accumulation_steps times faster than intended,
            # causing warmup to end too early and cosine decay to bottom out
            # at min_lr well before max_train_steps is reached.
            if self.accelerator.sync_gradients:
                self.lr_scheduler.step()

        return {
            "action_dit_loss": action_loss.item(),
            "_optimizer_step": self.accelerator.sync_gradients,
        }

    def _finalize_training(self):
        """Training end processing."""
        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")
            final_checkpoint = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_checkpoint, exist_ok=True)
            state_dict = self.accelerator.get_state_dict(self.model)
            if save_format == "safetensors":
                from safetensors.torch import save_file

                save_file(state_dict, os.path.join(final_checkpoint, "model.safetensors"))
            elif save_format == "pt":
                torch.save(state_dict, os.path.join(final_checkpoint, "pytorch_model.pt"))
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")
            logger.info(f"Training complete. Final model saved at {final_checkpoint}")

        finish_failed = False
        finish_error = None
        if self.accelerator.is_main_process and getattr(self, "_wandb_enabled", False):
            try:
                wandb.finish()
            except Exception as exc:
                finish_error = exc
                finish_failed = getattr(self, "_wandb_required", False)
                if finish_failed:
                    logger.error(f"Required W&B finish failed: {exc}")
                else:
                    logger.warning(f"W&B finish failed: {exc}")

        if getattr(self, "_wandb_required", False):
            failure = torch.tensor(
                [1 if finish_failed else 0], dtype=torch.int32, device=self.accelerator.device
            )
            if dist.is_initialized():
                dist.broadcast(failure, src=0)
            if failure.item():
                raise RuntimeError(
                    "W&B final synchronization failed on rank 0; aborting because "
                    "WANDB_REQUIRE_ONLINE=1."
                ) from finish_error

        self.accelerator.wait_for_everyone()


def main(cfg) -> None:
    logger.info("VLA Training :: Warming Up")

    cfg = wrap_config(cfg)
    logger.info("✅ Configuration wrapped for access tracking")

    output_dir = setup_directories(cfg=cfg)
    vla = build_framework(cfg)
    vla_train_dataloader = prepare_data(cfg=cfg, accelerator=accelerator, output_dir=output_dir)
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

    trainer = VLATrainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )

    trainer.prepare_training()
    trainer.train()

    logger.info("... and that's all, folks!")
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/simBenchmarks/SimplerEnv/train_files/starvla_cotrain_oxe.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    # Normalise legacy YAML keys into the current `version_id == "0.21"` schema.
    # This is idempotent and does not modify framework class signatures.
    # See bar/config_tighten.md for the rationale.
    cfg = apply_config_compat(cfg)

    # Store source config path for later copying to output dir
    cfg.config_yaml = args.config_yaml

    if cfg.is_debug and dist.is_initialized() and dist.get_rank() == 0:
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("🔍 Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    main(cfg)
