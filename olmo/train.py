from __future__ import annotations

import cProfile
import functools
import gc
import logging
import math
import os
import random
import shutil
import time
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from pstats import SortKey
from typing import Any, Callable, Deque, Dict, List, Optional, TextIO, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils
import torch.utils.hooks
import wandb
from packaging import version
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from .aliases import PathOrStr
from .checkpoint import Checkpointer, FullCheckpointer, build_sharded_checkpointer
from .config import (
    CheckpointType,
    DDPGradSyncMode,
    DistributedStrategy,
    SchedulerUnits,
    ShardedCheckpointerType,
    SpeedMonitorConfig,
    TrainConfig,
)
from .data import IterableDataset
from .eval import Evaluator
from .exceptions import OLMoConfigurationError
from .model import OLMo
from .optim import Optimizer, Scheduler
from .torch_util import (
    barrier,
    gc_cuda,
    get_fs_local_rank,
    get_global_rank,
    get_world_size,
    move_to_device,
    peak_gpu_memory,
    synchronize_flag,
    synchronize_value,
)
from .util import upload

__all__ = ["SpeedMonitor", "LRMonitor", "Trainer"]

log = logging.getLogger(__name__)


@dataclass
class SpeedMonitor:
    cfg: SpeedMonitorConfig
    start_times: Deque[float] = field(default_factory=lambda: deque([]))
    global_total_tokens: int = 0
    total_training_Gflops: float = 0
    device_interval_tokens: Deque[int] = field(default_factory=lambda: deque([]))

    def batch_start(
        self,
        global_total_tokens: int,
        device_batch_num_tokens: int,
        num_fwd_flops: int,
        num_bck_flops: int,
        record: bool = True,
    ) -> None:
        self.global_total_tokens = global_total_tokens
        # num_fwd_flops and num_bck_flops from the OLMo model computes flops per token
        # converting to GFLOPs here prevents numerical issues while logging
        self.total_training_Gflops = (num_fwd_flops + num_bck_flops) * global_total_tokens / 1e9

        if record:
            if len(self.start_times) >= self.cfg.window_size:
                self.start_times.popleft()
                self.device_interval_tokens.popleft()
            self.start_times.append(time.monotonic())
            self.device_interval_tokens.append(device_batch_num_tokens)

    def reset(self) -> None:
        self.start_times.clear()
        self.device_interval_tokens.clear()

    def check(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {"throughput/total_tokens": self.global_total_tokens}

        # plot flops related metrics
        metrics["throughput/total_training_Gflops"] = self.total_training_Gflops
        metrics["throughput/total_training_log_Gflops"] = math.log(self.total_training_Gflops)

        if self.start_times:
            interval_seconds = time.monotonic() - self.start_times[0]
            interval_batches = len(self.start_times)
            interval_tokens = sum(self.device_interval_tokens)
            metrics["throughput/device/tokens_per_second"] = interval_tokens / interval_seconds
            metrics["throughput/device/batches_per_second"] = interval_batches / interval_seconds
        return metrics


@dataclass
class LRMonitor:
    optim: torch.optim.Optimizer

    def check(self) -> Dict[str, float]:
        lrs = [group["lr"] for group in self.optim.param_groups]
        return {f"optim/learning_rate_group{idx}": lr for idx, lr in enumerate(lrs)}


def cross_entropy_loss(
    logits,
    labels,
    ignore_index: int = -100,
    reduction: str = "mean",
    compute_z_loss: bool = False,
    z_loss_multiplier: float = 1e-4,
):
    loss = F.cross_entropy(logits, labels, ignore_index=ignore_index, reduction=reduction)

    if not compute_z_loss:
        return loss, None

    z_squared = logits.logsumexp(-1).pow(2)
    if reduction == "mean":
        z_squared = (z_squared * (labels != ignore_index)).mean()
    elif reduction == "sum":
        z_squared = (z_squared * (labels != ignore_index)).sum()

    z_loss = z_loss_multiplier * z_squared

    return loss, z_loss


fused_loss_fn: Optional[Callable]

try:
    import flash_attn
    from flash_attn.ops.triton.cross_entropy import (
        cross_entropy_loss as flash_cross_entropy_loss,  # type: ignore
    )

    def fused_loss_fn(
        logits,
        labels,
        ignore_index: int = -100,
        reduction: str = "mean",
        compute_z_loss: bool = False,
        z_loss_multiplier: float = 1e-4,
    ):
        # The `ignored_index` parameter of `cross_entropy_loss` was changed to `ignore_index` in v2.5.8 with commit https://github.com/Dao-AILab/flash-attention/commit/ec6d22143b5d375e253b2ebfc563b26a43f43684
        ce_loss_use_ignore_index_param = version.parse(flash_attn.__version__) >= version.parse("2.5.8")

        if ce_loss_use_ignore_index_param:
            ignore_index_kwarg = {"ignore_index": ignore_index}
        else:
            ignore_index_kwarg = {"ignored_index": ignore_index}

        loss, z_loss = flash_cross_entropy_loss(
            logits,
            labels,
            label_smoothing=0.0,
            logit_scale=1.0,
            lse_square_scale=z_loss_multiplier,
            inplace_backward=False,
            process_group=None,
            **ignore_index_kwarg,
        )

        mask = labels != ignore_index

        if reduction == "mean":
            loss = loss.sum() / mask.sum()
        elif reduction == "sum":
            loss = loss.sum()
        else:
            loss = loss

        if not compute_z_loss:
            return loss, None

        if reduction == "mean":
            z_loss = z_loss.sum() / mask.sum()
        elif reduction == "sum":
            z_loss = z_loss.sum()
        else:
            z_loss = z_loss

        return loss, z_loss

except ImportError:
    fused_loss_fn = None


@dataclass
class Trainer:
    cfg: TrainConfig
    model: OLMo
    dist_model: Union[DDP, FSDP]
    optim: Optimizer
    scheduler: Scheduler
    train_loaders: Dict[str, DataLoader]
    device: torch.device
    evaluators: List[Evaluator]
    epoch: Optional[int] = None
    global_step: int = 0
    global_train_examples_seen_this_epoch: int = 0
    """Tracks the global number of training examples seen in the current epoch for the purpose of restoring
    the data loader position on restarts."""
    global_train_tokens_seen: int = 0
    """Tracks the global total number of tokens trained on."""
    checkpoints: List[Path] = field(default_factory=list)
    unsharded_checkpoints: List[Path] = field(default_factory=list)
    ephemeral_checkpoints: List[Path] = field(default_factory=list)
    min_train_loss: float = float("inf")
    cur_train_loss: float = float("inf")
    indices_files: Dict[str, TextIO] = field(default_factory=dict)
    _start_time: float = 0.0
    _gc_init_state: bool = True
    loss_fn: Callable[..., torch.Tensor] = field(default_factory=lambda: cross_entropy_loss)  # type: ignore
    last_sharded_checkpoint_step: Optional[int] = None
    last_unsharded_checkpoint_step: Optional[int] = None
    current_stage_loader: Optional[DataLoader] = None
    current_stage_name: Optional[str] = None
    current_stage_loader_iter: Optional[iter] = None
    start_global_step_to_stage_name: Dict[int, str] = field(default_factory=dict)

    def _cleanup_current_iterator(self) -> None:
        """Helper method to clean up the current data loader iterator."""
        if self.current_stage_loader_iter is not None:
            # If the iterator has a close method, call it
            if hasattr(self.current_stage_loader_iter, "close"):
                self.current_stage_loader_iter.close()
            # Set to None to help garbage collection
            self.current_stage_loader_iter = None

    def _should_stop_training(self, stop_at: int) -> bool:
        """Helper method to check if training should stop.

        Args:
            stop_at: The step at which to stop training

        Returns:
            bool: True if training should stop, False otherwise
        """
        if self.global_step >= self.max_steps:
            log.info(f"Reached max_steps ({self.max_steps}), stopping training")
            return True
        if self.global_step >= stop_at:
            log.info(f"Reached stop_at step ({stop_at}), stopping training")
            return True
        return False

    def __post_init__(self):
        if self.cfg.fused_loss:
            if fused_loss_fn is not None:
                self.loss_fn = fused_loss_fn
            else:
                raise NameError("`fused_loss_fn` is not defined. Please ensure that `flash_attn` is installed.")

        # Create mapping from start steps to stage names
        self.start_global_step_to_stage_name = (
            {stage.start_global_step: stage.name for stage in self.cfg.data_stages} if self.cfg.data_stages else {}
        )

        # Initialize with first stage if available
        if self.train_loaders and not self.current_stage_loader:
            stage_name = next(iter(self.train_loaders))  # get the name of the first stage
            self.current_stage_loader = self.train_loaders[stage_name]
            self.current_stage_name = stage_name
            self.current_stage_loader_iter = iter(self.current_stage_loader)
            log.info(f"Current data stage: {stage_name}")

    @property
    def dataset(self) -> IterableDataset:
        assert self.current_stage_loader is not None, "No data loader is currently active"
        assert isinstance(self.current_stage_loader.dataset, IterableDataset)
        return self.current_stage_loader.dataset

    @property
    def tokens_per_batch(self) -> int:
        return self.cfg.global_train_batch_size * self.cfg.model.max_sequence_length

    @property
    def batches_per_epoch(self) -> int:
        return self.dataset.total_size // self.cfg.global_train_batch_size

    @property
    def max_epochs(self) -> int:
        return math.ceil(self.max_steps / self.batches_per_epoch)

    @property
    def max_steps(self) -> int:
        if isinstance(self.cfg.max_duration, int):
            return self.cfg.max_duration
        elif isinstance(self.cfg.max_duration, str):
            if self.cfg.max_duration.endswith("T"):
                # convert to float *first* to handle scientific notation
                max_tokens = int(float(self.cfg.max_duration[:-1].strip()))
                tokens_remaining = max(max_tokens - self.global_train_tokens_seen, 0)
                steps_remaining = math.ceil(tokens_remaining / self.tokens_per_batch)
                return self.global_step + steps_remaining
            elif self.cfg.max_duration.endswith("ep"):
                max_epochs = int(self.cfg.max_duration[:-2].strip())
                return max_epochs * self.batches_per_epoch
            else:
                # convert to float *first* to handle scientific notation
                return int(float(self.cfg.max_duration))
        else:
            raise TypeError(f"expected int or str for 'max_duration', found {type(self.cfg.max_duration)}")

    @property
    def max_tokens(self) -> int:
        if isinstance(self.cfg.max_duration, int):
            return (
                self.global_train_tokens_seen
                + max(self.cfg.max_duration - self.global_step, 0) * self.tokens_per_batch
            )
        elif isinstance(self.cfg.max_duration, str):
            if self.cfg.max_duration.endswith("T"):
                # convert to float *first* to handle scientific notation
                return int(float(self.cfg.max_duration[:-1].strip()))
            elif self.cfg.max_duration.endswith("ep"):
                max_epochs = int(self.cfg.max_duration[:-2].strip())
                return max_epochs * self.batches_per_epoch * self.tokens_per_batch
            else:
                # convert to float *first* to handle scientific notation
                return (
                    self.global_train_tokens_seen
                    + max(int(float(self.cfg.max_duration)) - self.global_step, 0) * self.tokens_per_batch
                )
        else:
            raise TypeError(f"expected int or str for 'max_duration', found {type(self.cfg.max_duration)}")

    @property
    def scheduler_current(self) -> int:
        if self.cfg.scheduler.units == SchedulerUnits.steps:
            return self.global_step
        elif self.cfg.scheduler.units == SchedulerUnits.tokens:
            return self.global_train_tokens_seen
        else:
            raise NotImplementedError(self.cfg.scheduler.units)

    @property
    def scheduler_max(self) -> int:
        if self.cfg.scheduler.units == SchedulerUnits.steps:
            return self.max_steps
        elif self.cfg.scheduler.units == SchedulerUnits.tokens:
            return self.max_tokens
        else:
            raise NotImplementedError(self.cfg.scheduler.units)

    def trainer_state_dict(self) -> Dict[str, Any]:
        return {
            "epoch": self.epoch or 0,
            "global_step": self.global_step,
            "global_train_examples_seen_this_epoch": self.global_train_examples_seen_this_epoch,
            "global_train_tokens_seen": self.global_train_tokens_seen,
            "world_size": get_world_size(),
            "checkpoints": self.checkpoints,
            "unsharded_checkpoints": self.unsharded_checkpoints,
            "ephemeral_checkpoints": self.ephemeral_checkpoints,
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.random.get_rng_state(),
                "cuda": torch.cuda.get_rng_state(),
            },
        }

    def load_trainer_state_dict(self, state_dict: Dict[str, Any]) -> None:
        # Checkpoint paths.
        self.checkpoints = [
            path
            for path in state_dict["checkpoints"]
            if path.is_dir() and path.resolve().parent == Path(self.cfg.save_folder).resolve()
        ]
        self.unsharded_checkpoints = [
            path
            for path in state_dict["unsharded_checkpoints"]
            if path.is_dir() and path.resolve().parent == Path(self.cfg.save_folder).resolve()
        ]
        self.ephemeral_checkpoints = [
            path
            for path in state_dict.get("ephemeral_checkpoints", [])
            if path.is_dir() and path.resolve().parent == Path(self.cfg.save_folder).resolve()
        ]

        # Dataset / dataloader position.
        checkpoint_epoch = state_dict.get("epoch") or 0
        self.global_step = state_dict["global_step"]
        self.global_train_examples_seen_this_epoch = state_dict.get(
            "global_train_examples_seen_this_epoch",
            state_dict.get(  # for backwards compatibility
                "global_train_examples_seen",
                state_dict.get("global_data_step", self.global_step) * self.cfg.global_train_batch_size,
            ),
        )
        self.global_train_tokens_seen = state_dict.get(
            "global_train_tokens_seen",
            state_dict.get("global_data_step", self.global_step)  # for backwards compatibility
            * self.cfg.global_train_batch_size
            * self.cfg.model.max_sequence_length,
        )

        if not self.cfg.restore_dataloader:
            self.epoch = 0
            self.global_step = 0
            self.global_train_tokens_seen = 0
            self.global_train_examples_seen_this_epoch = 0
        elif self.epoch is None:
            self.epoch = checkpoint_epoch
        elif checkpoint_epoch != self.epoch:
            log.info(f"Starting new epoch (epoch = {self.epoch})")
            self.global_train_examples_seen_this_epoch = 0

        # Find and set the correct data loader based on global_step
        if (stage_name := self.start_global_step_to_stage_name.get(self.global_step)) is not None:
            if stage_name != self.current_stage_name:
                log.info(f"Switching to data stage {stage_name} at step {self.global_step}")
                self.current_stage_name = stage_name
                # Clean up old iterator before switching
                self._cleanup_current_iterator()
                if isinstance(self.train_loaders[stage_name], DataLoader):
                    self.current_stage_loader = self.train_loaders[stage_name]
                else:
                    self.current_stage_loader = self.train_loaders[stage_name]()
                    self.train_loaders[stage_name] = self.current_stage_loader
                # Get an iterator for the current stage loader
                self.current_stage_loader_iter = iter(self.current_stage_loader)

        assert self.epoch is not None
        # Reshuffle dataset if needed.
        if self.dataset.epoch != self.epoch:
            log.info(f"Reshuffling data loader for epoch {self.epoch}...")
            self.dataset.reshuffle(self.epoch)

        if self.cfg.fast_forward_batches:
            log.info(f"Fast-forwarding data loader by {self.cfg.fast_forward_batches:,d} steps")
            # Technically we don't "see" these batches that we fast-forward through, but we use
            # this variable to update the position of the dataset so we need to include them here.
            self.global_train_examples_seen_this_epoch += (
                self.cfg.fast_forward_batches * self.cfg.global_train_batch_size
            )
            # NOTE: on the other hand we don't add anything to 'self.global_train_tokens_seen' here because
            # that variable is meant to track the actual number of tokens trained on.

        if self.global_train_examples_seen_this_epoch > 0:
            assert isinstance(self.dataset, IterableDataset)
            log.info(f"Data loader will start at instance index {self.global_train_examples_seen_this_epoch:,d}")
            self.dataset.start_index = self.global_train_examples_seen_this_epoch

        # Reset learning rate and weight decay to the values from the config, not the checkpoint.
        log.info("Resetting learning rate...")
        new_learning_rate = self.scheduler.get_lr(
            self.cfg.optimizer.learning_rate, self.scheduler_current, self.scheduler_max
        )
        for group in self.optim.param_groups:
            group["lr"] = new_learning_rate
            group["initial_lr"] = self.cfg.optimizer.learning_rate
            if "weight_decay" in group and group["weight_decay"] > 0.0:
                group["weight_decay"] = self.cfg.optimizer.weight_decay

        # RNG states.
        if "rng" in state_dict and state_dict.get("world_size", get_world_size()) == get_world_size():
            log.info("Restoring RNG states...")
            rng_state = state_dict["rng"]
            self.restore_rng_state(rng_state)
        else:
            log.warning(
                "Trainer will not restore RNG states since the RNG states in the checkpoint are missing or invalid. "
                "This typically happens when restoring from an unsharded checkpoint or a checkpoint that was saved "
                "with a different world size. If that's the case you can safely ignore this warning."
            )

    def restore_rng_state(self, rng_state: Dict[str, Any]) -> None:
        random.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])
        torch.set_rng_state(rng_state["torch"])
        torch.cuda.set_rng_state(rng_state["cuda"])

    def _save_checkpoint(
        self, checkpointer: Checkpointer, checkpoint_type: CheckpointType
    ) -> Tuple[PathOrStr, Optional[PathOrStr]]:
        if checkpoint_type == CheckpointType.sharded:
            suffix = ""
            current_checkpoints = self.checkpoints
            link_latest = get_fs_local_rank() == 0
            num_checkpoints_to_keep = self.cfg.save_num_checkpoints_to_keep
        elif checkpoint_type == CheckpointType.unsharded:
            suffix = "-unsharded"
            current_checkpoints = self.unsharded_checkpoints
            link_latest = get_global_rank() == 0
            num_checkpoints_to_keep = self.cfg.save_num_unsharded_checkpoints_to_keep
        elif checkpoint_type == CheckpointType.sharded_ephemeral:
            suffix = ""
            current_checkpoints = self.ephemeral_checkpoints
            link_latest = get_fs_local_rank() == 0
            num_checkpoints_to_keep = 1
        else:
            raise NotImplementedError(checkpoint_type)

        # Zero-gradients to avoid gathering them.
        self.optim.zero_grad(set_to_none=True)

        checkpoint_dir = Path(self.cfg.save_folder) / f"step{self.global_step}{suffix}"
        remote_checkpoint_dir: Optional[str] = None
        if self.cfg.remote_save_folder is not None:
            remote_checkpoint_dir = f"{self.cfg.remote_save_folder.rstrip('/')}/{checkpoint_dir.name}"
        current_checkpoints.append(checkpoint_dir)

        # Save the checkpoint.
        try:
            checkpointer.save_checkpoint(
                checkpoint_dir,
                self.dist_model,
                self.optim,
                self.trainer_state_dict(),
                upload_to=remote_checkpoint_dir,
            )
        except FileExistsError:
            raise OLMoConfigurationError(
                f"Checkpoint for step {self.global_step} already exists, use --save_overwrite to overwrite it"
            )

        if link_latest:
            # Link to 'latest'.
            latest_path = Path(self.cfg.save_folder) / f"latest{suffix}"
            latest_path.unlink(missing_ok=True)
            try:
                latest_path.symlink_to(checkpoint_dir.name, target_is_directory=True)
            except FileExistsError:
                # Same as above, caught when another (file-system) local rank 0 has already made the 'latest' symlink.
                # This can happen when nodes are saving to a common NFS drive but otherwise have distinct
                # file-systems.
                if latest_path.resolve().name != checkpoint_dir.name:
                    raise

        # Remove old checkpoints.
        # For DDP, checkpoint_type being passed to remove_checkpoint is always `unsharded`.
        if num_checkpoints_to_keep > 0:
            while len(current_checkpoints) > num_checkpoints_to_keep:
                self.remove_checkpoint(0, checkpoint_type)

        barrier()

        if remote_checkpoint_dir is not None:
            return remote_checkpoint_dir, checkpoint_dir
        else:
            return checkpoint_dir, None

        # After checkpoint operations
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def save_sharded_checkpoint(self) -> Tuple[PathOrStr, Optional[PathOrStr]]:
        checkpointer = build_sharded_checkpointer(self.cfg)
        result = self._save_checkpoint(checkpointer, CheckpointType.sharded)
        self.last_sharded_checkpoint_step = self.global_step
        return result

    def save_ephemeral_checkpoint(self) -> Tuple[PathOrStr, Optional[PathOrStr]]:
        checkpointer = build_sharded_checkpointer(self.cfg)
        result = self._save_checkpoint(checkpointer, CheckpointType.sharded_ephemeral)
        self.last_sharded_checkpoint_step = self.global_step
        return result

    def _remove_sharded_checkpoint(self, idx: int, checkpoints: List[Path]):
        oldest_checkpoint = checkpoints.pop(idx)
        barrier()
        if get_fs_local_rank() == 0 and oldest_checkpoint.is_dir():
            shutil.rmtree(oldest_checkpoint, ignore_errors=True)
            latest_path = Path(self.cfg.save_folder) / "latest"
            if latest_path.resolve() == oldest_checkpoint.resolve():
                latest_path.unlink()
        barrier()

    def remove_sharded_checkpoint(self, idx: int = 0):
        self._remove_sharded_checkpoint(idx, self.checkpoints)

    def remove_ephemeral_checkpoint(self, idx: int = 0):
        self._remove_sharded_checkpoint(idx, self.ephemeral_checkpoints)

    def restore_sharded_checkpoint(
        self,
        load_path: PathOrStr,
        local_cache: Optional[PathOrStr] = None,
        *,
        load_optimizer_state: bool = True,
        load_trainer_state: bool = True,
        sharded_checkpointer: Optional[ShardedCheckpointerType] = None,
    ):
        # Zero-gradients to avoid gathering them.
        self.optim.zero_grad(set_to_none=True)
        checkpointer = build_sharded_checkpointer(self.cfg, name=sharded_checkpointer)
        trainer_state = checkpointer.restore_checkpoint(
            load_path,
            self.dist_model,
            self.optim,
            local_cache=local_cache,
            load_optimizer_state=load_optimizer_state,
        )
        if load_trainer_state:
            self.load_trainer_state_dict(trainer_state)
        barrier()

    def save_unsharded_checkpoint(self) -> Tuple[PathOrStr, Optional[PathOrStr]]:
        checkpointer = FullCheckpointer(self.cfg)
        result = self._save_checkpoint(checkpointer, CheckpointType.unsharded)
        self.last_unsharded_checkpoint_step = self.global_step
        return result

    def remove_unsharded_checkpoint(self, idx: int = 0):
        barrier()
        oldest_checkpoint = self.unsharded_checkpoints.pop(idx)
        if get_global_rank() == 0 and oldest_checkpoint.is_dir():
            shutil.rmtree(oldest_checkpoint, ignore_errors=True)
            latest_path = Path(self.cfg.save_folder) / "latest-unsharded"
            if latest_path.resolve() == oldest_checkpoint.resolve():
                latest_path.unlink()
        barrier()

    def restore_unsharded_checkpoint(
        self,
        load_path: PathOrStr,
        local_cache: Optional[PathOrStr] = None,
        *,
        load_optimizer_state: bool = True,
        load_trainer_state: bool = True,
    ):
        # Zero-gradients to avoid gathering them.
        self.optim.zero_grad(set_to_none=True)
        checkpointer = FullCheckpointer(self.cfg)
        trainer_state = checkpointer.restore_checkpoint(
            load_path,
            self.dist_model,
            self.optim,
            local_cache=local_cache,
            load_optimizer_state=load_optimizer_state,
        )
        if load_trainer_state:
            self.load_trainer_state_dict(trainer_state)
        barrier()

    def save_checkpoint(
        self, checkpoint_type: CheckpointType = CheckpointType.sharded
    ) -> Tuple[PathOrStr, Optional[PathOrStr]]:
        result: Tuple[PathOrStr, Optional[PathOrStr]]
        if checkpoint_type == CheckpointType.sharded:
            result = self.save_sharded_checkpoint()
        elif checkpoint_type == CheckpointType.unsharded:
            result = self.save_unsharded_checkpoint()
        elif checkpoint_type == CheckpointType.sharded_ephemeral:
            result = self.save_ephemeral_checkpoint()
        else:
            raise NotImplementedError(checkpoint_type)

        gc_cuda()
        return result

    def restore_checkpoint(
        self,
        load_path: PathOrStr,
        *,
        checkpoint_type: Optional[CheckpointType] = None,
        local_cache: Optional[PathOrStr] = None,
        load_optimizer_state: bool = True,
        load_trainer_state: bool = True,
        sharded_checkpointer: Optional[ShardedCheckpointerType] = None,
    ):
        if checkpoint_type == CheckpointType.unsharded or (
            checkpoint_type is None and str(load_path).rstrip("/").endswith("-unsharded")
        ):
            self.restore_unsharded_checkpoint(
                load_path,
                local_cache=local_cache,
                load_optimizer_state=load_optimizer_state,
                load_trainer_state=load_trainer_state,
            )
        elif checkpoint_type == CheckpointType.sharded or checkpoint_type is None:
            self.restore_sharded_checkpoint(
                load_path,
                local_cache=local_cache,
                load_optimizer_state=load_optimizer_state,
                load_trainer_state=load_trainer_state,
                sharded_checkpointer=sharded_checkpointer,
            )
        elif checkpoint_type is not None:
            raise NotImplementedError(checkpoint_type)

        gc_cuda()

    def remove_checkpoint(self, idx: int = 0, checkpoint_type: CheckpointType = CheckpointType.sharded):
        if checkpoint_type == CheckpointType.sharded:
            self.remove_sharded_checkpoint(idx=idx)
        elif checkpoint_type == CheckpointType.unsharded:
            self.remove_unsharded_checkpoint(idx=idx)
        elif checkpoint_type == CheckpointType.sharded_ephemeral:
            self.remove_ephemeral_checkpoint(idx=idx)
        else:
            raise NotImplementedError(checkpoint_type)

    def _setup_module_output_save_hooks(self, micro_batch_idx: int) -> List[torch.utils.hooks.RemovableHandle]:
        if (
            self.cfg.module_outputs_save_steps is None
            or self.global_step not in self.cfg.module_outputs_save_steps
        ):
            return []

        if micro_batch_idx != 0 or get_global_rank() != 0:
            # Hook is currently only used on the first microbatch of rank 0
            return []

        trace_save_folder = Path(self.cfg.save_folder) / f"traces/step{self.global_step}"
        if trace_save_folder.exists():
            if self.cfg.save_overwrite:
                shutil.rmtree(trace_save_folder)
            else:
                raise OLMoConfigurationError(
                    f"Attempting to overwrite traces at step {self.global_step} without --save_overwrite"
                )
        trace_save_folder.mkdir(parents=True)

        def trace_outputs_hook(
            module_name: str, _: torch.nn.Module, args: Tuple[torch.Tensor, ...], output: torch.Tensor
        ) -> None:
            if len(args) == 0:
                log.info("No input args for module %s, output %s", module_name, output)

            module_input = args[0] if len(args) > 0 else torch.tensor(())
            trace_save_folder = Path(self.cfg.save_folder) / f"traces/step{self.global_step}"
            trace_save_folder.mkdir(parents=True, exist_ok=True)

            module_occurence_num = 0
            while (
                module_input_filepath := trace_save_folder / f"{module_name}_{module_occurence_num}_input.pt"
            ).exists():
                module_occurence_num += 1
            torch.save(module_input, module_input_filepath)

            module_output_filepath = trace_save_folder / f"{module_name}_{module_occurence_num}_output.pt"
            torch.save(output, module_output_filepath)

        output_hooks = []
        for module_name, module in self.model.named_modules(prefix="model"):
            output_hooks.append(module.register_forward_hook(functools.partial(trace_outputs_hook, module_name)))

        return output_hooks

    def get_labels(self, batch: Dict[str, Any]) -> torch.Tensor:
        # Labels are just input IDs shifted to the left (first item is ignored).
        labels, label_mask, attention_mask, instance_mask = (
            batch["input_ids"].clone(),
            batch.get("label_mask"),
            batch.get("attention_mask"),
            batch.get("instance_mask"),
        )
        if label_mask is not None:
            labels.masked_fill_(~label_mask, -100)
        if attention_mask is not None:
            labels.masked_fill_(attention_mask == 0.0, -100)
        if instance_mask is not None:
            labels.masked_fill_(~instance_mask.unsqueeze(-1), value=-100)
        return labels[..., 1:].contiguous()

    def model_forward(
        self, batch: Dict[str, Any], loss_reduction: str = "mean", compute_z_loss: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        # shape: (batch_size, seq_len, vocab_size)
        logits = self.dist_model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            attention_bias=batch.get("attention_bias"),
            doc_lens=batch.get("doc_lens"),
            max_doc_lens=batch.get("max_doc_lens"),
        ).logits
        logits_for_loss = logits[..., :-1, :].contiguous()
        # shape: (batch_size * seq_len, vocab_size)
        logits_for_loss = logits_for_loss.view(-1, logits_for_loss.size(-1))
        # shape: (batch_size, seq_len)
        labels = self.get_labels(batch)
        # shape: (batch_size * seq_len,)
        labels = labels.view(-1)
        ce_loss, z_loss = self.loss_fn(
            logits_for_loss, labels, ignore_index=-100, reduction=loss_reduction, compute_z_loss=compute_z_loss
        )
        if loss_reduction == "none":
            # Reshape (batch_size * seq_len,) -> (batch_size, seq_len)
            ce_loss = ce_loss.view(batch["input_ids"].shape[0], -1)
            if z_loss is not None:
                z_loss = z_loss.view(batch["input_ids"].shape[0], -1)
        return ce_loss, z_loss, logits

    def train_micro_batch(
        self, micro_batch: Dict[str, Any], batch_size_in_tokens: int
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        ce_loss, z_loss, logits = self.model_forward(
            micro_batch, compute_z_loss=self.cfg.softmax_auxiliary_loss, loss_reduction="sum"
        )
        ce_loss = ce_loss / batch_size_in_tokens

        # In case this helps with memory utilization.
        del micro_batch

        # Get loss to optimize for.
        if self.cfg.softmax_auxiliary_loss:
            assert z_loss is not None
            z_loss = z_loss / batch_size_in_tokens
            loss = ce_loss + z_loss
        else:
            loss = ce_loss

        del logits

        return loss, ce_loss, z_loss

    def train_batch(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        # Split into micro-batches.
        micro_batches = self.split_batch(batch)
        batch_size_in_tokens = batch["input_ids"].numel()

        # In case this helps with memory utilization.
        del batch

        ce_batch_loss = torch.tensor(0.0, device=self.device)
        z_batch_loss = None if not self.cfg.softmax_auxiliary_loss else torch.tensor(0.0, device=self.device)
        num_micro_batches = len(micro_batches)

        for micro_batch_idx, micro_batch in enumerate(micro_batches):
            # setup sync context for DDP for all micro-batches except the last
            grad_sync_context = nullcontext
            if (
                self.cfg.distributed_strategy == DistributedStrategy.ddp
                and self.cfg.ddp is not None
                and self.cfg.ddp.grad_sync_mode == DDPGradSyncMode.batch
            ):
                if micro_batch_idx != num_micro_batches - 1:
                    grad_sync_context = self.dist_model.no_sync

            # Register output hooks
            output_hooks: List[torch.utils.hooks.RemovableHandle] = []
            output_hooks += self._setup_module_output_save_hooks(micro_batch_idx)

            with grad_sync_context():
                with torch.autocast("cuda", enabled=True, dtype=self.cfg.autocast_precision):
                    # Run forward pass.
                    loss, ce_loss, z_loss = self.train_micro_batch(micro_batch, batch_size_in_tokens)

                    # Update overall CE batch loss.
                    ce_batch_loss += ce_loss.detach()

                    # Update overall Z batch loss.
                    if z_loss is not None:
                        assert z_batch_loss is not None
                        z_batch_loss += z_loss.detach()

                # Run backward pass.
                loss.backward()

            # Remove output hooks
            for hook in output_hooks:
                hook.remove()

        return ce_batch_loss, z_batch_loss

    def train_step(self, batch: Dict[str, Any], reduce_global_loss: bool = True) -> Dict[str, float]:
        metrics: Dict[str, float] = {}

        # Write data-indices to file.
        if self.indices_files and "index" in batch and self.current_stage_name:
            indices = "\t".join(str(int(i)) for i in batch["index"])
            self.indices_files[self.current_stage_name].write(f"{self.global_step}\t{indices}\n")

        # Record how many instances are going to be skipped (masked out).
        if (instance_mask := batch.get("instance_mask")) is not None:
            metrics["train/masked_instances_local_rank"] = (~instance_mask).sum().item()

        # Zero-gradients.
        self.optim.zero_grad(set_to_none=True)

        # Move tensors to the right device.
        batch = move_to_device(batch, self.device)

        # Run forward-backward pass.
        ce_batch_loss, z_batch_loss = self.train_batch(batch)

        # Collect loss, potentially reducing over all ranks.
        if reduce_global_loss:
            dist.reduce(ce_batch_loss, 0)
            ce_batch_loss.div_(get_world_size())
            if z_batch_loss is not None:
                dist.reduce(z_batch_loss, 0)
                z_batch_loss.div_(get_world_size())

        # Clip gradient norms and collect param/gradient/optim metrics.
        should_log_optim_metrics_this_step = self.should_log_optim_metrics_this_step()
        optim_metrics = self.optim.clip_grads_and_collect_metrics(
            self.global_step,
            collect_param_metrics=should_log_optim_metrics_this_step,
            # passing this process group here ensures metrics are reduced correctly when we're using
            # HYBRID sharding.
            process_group=self.dist_model.process_group,
        )

        # Adjust the learning rate.
        for group in self.optim.param_groups:
            # TODO (epwalsh): if we want to enable different LRs or gradient clipping settings per group
            # we should pass `group["initial_lr"]` or `group["initial_max_grad_norm"]` here instead of
            # the corresponding values from `self.cfg`.
            group["lr"] = self.scheduler.get_lr(
                self.cfg.optimizer.learning_rate, self.scheduler_current, self.scheduler_max
            )
            group["max_grad_norm"] = self.scheduler.get_max_grad_norm(
                self.cfg.max_grad_norm, self.scheduler_current, self.scheduler_max
            )
            group["max_grad_norm_ratio"] = self.scheduler.get_max_grad_norm(
                self.cfg.max_grad_norm_ratio, self.scheduler_current, self.scheduler_max
            )

        # Optimizer step.
        self.optim.step()

        # Collect metrics and check for NaN loss.
        # NOTE: this involves a bunch of host-device syncs so we wait until the last moment to do this.
        if torch.isnan(ce_batch_loss):
            raise ValueError("nan loss encountered")
        if z_batch_loss is not None and torch.isnan(z_batch_loss):
            raise ValueError("nan loss encountered")
        for key, value in optim_metrics.items():
            metrics[f"optim/{key}"] = value.item()
        self.cur_train_loss = ce_batch_loss.item()
        self.min_train_loss = min(self.min_train_loss, self.cur_train_loss)
        metrics["train/CrossEntropyLoss"] = self.cur_train_loss
        metrics["train/Perplexity"] = math.exp(self.cur_train_loss)
        if z_batch_loss is not None:
            metrics["train/ZLoss"] = z_batch_loss.item()

        # Maybe collect post-step optimizer-specific metrics.
        if should_log_optim_metrics_this_step:
            optim_metrics = self.optim.get_post_step_metrics(
                self.dist_model, process_group=self.dist_model.process_group
            )
            for key, value in optim_metrics.items():
                metrics[f"optim/{key}"] = value.item()

        run = wandb.run
        if run is not None and self.cfg.wandb is not None:
            metrics["global_step"] = self.global_step
            run.log(metrics, step=self.global_step)

        return metrics

    def eval_batch(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.autocast("cuda", enabled=True, dtype=self.cfg.autocast_precision):
            ce_loss, _, logits = self.model_forward(batch, loss_reduction="none")
        return ce_loss.mean(dim=-1), logits

    def eval_step(self, batch: Dict[str, Any], evaluator: Evaluator) -> None:
        # Move tensors to the right device.
        batch = move_to_device(batch, self.device)

        # Run forward pass.
        with torch.no_grad():  # NOTE: 'torch.inference_mode()' doesn't work with 'torch.compile()'.
            ce_loss, logits = self.eval_batch(batch)

        # Update metrics.
        evaluator.update_metrics(
            batch, ce_loss, logits
        )  # batch includes all keys that the downstream evaluation needs

        barrier()

    def split_batch(self, batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        microbatch_size = self.cfg.device_train_microbatch_size
        batch_size = batch["input_ids"].shape[0]
        if batch_size <= microbatch_size:
            return [batch]
        else:
            micro_batches = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    micro_batches[key] = value.split(microbatch_size, dim=0)
                elif isinstance(value, list):
                    micro_batches[key] = [
                        value[microbatch_size * i : microbatch_size * i + microbatch_size]
                        for i in range(math.ceil(batch_size / microbatch_size))
                    ]
                else:
                    raise ValueError(f"unexpected item in batch: '{key}={value}'")
            return [
                {key: value[i] for key, value in micro_batches.items()}  # type: ignore
                for i in range(len(micro_batches["input_ids"]))
            ]

    def system_metrics(self) -> Dict[str, float]:
        metrics = {}
        if self.global_step < 3 or self.global_step % 10 == 0:
            peak_gpu_mb = peak_gpu_memory()
            if peak_gpu_mb is not None:
                metrics["System/Peak GPU Memory (MB)"] = peak_gpu_mb
        return metrics

    def log_metrics_to_console(self, prefix: str, metrics: Dict[str, float]):
        def format_float(value: float) -> str:
            if value < 0.0001:
                return str(value)  # scientific notation
            elif value > 1000:
                return f"{int(value):,d}"
            elif value > 100:
                return f"{value:.1f}"
            elif value > 10:
                return f"{value:.2f}"
            elif value > 1:
                return f"{value:.3f}"
            else:
                return f"{value:.4f}"

        log.info(
            f"{prefix}\n"
            + "\n".join(
                [
                    f"    {name}={format_float(value)}"
                    for name, value in metrics.items()
                    if name == "optim/total_grad_norm"
                    or not name.startswith("optim/")  # there's too many optimizer metrics
                ]
            )
        )

    def should_log_optim_metrics_this_step(self) -> bool:
        if self.cfg.wandb is None:
            # We only log optimizer-specific metrics to W&B, since there are usually too many metrics
            # to log to the console.
            return False
        optim_log_interval = self.cfg.optimizer.metrics_log_interval
        if optim_log_interval is None:
            optim_log_interval = self.cfg.wandb.log_interval
        else:
            optim_log_interval = max(optim_log_interval, self.cfg.wandb.log_interval)
        return self.global_step % optim_log_interval == 0

    def should_log_this_step(self) -> bool:
        if self.global_step % self.cfg.console_log_interval == 0:
            return True
        elif self.cfg.wandb is not None and self.global_step % self.cfg.wandb.log_interval == 0:
            return True
        else:
            return False

    def eval(self) -> Dict[str, Any]:
        # Zero gradients and set model to 'eval' mode.
        self.optim.zero_grad(set_to_none=True)
        self.dist_model.eval()

        eval_metrics = {}
        for evaluator in self.evaluators:
            log.info(f"Running evaluation for '{evaluator.label}'...")

            # Reset metrics.
            evaluator.reset_metrics()

            # Initialize data loader iterator.
            eval_batches = iter(evaluator.eval_loader)

            # Adjust how many batches to evaluate on.
            num_eval_batches = (
                evaluator.subset_num_batches
                if evaluator.subset_num_batches is not None
                else self.cfg.eval_subset_num_batches
            )
            if num_eval_batches > 0:
                num_eval_batches = min(num_eval_batches, len(evaluator.eval_loader))
                eval_batches = islice(eval_batches, num_eval_batches)

            # Run model over batches.
            for eval_step, eval_batch in enumerate(eval_batches):
                self.eval_step(eval_batch, evaluator)

                # Log to console.
                if eval_step + 1 == num_eval_batches or (eval_step + 1) % self.cfg.console_log_interval == 0:
                    log.info(f"[eval_step={eval_step + 1}/{num_eval_batches}]")

            # Get final metrics.
            metrics = evaluator.compute_metrics()
            eval_metrics.update(metrics)
            self.log_metrics_to_console(f"{evaluator.label}", metrics)

            del eval_batches

        return eval_metrics

    def check_if_cancelled(self) -> Tuple[bool, int]:
        should_cancel = False
        cancel_reason: Optional[str] = None
        extra_steps = 0
        if get_global_rank() == 0:
            if self.cfg.time_limit is not None and time.time() - self._start_time >= self.cfg.time_limit:
                # First check if we've reached the training time limit.
                should_cancel = True
                cancel_reason = "time limit reached"
                extra_steps = self.cfg.extra_steps_after_cancel
            elif (
                self.cfg.early_stopping_factor is not None
                and self.global_step > self.cfg.scheduler.t_warmup
                and self.cur_train_loss > self.cfg.early_stopping_factor * self.min_train_loss
            ):
                # Next check if early stopping loss criteria is met.
                should_cancel = True
                cancel_reason = "early stopping from loss increase"
            elif wandb.run is not None and (api_key := os.environ.get("WANDB_API_KEY")) is not None:
                # Finally, check if someone canceled the run from W&B by adding the 'cancel' / 'canceled' tag..
                # We won't see it in the run object. So we have to use the import/export API to check.
                from requests.exceptions import RequestException
                from wandb.errors import CommError

                try:
                    api = wandb.Api(api_key=api_key)
                    run = api.run(wandb.run.path)
                    for tag in run.tags or []:
                        if tag.lower() in {"cancel", "canceled", "cancelled"}:
                            should_cancel = True
                            cancel_reason = "Weights & Biases tag"
                            extra_steps = self.cfg.extra_steps_after_cancel
                            break
                except (RequestException, CommError):
                    log.info("Failed to check if W&B run is cancelled, continuing run.")

        run_canceled = synchronize_flag(should_cancel, self.device)
        if run_canceled:
            extra_steps = synchronize_value(extra_steps, self.device)
            if cancel_reason is None:
                if extra_steps > 0:
                    log.warning(f"Run canceled, stopping in {extra_steps} more steps...")
                else:
                    log.warning("Run canceled")
            else:
                if extra_steps > 0:
                    log.warning(f"Run canceled due to {cancel_reason}, stopping in {extra_steps} more steps...")
                else:
                    log.warning(f"Run canceled due to {cancel_reason}")

        return run_canceled, extra_steps

    def _initialize_training(self):
        """Initialize training parameters and state."""
        # Set stopping points
        if self.cfg.stop_after is not None:
            if self.cfg.stop_at is None:
                self.cfg.stop_at = self.global_step + self.cfg.stop_after
            else:
                self.cfg.stop_at = min(self.cfg.stop_at, self.global_step + self.cfg.stop_after)
        if self.cfg.stop_at is None:
            self.cfg.stop_at = self.max_steps + 10

        # Initialize time tracking and state
        self._start_time = time.time()
        self._gc_init_state = gc.isenabled()
        self._cancel_initiated = False
        self._stop_at = self.cfg.stop_at
        self._save_checkpoints = True

        # Disable automatic garbage collection if configured
        if self.cfg.gen1_gc_interval is not None:
            gc.disable()

    def _run_initial_evaluation(self):
        """Run evaluation at the start of training if required."""
        if self.cfg.load_path is not None and self.global_step > 0 and self.cfg.eval_on_load:
            eval_metrics = self.eval()
            if wandb.run is not None:
                eval_metrics["global_step"] = self.global_step
                wandb.log(eval_metrics, step=self.global_step)

    def _initialize_monitors(self):
        """Initialize monitoring utilities."""
        assert self.cfg.device_train_batch_size is not None
        speed_monitor = SpeedMonitor(self.cfg.speed_monitor)
        lr_monitor = LRMonitor(self.optim)

        # Log initial system metrics
        sys_metrics = self.system_metrics()
        if sys_metrics:
            self.log_metrics_to_console("Pre-train system metrics", sys_metrics)
            if wandb.run is not None:
                sys_metrics["global_step"] = 0
                wandb.log(sys_metrics, step=0)

        return speed_monitor, lr_monitor

    def _setup_profilers(self):
        """Set up profiling tools if enabled."""
        # Set up Python profiler
        self._python_profiler = cProfile.Profile() if self.cfg.python_profiling else None

        # Set up PyTorch profiler if enabled
        if self.cfg.torch_profiling and get_global_rank() == 0:
            from torch.profiler import schedule, ProfilerActivity

            # Configure profiler with schedule and callbacks
            profiling_schedule = schedule(
                wait=self.cfg.profiler_config.get("wait", 1),
                warmup=self.cfg.profiler_config.get("warmup", 5),
                active=self.cfg.profiler_config.get("active", 3),
                repeat=self.cfg.profiler_config.get("repeat", 1),
            )

            # Define trace output handler
            def on_trace_ready(p):
                profiler_output_dir = Path(self.cfg.save_folder) / "profiler"
                profiler_output_dir.mkdir(exist_ok=True)
                # Output profiling summaries
                output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=32)
                log.info(f"Profile by total GPU time at step {p.step_num}:\n{output}")
                output = p.key_averages().table(sort_by="self_cpu_time_total", row_limit=32)
                log.info(f"Profile by total CPU time at step {p.step_num}:\n{output}")
                # Save trace file
                trace_path = profiler_output_dir / f"{p.step_num}.chrome_trace.json.gz"
                p.export_chrome_trace(str(trace_path))
                # Upload if remote save configured
                if self.cfg.remote_save_folder is not None:
                    upload_folder = f"{self.cfg.remote_save_folder.rstrip('/')}/profiler"
                    log.info(f"Tracing complete, uploading results to '{upload_folder}'...")
                    upload(trace_path, f"{upload_folder}/{trace_path.name}")

            # Create profiler with configuration
            torch_profiler = torch.profiler.profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=False,
                profile_memory=False,
                with_stack=True,
                schedule=profiling_schedule,
                on_trace_ready=on_trace_ready,
            )
        else:
            import contextlib

            torch_profiler = contextlib.nullcontext()

        return torch_profiler

    def _check_data_stage_transition(self):
        """Check and handle transitions between data stages."""
        stage_name = self.start_global_step_to_stage_name.get(self.global_step)
        if stage_name is not None and stage_name != self.current_stage_name:
            log.info(f"Switching to data stage {stage_name} at step {self.global_step}")

            # Save checkpoint before switching data stages
            try:
                log.info("Saving checkpoint before data stage transition...")
                if self.cfg.save_unshared_data_stage_checkpoints:
                    checkpoint_path, _ = self.save_checkpoint(CheckpointType.unsharded)
                    log.info(f"Data stage transition checkpoint saved to {checkpoint_path}")
            except Exception as e:
                log.error(f"Failed to save checkpoint during data stage transition: {e}")
                # Continue with stage transition despite checkpoint failure

            self.current_stage_name = stage_name
            # Clean up old iterator
            self._cleanup_current_iterator()
            # Initialize new loader
            try:
                if isinstance(self.train_loaders[stage_name], DataLoader):
                    self.current_stage_loader = self.train_loaders[stage_name]
                else:
                    self.current_stage_loader = self.train_loaders[stage_name]()
                    self.train_loaders[stage_name] = self.current_stage_loader
                # Create iterator for new loader
                self.current_stage_loader_iter = iter(self.current_stage_loader)
            except Exception as e:
                log.error(f"Failed to initialize data stage {stage_name}: {e}")
                raise RuntimeError(f"Failed to initialize data stage {stage_name}") from e
        barrier()  # Ensure all ranks have completed the transition

    def _get_next_batch(self):
        """Get the next batch from the current data loader."""
        try:
            if self.current_stage_loader_iter is None:
                log.warning("Data loader iterator is None, initializing new iterator")
                self.current_stage_loader_iter = iter(self.current_stage_loader)
            return next(self.current_stage_loader_iter)
        except StopIteration:
            # Signal end of epoch
            return None

    def _end_epoch(self, epoch):
        """Handle end of epoch tasks."""
        log.info("Training epoch complete")
        self.epoch = epoch + 1
        self.global_train_examples_seen_this_epoch = 0
        self.dataset.start_index = 0
        # Clean up current iterator
        self._cleanup_current_iterator()
        # Prepare for next epoch if needed
        if self.epoch < self.max_epochs:
            log.info(f"Reshuffling data loader for epoch {self.epoch}...")
            self.dataset.reshuffle(self.epoch)
            # Create new iterator
            self.current_stage_loader_iter = iter(self.current_stage_loader)

    def _process_batch(self, batch, speed_monitor, lr_monitor, first_batch):
        """Process a single training batch."""
        # Validate batch dimensions
        self._validate_batch(batch)

        # Update training metrics
        batch_size, seq_len = batch["input_ids"].shape
        global_batch_size = batch_size * get_world_size()
        self.global_step += 1
        self.global_train_examples_seen_this_epoch += global_batch_size
        self.global_train_tokens_seen += global_batch_size * seq_len

        # Update speed monitoring
        speed_monitor.batch_start(
            global_total_tokens=self.global_train_tokens_seen,
            device_batch_num_tokens=batch_size * seq_len,
            num_fwd_flops=self.model.num_fwd_flops,
            num_bck_flops=self.model.num_bck_flops,
            record=not first_batch,
        )

        # Determine if metrics should be logged
        should_log_this_step = self.should_log_this_step()

        # Execute training step
        try:
            metrics = self.train_step(batch, reduce_global_loss=should_log_this_step)
        except Exception as e:
            log.error(f"Error in training step: {e}")
            # Try to save state before re-raising
            self._emergency_checkpoint()
            raise

        # Process metrics if needed
        if should_log_this_step:
            self._process_metrics(metrics, speed_monitor, lr_monitor)

        # Handle checkpointing and evaluation
        self._handle_checkpoints_and_eval(speed_monitor)

        # Check for Python profiler actions
        self._handle_python_profiler()

    def _validate_batch(self, batch):
        """Validate batch dimensions and structure."""
        batch_size, seq_len = batch["input_ids"].shape
        if seq_len != self.cfg.model.max_sequence_length:
            raise ValueError(
                f"Unexpected sequence length: got {seq_len}, " f"expected {self.cfg.model.max_sequence_length}"
            )
        if batch_size != self.cfg.device_train_batch_size:
            raise ValueError(
                f"Unexpected batch size: got {batch_size}, " f"expected {self.cfg.device_train_batch_size}"
            )

    def _process_metrics(self, metrics, speed_monitor, lr_monitor):
        """Process and log training metrics."""
        # Add additional metrics
        metrics.update(speed_monitor.check())
        metrics.update(self.system_metrics())
        metrics.update(lr_monitor.check())

        # Log to console if interval reached
        if self.global_step % self.cfg.console_log_interval == 0:
            if get_global_rank() == 0:
                self.log_metrics_to_console(
                    f"[step={self.global_step}/{self.max_steps},epoch={self.epoch or 0}]",
                    metrics,
                )
            else:
                log.info(f"[step={self.global_step}/{self.max_steps},epoch={self.epoch or 0}]")

        # Log to W&B if enabled
        if (
            wandb.run is not None
            and self.cfg.wandb is not None
            and self.global_step % self.cfg.wandb.log_interval == 0
        ):
            metrics["global_step"] = self.global_step
            wandb.log(metrics, step=self.global_step)

        # Check for cancellation
        if not self._cancel_initiated and self.global_step % self.cfg.canceled_check_interval == 0:
            self._cancel_initiated, extra_steps = self.check_if_cancelled()
            if self._cancel_initiated:
                self._stop_at = min(self._stop_at, self.global_step + extra_steps)

    def _handle_checkpoints_and_eval(self, speed_monitor):
        """Handle checkpoint saving and evaluation."""
        # Save sharded checkpoint if needed
        self._maybe_save_sharded_checkpoint(speed_monitor)

        # Save unsharded checkpoint if needed
        self._maybe_save_unsharded_checkpoint(speed_monitor)

        # Run evaluation if needed
        if not self._cancel_initiated and (
            self.global_step % self.cfg.eval_interval == 0 or self.global_step >= self._stop_at
        ):
            self._run_evaluation(speed_monitor)

    def _maybe_save_sharded_checkpoint(self, speed_monitor):
        """Save sharded checkpoint if conditions are met."""
        if self.cfg.distributed_strategy == DistributedStrategy.ddp:
            return

        if self._save_checkpoints and (
            self._cancel_initiated
            or (
                self.cfg.save_interval is not None
                and self.global_step % self.cfg.save_interval == 0
                and self.cfg.save_num_checkpoints_to_keep != 0
            )
        ):
            try:
                log.info("Saving checkpoint...")
                checkpoint_path, _ = self.save_checkpoint(CheckpointType.sharded)
                log.info(f"Checkpoint saved to {checkpoint_path}")

                # Remove ephemeral checkpoints
                while self.ephemeral_checkpoints:
                    self.remove_ephemeral_checkpoint()

                # Reset speed monitor
                speed_monitor.reset()

                # Update state if canceled
                if self._cancel_initiated:
                    self._save_checkpoints = False
            except Exception as e:
                log.error(f"Failed to save checkpoint: {e}")
                # Continue training despite checkpoint failure

        elif (
            self.cfg.save_interval_ephemeral is not None
            and self.global_step % self.cfg.save_interval_ephemeral == 0
        ):
            try:
                log.info("Saving ephemeral checkpoint...")
                checkpoint_path, _ = self.save_checkpoint(CheckpointType.sharded_ephemeral)
                log.info(f"Checkpoint saved to {checkpoint_path}")

                # Reset speed monitor
                speed_monitor.reset()
            except Exception as e:
                log.error(f"Failed to save ephemeral checkpoint: {e}")
                # Continue training despite checkpoint failure

    def _maybe_save_unsharded_checkpoint(self, speed_monitor):
        """Save unsharded checkpoint if conditions are met."""
        if (
            self._save_checkpoints
            and self.cfg.save_interval_unsharded is not None
            and self.global_step % self.cfg.save_interval_unsharded == 0
            and self.cfg.save_num_unsharded_checkpoints_to_keep != 0
        ):
            try:
                log.info("Saving unsharded checkpoint...")
                checkpoint_path, _ = self.save_checkpoint(CheckpointType.unsharded)
                log.info(f"Unsharded checkpoint saved to {checkpoint_path}")

                # Reset speed monitor
                speed_monitor.reset()
            except Exception as e:
                log.error(f"Failed to save unsharded checkpoint: {e}")
                # Continue training despite checkpoint failure

    def _run_evaluation(self, speed_monitor):
        """Run evaluation and log results."""
        try:
            eval_metrics = self.eval()

            # Log metrics to W&B
            if wandb.run is not None:
                eval_metrics["global_step"] = self.global_step
                wandb.log(eval_metrics, step=self.global_step)

            # Reset speed monitor
            speed_monitor.reset()

            # Reset model to train mode
            self.dist_model.train()
        except Exception as e:
            log.error(f"Error during evaluation: {e}")
            # Continue training despite evaluation failure

    def _collect_garbage_if_needed(self):
        """Collect garbage if interval is reached."""
        if self.cfg.gen1_gc_interval is not None and self.global_step % self.cfg.gen1_gc_interval == 0:
            gc.collect(1)

    def _handle_python_profiler(self):
        """Handle Python profiler state transitions."""
        if self._python_profiler is not None:
            try:
                # Enable profiler at specified step
                if self.global_step == self.cfg.profiler_config.get("start_step", 5):
                    self._python_profiler.enable()

                # Disable and output results at specified step
                if self.global_step >= self.cfg.profiler_config.get("end_step", 8):
                    self._python_profiler.disable()
                    self._python_profiler.print_stats(sort=SortKey.CUMULATIVE)
                    self._python_profiler = None
            except Exception as e:
                log.error(f"Error in Python profiler: {e}")
                # Continue training despite profiler issues

    def _should_stop_training(self):
        """Determine if training should stop."""
        if self.global_step >= self.max_steps:
            log.info(f"Reached max_steps ({self.max_steps}), stopping training")
            return True
        if self.global_step >= self._stop_at:
            log.info(f"Reached stop_at step ({self._stop_at}), stopping training")
            return True
        return False

    def _save_final_checkpoint(self):
        """Save final checkpoint at end of training."""
        if not self._save_checkpoints:
            return

        try:
            if (
                self.cfg.save_interval_unsharded is not None
                and self.last_unsharded_checkpoint_step != self.global_step
            ):
                log.info("Saving final unsharded model checkpoint...")
                checkpoint_path, _ = self.save_checkpoint(CheckpointType.unsharded)
                log.info(f"Unsharded checkpoint saved to {checkpoint_path}")
            elif (
                self.cfg.save_num_checkpoints_to_keep != 0
                and self.last_sharded_checkpoint_step != self.global_step
                and self.cfg.distributed_strategy == DistributedStrategy.fsdp
            ):
                log.info("Saving final checkpoint...")
                checkpoint_path, _ = self.save_checkpoint(CheckpointType.sharded)
                log.info(f"Checkpoint saved to {checkpoint_path}")
        except Exception as e:
            log.error(f"Failed to save final checkpoint: {e}")

    def _emergency_checkpoint(self):
        """Save emergency checkpoint when errors occur."""
        try:
            if self._save_checkpoints and self.global_step > 0:
                log.warning("Attempting to save emergency checkpoint...")
                checkpoint_path, _ = self.save_checkpoint(
                    CheckpointType.sharded
                    if self.cfg.distributed_strategy != DistributedStrategy.ddp
                    else CheckpointType.unsharded
                )
                log.warning(f"Emergency checkpoint saved to {checkpoint_path}")
        except Exception as e:
            log.error(f"Failed to save emergency checkpoint: {e}")

    def _cleanup_resources(self):
        """Clean up resources when training ends."""
        gc_cuda()

        # Clean up current iterator
        self._cleanup_current_iterator()

        # Close all indices files
        for file in self.indices_files.values():
            try:
                # Check if file is still open before attempting to close
                if not file.closed:
                    file.flush()
                    file.close()
            except Exception as e:
                log.error(f"Error closing indices file: {e}")

        # Clean up distributed process groups
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception as e:
            log.error(f"Error destroying process group: {e}")

        # Restore garbage collection state
        if self._gc_init_state:
            gc.enable()
        else:
            gc.disable()

    def fit(self):
        """Main training loop."""
        try:
            # Initialize training parameters
            self._initialize_training()

            # Run initial evaluation if needed
            self._run_initial_evaluation()

            # Set model to train mode
            self.dist_model.train()

            # Initialize training state tracking
            speed_monitor, lr_monitor = self._initialize_monitors()
            first_batch = True

            # Set up profilers if enabled
            with self._setup_profilers() as profiler:
                # Main epoch loop
                for epoch in range(self.epoch or 0, self.max_epochs):
                    if self._should_stop_training():
                        break

                    # Epoch training loop
                    while True:
                        try:
                            # Check for data stage transitions
                            self._check_data_stage_transition()

                            # Ensure data loader is active
                            if self.current_stage_loader is None:
                                raise RuntimeError("No data loader is currently active")

                            # Get next batch or end epoch if exhausted
                            batch = self._get_next_batch()
                            if batch is None:  # End of epoch
                                self._end_epoch(epoch)
                                break

                            # Process batch and update training state
                            self._process_batch(batch, speed_monitor, lr_monitor, first_batch)
                            first_batch = False

                            # Update profiler if active
                            if profiler is not None:
                                profiler.step()

                            # Collect garbage if needed
                            self._collect_garbage_if_needed()

                            # Check for stopping conditions
                            if self._should_stop_training():
                                break

                        except StopIteration:
                            # Handle end of dataset
                            self._end_epoch(epoch)
                            break
                        except Exception as e:
                            # Log unexpected errors but try to save state before raising
                            log.error(f"Error during training: {e}")
                            self._emergency_checkpoint()
                            raise

                    # Break out of epoch loop if stopping
                    if self._should_stop_training():
                        break

            # Save final checkpoint
            self._save_final_checkpoint()

        except Exception as e:
            # Handle any unhandled exceptions
            log.exception(f"Unhandled exception during training: {e}")
            self._emergency_checkpoint()
            raise
        finally:
            # Ensure resources are cleaned up
            self._cleanup_resources()

    def __enter__(self) -> "Trainer":
        """Enable use of trainer as a context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Clean up resources when exiting the context manager."""
        del exc_val, exc_tb
        self.close(0 if exc_type is None else 1)

    def close(self, exit_code: int = 0) -> None:
        """Clean up resources when training ends."""
        # Call the cleanup method that contains the resource cleanup logic
        self._cleanup_resources()

        # Finish wandb run if active
        if wandb.run is not None:
            wandb.finish(exit_code=exit_code, quiet=True)
