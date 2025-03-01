"""Run this script with 'torchrun'."""

import logging
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from itertools import islice

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.nn.parallel import DistributedDataParallel as DDP

from olmo.train import cross_entropy_loss
from olmo.config import (
    DistributedStrategy,
    EvalConfig,
    TrainConfig,
)
from olmo.exceptions import OLMoCliError, OLMoConfigurationError
from olmo.model import OLMo
from olmo.torch_util import (
    barrier,
    get_default_device,
    get_global_rank,
    get_local_rank,
    get_world_size,
    peak_gpu_memory,
    seed_all,
    move_to_device,
)
from olmo.eval import build_evaluators, Evaluator
from olmo.util import (
    add_cached_path_clients,
    clean_opt,
    prepare_cli_environment,
)
import numpy as np

log = logging.getLogger("eval")


# mm: this method is copied from the Trainer class
def _get_labels(batch: Dict[str, Any]) -> torch.Tensor:
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


# mm: this method is copied from the Trainer class
def _model_forward(
    dist_model: torch.nn.Module,
    batch: Dict[str, Any],
    loss_reduction: str = "mean",
    compute_z_loss: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    # shape: (batch_size, seq_len, vocab_size)
    logits = dist_model(
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
    original_labels = _get_labels(batch)
    # shape: (batch_size * seq_len,)
    labels = original_labels.view(-1)
    ce_loss, z_loss = cross_entropy_loss(
        logits_for_loss, labels, ignore_index=-100, reduction=loss_reduction, compute_z_loss=compute_z_loss
    )
    if loss_reduction == "none":
        # Reshape (batch_size * seq_len,) -> (batch_size, seq_len)
        ce_loss = ce_loss.view(batch["input_ids"].shape[0], -1)
        if z_loss is not None:
            z_loss = z_loss.view(batch["input_ids"].shape[0], -1)
    return ce_loss, z_loss, logits, original_labels


# mm: this method is copied from the Trainer class
def _eval(
    model: OLMo, evaluators: List[Evaluator], cfg: EvalConfig, device: torch.DeviceObjType
) -> Dict[str, Any]:
    # Zero gradients and set model to 'eval' mode.
    model.eval()

    eval_metrics = {}
    all_losses_per_evaluator = {}
    all_labels_per_evaluator = {}
    for evaluator in evaluators:
        log.info(f"Running evaluation for '{evaluator.label}'...")

        # Add list to collect losses
        all_losses_per_evaluator[evaluator.label] = []
        all_labels_per_evaluator[evaluator.label] = []
        # Reset metrics.
        evaluator.reset_metrics()

        # Initialize data loader iterator.
        eval_batches = iter(evaluator.eval_loader)

        # Adjust how many batches to evaluate on.
        num_eval_batches = (
            evaluator.subset_num_batches
            if evaluator.subset_num_batches is not None
            else cfg.eval_subset_num_batches
        )
        if num_eval_batches > 0:
            num_eval_batches = min(num_eval_batches, len(evaluator.eval_loader))
            eval_batches = islice(eval_batches, num_eval_batches)

        # Run model over batches.
        for eval_step, eval_batch in enumerate(eval_batches):
            # Move tensors to the right device.
            batch = move_to_device(eval_batch, device)

            # Run forward pass.
            with torch.no_grad():  # NOTE: 'torch.inference_mode()' doesn't work with 'torch.compile()'.
                with torch.autocast("cuda", enabled=True, dtype=cfg.autocast_precision):
                    ce_loss, _, logits, labels = _model_forward(model, batch, loss_reduction="none")

                    # Store the position-wise losses
                    # ce_loss shape: (batch_size, seq_len)
                    all_losses_per_evaluator[evaluator.label].append(ce_loss.detach().cpu())
                    all_labels_per_evaluator[evaluator.label].append(labels.detach().cpu())
            # Update metrics.
            evaluator.update_metrics(
                batch, ce_loss, logits
            )  # batch includes all keys that the downstream evaluation needs

            barrier()

            # Log to console.
            if eval_step + 1 == num_eval_batches or (eval_step + 1) % cfg.console_log_interval == 0:
                log.info(f"[eval_step={eval_step + 1}/{num_eval_batches}]")

        # Concatenate all losses before computing final metrics
        combined_losses = torch.cat(all_losses_per_evaluator[evaluator.label], dim=0)
        all_losses_per_evaluator[evaluator.label] = combined_losses
        combined_labels = torch.cat(all_labels_per_evaluator[evaluator.label], dim=0)
        all_labels_per_evaluator[evaluator.label] = combined_labels

        metrics = evaluator.compute_metrics()
        eval_metrics.update(metrics)
        del eval_batches, combined_losses, combined_labels  # Clean up memory

    return eval_metrics, all_losses_per_evaluator, all_labels_per_evaluator


def main(cfg: EvalConfig) -> None:
    # Ensure checkpoint path is set
    if cfg.load_path is None:
        raise OLMoConfigurationError("--load_path is required")

    barrier()

    device = torch.device("cuda")

    # Fill some configuration options
    cfg.device_batch_size = cfg.global_eval_batch_size // get_world_size()

    barrier()

    # Set seed
    seed_all(cfg.seed)

    # Load the model
    log.info(f"Loading model from checkpoint {cfg.load_path}...")
    olmo_model = OLMo.from_checkpoint(cfg.load_path)
    log.info(f"Total number of parameters: {olmo_model.num_params():,d}")
    log.info(f"Peak GPU Memory (MB) before {cfg.distributed_strategy}: {int(peak_gpu_memory() or 0)}")

    if cfg.distributed_strategy == DistributedStrategy.ddp:
        log.info("Wrapping model with DDP...")
        if cfg.init_device != "cuda":
            raise OLMoConfigurationError("DDP does not work with init_device set to anything other than `cuda`.")

        # move to cuda before calling ddp
        dist_model = DDP(olmo_model.to(device))
    elif cfg.distributed_strategy == DistributedStrategy.fsdp:
        log.info("Wrapping model with FSDP...")
        wrap_policy = olmo_model.get_fsdp_wrap_policy(cfg.fsdp.wrapping_strategy)

        dist_model = FSDP(
            olmo_model,
            sharding_strategy=cfg.fsdp.sharding_strategy,
            mixed_precision=cfg.fsdp_precision,
            auto_wrap_policy=wrap_policy,
            use_orig_params=cfg.fsdp.use_orig_params,
            limit_all_gathers=True,
            device_id=get_local_rank(),
        )
    else:
        raise NotImplementedError("Single accelerator evaluation not implemented yet!")

    log.info(f"Peak GPU Memory (MB) after {cfg.distributed_strategy}: {int(peak_gpu_memory() or 0)}")

    # Construct evaluators
    evaluators = build_evaluators(cfg, device)

    barrier()

    # Run evaluation
    log.info("Starting evaluation...")
    metrics, all_losses_per_evaluator, all_labels_per_evaluator = _eval(dist_model, evaluators, cfg, device)

    # Print results on rank 0
    if get_global_rank() == 0:
        log.info("Evaluation Results:")
        log.info(metrics)
        for evaluator in evaluators:
            log.info(f"Position-wise losses for '{evaluator.label}':")
            log.info(all_losses_per_evaluator[evaluator.label].shape)

            # save the losses to a file
            output_dir = cfg.load_path
            torch.save(all_losses_per_evaluator[evaluator.label], f"{output_dir}/{evaluator.label}_losses.pt")
            torch.save(all_labels_per_evaluator[evaluator.label], f"{output_dir}/{evaluator.label}_labels.pt")
    # Add cleanup of process group
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError as e:
        print(f"failed to set multiprocessing start method: {e}")
    log.info(f"Multiprocessing start method set to '{mp.get_start_method()}'")

    # Set CUDA device
    torch.cuda.set_device(f"cuda:{get_local_rank()}")

    # Initialize process group
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
    log.info("Process group initialized")

    prepare_cli_environment()
    log.info("CLI environment prepared")

    add_cached_path_clients()

    try:
        yaml_path, args_list = sys.argv[1], sys.argv[2:]
    except IndexError:
        raise OLMoCliError(f"Usage: {sys.argv[0]} [CONFIG_PATH] [OPTIONS]")

    # load config files
    cfg = EvalConfig.load(yaml_path, [clean_opt(s) for s in args_list])

    # also load the training config in the load_path
    train_cfg = TrainConfig.load(Path(cfg.load_path) / "config.yaml")

    # add the training config to the eval config
    cfg.model = train_cfg.model

    main(cfg)
