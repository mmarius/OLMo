"""Run this script with 'torchrun'."""

import os
import gzip
import logging
import hashlib
import sys
from datetime import timedelta
from pathlib import Path
from typing import Optional, TextIO
import json
import shutil

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb
from packaging import version
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.nn.parallel import DistributedDataParallel as DDP

from olmo.config import (
    CheckpointType,
    DDPGradSyncMode,
    DistributedStrategy,
    TrainConfig,
)
from olmo.data import build_train_dataloader
from olmo.eval import build_evaluators
from olmo.exceptions import OLMoCliError, OLMoConfigurationError
from olmo.model import OLMo
from olmo.optim import BoltOnWarmupScheduler, build_optimizer, build_scheduler
from olmo.torch_util import (
    barrier,
    get_default_device,
    get_global_rank,
    get_local_rank,
    get_local_world_size,
    get_world_size,
    peak_gpu_memory,
    seed_all,
)
from olmo.train import Trainer
from olmo.util import (
    add_cached_path_clients,
    clean_opt,
    find_latest_checkpoint,
    log_extra_field,
    prepare_cli_environment,
)

log = logging.getLogger("train")


def main(cfg: TrainConfig) -> None:
    # Ensure run name set.
    if cfg.run_name is None:
        raise OLMoConfigurationError("--run_name is required")
    log_extra_field("run_name", cfg.run_name)

    # Sanity check
    if (cfg.reset_optimizer_state or cfg.reset_trainer_state) and cfg.load_path is None:
        log.warning(
            "You want to reset the optimizer or trainer state, but we're not loading from the checkpoint. The"
            "setting has no effect."
        )

    barrier()

    device = torch.device("cuda")

    # Fill some configuration options.
    cfg.model.precision = cfg.precision
    cfg.device_train_batch_size = cfg.global_train_batch_size // get_world_size()
    assert cfg.device_train_batch_size is not None  # for mypy
    cfg.device_train_grad_accum = cfg.device_train_batch_size // cfg.device_train_microbatch_size
    if cfg.optimizer.no_decay_norm_and_bias is not None:
        log.warning(
            "You set the deprecated config option `no_decay_norm_and_bias`. For compatibility, this"
            "setting will take precedence over all other weight decay configurations. Please change"
            "your config to use `decay_norm_and_bias` and `decay_embeddings` instead."
        )
        cfg.optimizer.decay_norm_and_bias = not cfg.optimizer.no_decay_norm_and_bias
        cfg.optimizer.decay_embeddings = not cfg.optimizer.no_decay_norm_and_bias
        cfg.optimizer.no_decay_norm_and_bias = None  # So nobody uses this by accident.

    # Create a unique run name based on the config
    # Compute a hash of the config
    config_hash = hashlib.sha256(str(cfg.asdict()).encode()).hexdigest()
    config_hash = config_hash[:16]  # truncate the hash to 16 characters
    RUN_NAME = f"{cfg.run_name}-{config_hash}"

    # Update the save_folder
    cfg.parent_save_folder = cfg.save_folder
    cfg.save_folder = str(Path(cfg.parent_save_folder) / RUN_NAME)

    # Check if output folder exists and try to find latest checkpoint
    wandb_resume = cfg.wandb.resume if cfg.wandb is not None else "must"
    if Path(cfg.save_folder).exists() and cfg.load_path is None:
        log.info(f"Output folder {cfg.save_folder} already exists, searching for latest checkpoint...")
        if (checkpoint_dir := find_latest_checkpoint(cfg.save_folder)) is not None:
            log.info(f"Found latest checkpoint at {checkpoint_dir}")
            cfg.load_path = str(checkpoint_dir)
            if "step0-" in cfg.load_path:
                # in this case we need to change the wandb.resume arg to "must"
                wandb_resume = "must"
        else:
            # In this case, the checkpoint folder exists but contains to checkpoints
            # Hence, we will start training effectively from scratch
            # Therefore, we delete everything in the save_folder
            log.warning("No checkpoint found in existing output folder")
            # delete everything in the save_folder
            for file in Path(cfg.save_folder).glob("*"):
                if file.is_file():
                    file.unlink()
                elif file.is_dir():
                    shutil.rmtree(file)
            # in this case we need to change the wandb.resume arg to "must"
            wandb_resume = "must"

    # Display and save configuration.
    if get_global_rank() == 0:
        if cfg.data.paths is not None and len(cfg.data.paths) < 50:
            log.info("Configuration:")
            log.info(cfg)
        if not cfg.dry_run and (cfg.load_path is None or Path(cfg.load_path).parent != Path(cfg.save_folder)):
            # Save config.
            save_path = Path(cfg.save_folder) / "config.yaml"
            if save_path.is_file() and not cfg.save_overwrite:
                raise OLMoConfigurationError(f"{save_path} already exists, use --save_overwrite to overwrite")
            else:
                log.info(f"Saving config to {save_path}")
                save_path.parent.mkdir(exist_ok=True, parents=True)
                cfg.save(save_path)
            del save_path

    barrier()

    # Set seed.
    seed_all(cfg.seed)

    # Construct data loader.
    train_loader = build_train_dataloader(cfg)

    # Construct evaluators.
    evaluators = build_evaluators(cfg, device)
    barrier()

    # Initialize the model.
    log.info("Building model...")
    olmo_model = OLMo(cfg.model)
    total_params = olmo_model.num_params()
    total_non_embedding_params = olmo_model.num_params(include_embedding=False)
    log.info(f"Total number of parameters: {total_params:,d}")
    log.info(f"Number of non-embedding parameters: {total_non_embedding_params:,d}")
    log.info(f"Peak GPU Memory (MB) before {cfg.distributed_strategy}: {int(peak_gpu_memory() or 0)}")

    # Maybe start W&B run.
    if cfg.wandb is not None and (get_global_rank() == 0 or not cfg.wandb.rank_zero_only):
        wandb_dir = Path(cfg.save_folder) / "wandb"
        wandb_dir.mkdir(parents=True, exist_ok=True)

        # convert config to dict and remove wandb
        config_dict = cfg.asdict(exclude=["wandb"])
        config_dict["total_params"] = total_params
        config_dict["total_non_embedding_params"] = total_non_embedding_params

        wandb.init(
            dir=wandb_dir,
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            group=cfg.wandb.group,
            name=RUN_NAME,
            tags=cfg.wandb.tags,
            config=config_dict,
            resume=wandb_resume,
            id=RUN_NAME,
        )

    barrier()

    olmo_model.set_activation_checkpointing(cfg.activation_checkpointing)

    if cfg.distributed_strategy == DistributedStrategy.ddp:
        log.info("Wrapping model with DDP...")
        assert cfg.ddp is not None, "DistributedStrategy ddp needs cfg.ddp to be set!"

        if cfg.model.init_device != "cuda":
            raise OLMoConfigurationError("DDP does not work with init_device set to anything other than `cuda`.")

        if cfg.ddp.find_unused_params is True and cfg.ddp.grad_sync_mode != DDPGradSyncMode.micro_batch:
            raise OLMoConfigurationError(
                "`find_unused_params` is set to True. DDP needs to synchronize gradients for every micro-batch to avoid errors. Set `grad_sync_mode` to `micro_batch`."
            )

        param_init_fn = None

        # move to cuda before calling ddp
        dist_model = DDP(olmo_model.to(device), find_unused_parameters=cfg.ddp.find_unused_params)
    elif cfg.distributed_strategy == DistributedStrategy.fsdp:
        # Wrap the model in FSDP.
        log.info("Wrapping model with FSDP...")
        assert cfg.fsdp is not None, "DistributedStrategy fsdp needs cfg.fsdp to be set!"
        wrap_policy = olmo_model.get_fsdp_wrap_policy(cfg.fsdp.wrapping_strategy)

        if version.parse(torch.__version__) >= version.parse("2.1.0"):
            # This prevents any parameters from being initialized twice
            def dummy_init_fn(module: torch.nn.Module) -> None:
                module.to_empty(device=get_default_device())

            param_init_fn = dummy_init_fn
        else:
            param_init_fn = None

        # Set up device mesh for hybrid sharding in order to specify which nodes are assoicated to a given model replica
        device_mesh = None
        hybrid_sharding_fsdp_kwargs = {}
        if cfg.fsdp.sharding_strategy in (ShardingStrategy.HYBRID_SHARD, ShardingStrategy._HYBRID_SHARD_ZERO2):
            if version.parse(torch.__version__) < version.parse("2.2.0"):
                # Device mesh was not added to PyTorch until v2.2.0
                raise OLMoConfigurationError(
                    "OLMo training does not correctly support hybrid sharding before torch 2.2.0"
                )

            from torch.distributed.device_mesh import init_device_mesh

            num_model_replicas = cfg.fsdp.hybrid_sharding_num_model_replicas or (
                get_world_size() // get_local_world_size()
            )

            if num_model_replicas <= 0:
                raise OLMoConfigurationError("fsdp.hybrid_sharding_num_model_replicas must be a positive integer")

            if get_world_size() % num_model_replicas != 0:
                raise OLMoConfigurationError("fsdp.hybrid_sharding_num_model_replicas must divide world size")

            device_mesh = init_device_mesh("cuda", (num_model_replicas, get_world_size() // num_model_replicas))
            hybrid_sharding_fsdp_kwargs["device_mesh"] = device_mesh

        dist_model = FSDP(
            olmo_model,
            sharding_strategy=cfg.fsdp.sharding_strategy,
            mixed_precision=cfg.fsdp_precision,
            auto_wrap_policy=wrap_policy,
            use_orig_params=cfg.fsdp.use_orig_params,  # needed for compile and some of our optimizer/parameter metrics
            limit_all_gathers=True,
            device_id=get_local_rank(),
            param_init_fn=param_init_fn,
            **hybrid_sharding_fsdp_kwargs,
        )
    elif cfg.distributed_strategy is None:
        raise NotImplementedError("Single accelerator training not implemented yet!")

    # when param_init_fn is None, FSDP will call reset_parameters() automatically
    if param_init_fn is not None or cfg.distributed_strategy == DistributedStrategy.ddp:
        olmo_model.reset_parameters()

    log.info(f"Peak GPU Memory (MB) after {cfg.distributed_strategy}: {int(peak_gpu_memory() or 0)}")
    log.info("Model:")
    log.info(dist_model)

    # Construct optimizer and learning rate scheduler.
    optim = build_optimizer(cfg, dist_model)
    scheduler = build_scheduler(cfg)

    # Data indices file.
    indices_file: Optional[TextIO] = None
    if cfg.save_data_indices:
        indices_file_path = Path(cfg.save_folder) / f"data-indices/rank{get_global_rank()}.tsv.gz"
        if indices_file_path.exists() and not cfg.save_overwrite:
            # raise OLMoConfigurationError(f"{indices_file_path} already exists, use --save_overwrite to overwrite")
            # TODO(mm): we skip this for now
            pass
        indices_file_path.parent.mkdir(exist_ok=True, parents=True)
        indices_file = gzip.open(indices_file_path, "wt")

    # Consolidate components into `Trainer` object.
    with Trainer(
        cfg=cfg,
        epoch=cfg.epoch,
        model=olmo_model,
        dist_model=dist_model,
        optim=optim,
        scheduler=scheduler,
        train_loader=train_loader,
        device=device,
        evaluators=evaluators,
        indices_file=indices_file,
    ) as trainer:
        if cfg.try_load_latest_save:
            # find the last checkpoint in the parent_save_folder
            if cfg.parent_save_folder is not None:
                # log.info("Searching for checkpoints in %s", cfg.parent_save_folder)

                # # find all subdirectories in the parent_save_folder
                # all_subdirs = [
                #     os.path.join(cfg.parent_save_folder, d)
                #     for d in os.listdir(cfg.parent_save_folder)
                #     if os.path.isdir(os.path.join(cfg.parent_save_folder, d))
                # ]
                # if len(all_subdirs) > 0:
                #     # latest_subdir = max(all_subdirs, key=os.path.getmtime) # get the subdir that was modified last
                #     all_subdirs.sort(key=lambda x: os.path.getmtime(x))
                #     latest_subdir = Path(all_subdirs[-2])

                #     log.info(f"Latest checkpoint dir: {latest_subdir}")

                #     # load check  point
                #     if (checkpoint_dir := find_latest_checkpoint(latest_subdir)) is not None:
                #         log.info("Setting load path to local checkpoint %s", checkpoint_dir)
                #         cfg.load_path = str(checkpoint_dir)
                #     else:
                #         log.warning("No checkpoint found to load. Starting from cfg.load_path instead.")
                log.warning(
                    "try_load_latest_save is deprecated - checkpoints are now automatically loaded from existing output folders"
                )

        # start from pre-trained checkpoint
        # restoring the optimizer state, scheduler state, and trainer state happens inside the restore_checkpoint method
        # to be precise, in the load_trainer_state_dict method
        if not cfg.dry_run and not cfg.no_pre_train_checkpoint and cfg.load_path is None:
            if cfg.distributed_strategy == DistributedStrategy.ddp:
                checkpoint_type = CheckpointType.unsharded

                if cfg.save_interval_unsharded is None:
                    log.warning(
                        "DDP requires setting `save_interval_unsharded`. Using the value set for `save_interval`."
                    )
                    cfg.save_interval_unsharded = cfg.save_interval

                if cfg.save_num_unsharded_checkpoints_to_keep == 0:
                    log.warning(
                        "DDP requires setting `save_num_unsharded_checkpoints_to_keep`. Using the value set for `save_num_checkpoints_to_keep`."
                    )
                    cfg.save_num_unsharded_checkpoints_to_keep = cfg.save_num_checkpoints_to_keep
            elif cfg.distributed_strategy == DistributedStrategy.fsdp:
                checkpoint_type = (
                    CheckpointType.sharded if cfg.save_num_checkpoints_to_keep != 0 else CheckpointType.unsharded
                )
            else:
                raise NotImplementedError(f"Distributed strategy {cfg.distributed_strategy} not supported yet!")

            # We save a checkpoint up-front to make sure this won't fail (due to disk space or whatever).
            log.info("Saving pre-train checkpoint...")
            checkpoint_path, local_checkpoint_cache = trainer.save_checkpoint(checkpoint_type=checkpoint_type)
            log.info(f"Checkpoint saved to {checkpoint_path}")

            # And they we verify that we can load it.
            log.info("Attempting to load pre-train checkpoint...")
            trainer.restore_checkpoint(
                checkpoint_path, checkpoint_type=checkpoint_type, local_cache=local_checkpoint_cache
            )
            log.info("Checkpoint successfully loaded")

            # NOTE: https://github.com/allenai/LLM/issues/233
            #  log.info("Removing pre-train checkpoint...")
            #  trainer.remove_checkpoint(checkpoint_type=checkpoint_type)
            #  log.info("Successfully removed checkpoint")

        # start from load_path
        if cfg.load_path is not None:
            log.info(f"Loading checkpoint from {cfg.load_path}...")
            trainer.restore_checkpoint(
                cfg.load_path,
                load_optimizer_state=not cfg.reset_optimizer_state,
                load_trainer_state=not cfg.reset_trainer_state,
                sharded_checkpointer=cfg.load_path_sharded_checkpointer,
            )
            log.info("Checkpoint successfully loaded")

            # If we have to, set a new scheduler:
            if cfg.reset_optimizer_state and not cfg.reset_trainer_state:
                trainer.scheduler = BoltOnWarmupScheduler.wrap(
                    trainer.scheduler,
                    trainer.global_step,
                    int(trainer.global_step + cfg.scheduler.t_warmup),
                )

        if cfg.force_save_unsharded and cfg.distributed_strategy != DistributedStrategy.ddp:
            log.info("Saving unsharded checkpoint...")
            checkpoint_path, _ = trainer.save_checkpoint(checkpoint_type=CheckpointType.unsharded)
            log.info(f"Unsharded checkpoint saved to {checkpoint_path}")

        if cfg.compile is not None:
            # TODO (epwalsh): trying to compile the whole train step results in a compile-time error from within
            # the optimizer. We should investigate this further at some point.
            #  trainer.train_step = torch.compile(trainer.train_step, **cfg.compile.asdict())
            trainer.train_batch = torch.compile(trainer.train_batch, **cfg.compile.asdict())  # type: ignore
            # TODO (epwalsh): compiling the `eval_batch()` method is a little sketchy since the inputs will look
            # different for different eval tasks. That might be okay, but it might not be.
            #  trainer.eval_batch = torch.compile(trainer.eval_batch, **cfg.compile.asdict())  # type: ignore
            # Alternatively, could just do this:
            #  trainer.fsdp_model = torch.compile(trainer.fsdp_model, **cfg.compile.asdict())

        if not cfg.dry_run:
            log.info("Starting training...")
            trainer.fit()
            log.info("Training complete")
        else:
            log.info("Dry run complete")


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError as e:
        print(f"failed to set multiprocessing start method: {e}")
    log.info(f"Multiprocessing start method set to '{mp.get_start_method()}'")

    # Set CUDA device.
    torch.cuda.set_device(f"cuda:{get_local_rank()}")

    # Initialize process group.
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
    log.info("Process group initialized")

    prepare_cli_environment()
    log.info("CLI environment prepared")

    add_cached_path_clients()

    try:
        yaml_path, args_list = sys.argv[1], sys.argv[2:]
    except IndexError:
        raise OLMoCliError(f"Usage: {sys.argv[0]} [CONFIG_PATH] [OPTIONS]")

    cfg = TrainConfig.load(yaml_path, [clean_opt(s) for s in args_list])
    main(cfg)
