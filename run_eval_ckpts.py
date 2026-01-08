#!/usr/bin/env python3
"""
Run scripts/eval.py via torchrun for all matching checkpoints of a given family.

Checkpoint name format:
  "{family}_steps-<steps>_cosine-<cosine>_seed-<seed>-<uid>"

Example:
  "20M_steps-1912_cosine-05_seed-479-254a0e18a128e5e4"

Usage examples:
  python run_eval_ckpts.py 20M
  python run_eval_ckpts.py 20M --seed 479
  python run_eval_ckpts.py 20M --cosine 05 --steps 1912
  python run_eval_ckpts.py 50M --dry_run
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


# ---- Edit this mapping as needed ----
FAMILY_TO_EVAL_MICROBATCH = {
    "20M": 48,
    "150M": 24,
}


CKPT_RE = re.compile(
    r"^(?P<family>[^_]+)_steps-(?P<steps>\d+)_cosine-(?P<cosine>[^_]+)_seed-(?P<seed>\d+)-(?P<uid>.+)$"
)


@dataclass(frozen=True)
class CkptInfo:
    name: str
    family: str
    steps: int
    cosine: str
    seed: int
    uid: str
    load_path: Path


def detect_num_gpus() -> int:
    """
    Determine visible GPU count.
    Priority:
      1) CUDA_VISIBLE_DEVICES (common on clusters)
      2) torch.cuda.device_count() if torch is importable
      3) nvidia-smi count fallback
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None:
        cvd = cvd.strip()
        if cvd == "":
            return 0
        # Common patterns: "0,1,2,3" or "0" or "GPU-<uuid>,GPU-<uuid>"
        parts = [p.strip() for p in cvd.split(",") if p.strip() != ""]
        return len(parts)

    try:
        import torch  # type: ignore

        return int(torch.cuda.device_count())
    except Exception:
        pass

    # Fallback: nvidia-smi
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True)
        lines = [ln for ln in out.splitlines() if ln.strip().startswith("GPU ")]
        return len(lines)
    except Exception:
        return 0


def iter_matching_ckpts(
    ckpt_root: Path,
    family: str,
    steps: Optional[int],
    cosine: Optional[str],
    seed: Optional[int],
) -> List[CkptInfo]:
    """
    Scan ckpt_root for dirs matching pattern and filters, and return sorted list.
    Sorting: (steps asc, seed asc, name asc)
    """
    if not ckpt_root.exists():
        raise FileNotFoundError(f"Checkpoint root does not exist: {ckpt_root}")

    matches: List[CkptInfo] = []
    for p in ckpt_root.iterdir():
        if not p.is_dir():
            continue

        m = CKPT_RE.match(p.name)
        if not m:
            continue

        fam = m.group("family")
        if fam != family:
            continue

        st = int(m.group("steps"))
        co = m.group("cosine")
        sd = int(m.group("seed"))
        uid = m.group("uid")

        if steps is not None and st != steps:
            continue
        if cosine is not None and co != cosine:
            continue
        if seed is not None and sd != seed:
            continue

        load_path = p / "latest-unsharded"
        if not load_path.exists():
            # Still record it, but you'll likely want to skip at runtime
            # We keep it and warn later so you can see which ckpts are missing.
            pass

        matches.append(
            CkptInfo(
                name=p.name,
                family=fam,
                steps=st,
                cosine=co,
                seed=sd,
                uid=uid,
                load_path=load_path,
            )
        )

    matches.sort(key=lambda x: (x.steps, x.seed, x.name))
    return matches


def build_command(
    num_gpus: int,
    config_path: Path,
    load_path: Path,
    microbatch: int,
    extra_eval_args: List[str],
) -> List[str]:
    """
    Return argv list for subprocess.run (no shell needed).
    Equivalent to:
      torchrun --nproc_per_node=<num_gpus> scripts/eval.py <config_path>
        --load_path=<load_path> --device_eval_microbatch_size=<microbatch> ...
    """
    cmd = [
        "torchrun",
        f"--nproc_per_node={num_gpus}",
        "scripts/eval.py",
        str(config_path),
        f"--load_path={str(load_path)}",
        f"--device_eval_microbatch_size={microbatch}",
    ]
    cmd.extend(extra_eval_args)
    return cmd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("family", help='Model family, e.g. "20M", "50M"')
    ap.add_argument("--seed", type=int, default=None, help="Filter checkpoints by seed")
    ap.add_argument("--steps", type=int, default=None, help="Filter checkpoints by steps")
    ap.add_argument(
        "--cosine",
        type=str,
        default=None,
        help='Filter checkpoints by cosine tag EXACTLY as in name (e.g. "05")',
    )

    ap.add_argument(
        "--ckpt_root",
        type=Path,
        default=Path("/work/scratch/olmo/checkpoints"),
        help="Root directory containing checkpoint directories",
    )
    ap.add_argument(
        "--config_template",
        type=str,
        default="configs/dolma_c4/evaluate/OLMo-{family}.yaml",
        help="Config path template; uses {family}",
    )

    ap.add_argument(
        "--microbatch",
        type=int,
        default=None,
        help="Override eval microbatch size (otherwise uses built-in family mapping)",
    )

    ap.add_argument(
        "--skip_missing_load_path",
        action="store_true",
        help="Skip checkpoints where latest-unsharded does not exist",
    )
    ap.add_argument("--dry_run", action="store_true", help="Print commands but do not run")
    ap.add_argument(
        "--extra_eval_args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra args passed to scripts/eval.py (prefix with --, after a '--' separator)",
    )

    args = ap.parse_args()

    num_gpus = detect_num_gpus()
    if num_gpus <= 0:
        print("ERROR: Could not detect any available GPUs (num_gpus <= 0).", file=sys.stderr)
        return 2

    family = args.family
    microbatch = args.microbatch
    if microbatch is None:
        if family not in FAMILY_TO_EVAL_MICROBATCH:
            print(
                f"ERROR: No default microbatch size for family={family}. "
                f"Edit FAMILY_TO_EVAL_MICROBATCH or pass --microbatch.",
                file=sys.stderr,
            )
            return 2
        microbatch = FAMILY_TO_EVAL_MICROBATCH[family]

    config_path = Path(args.config_template.format(family=family))
    if not config_path.exists():
        # You might want to allow non-existent configs, but usually this is a mistake.
        print(f"ERROR: Config file not found: {config_path}", file=sys.stderr)
        return 2

    ckpts = iter_matching_ckpts(
        ckpt_root=args.ckpt_root,
        family=family,
        steps=args.steps,
        cosine=args.cosine,
        seed=args.seed,
    )
    if not ckpts:
        print("No matching checkpoints found.")
        return 0

    print(f"Detected GPUs: {num_gpus}")
    print(f"Family: {family} | eval microbatch: {microbatch}")
    print(f"Found {len(ckpts)} matching checkpoints under {args.ckpt_root}.\n")

    failures: List[Tuple[str, int]] = []

    for i, ck in enumerate(ckpts, start=1):
        if args.skip_missing_load_path and not ck.load_path.exists():
            print(f"[{i}/{len(ckpts)}] SKIP (missing load path): {ck.name}")
            continue

        cmd = build_command(
            num_gpus=num_gpus,
            config_path=config_path,
            load_path=ck.load_path,
            microbatch=microbatch,
            extra_eval_args=args.extra_eval_args,
        )

        pretty = " ".join(cmd)
        print(f"[{i}/{len(ckpts)}] {ck.name}")
        print(f"  load_path: {ck.load_path}")
        print(f"  cmd: {pretty}")

        if args.dry_run:
            print("  (dry_run)\n")
            continue

        try:
            subprocess.run(cmd, check=True)
            print("  ✓ done\n")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ FAILED (exit={e.returncode})\n", file=sys.stderr)
            failures.append((ck.name, int(e.returncode)))

    if failures:
        print("Some runs failed:", file=sys.stderr)
        for name, rc in failures:
            print(f"  - {name}: exit {rc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
