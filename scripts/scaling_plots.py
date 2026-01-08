import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import curve_fit

COLOR_POINTS = "#0072B2"
COLOR_FIT = "#D55E00"
FIGSIZE = (8, 4.5)


def power_law(x, a, b, c):
    return a / (x**b) + c

def fit_power_law(x, y):
    try:
        c0 = max(0.0, float(np.min(y) - 0.2))
        b0 = 0.3
        a0 = float((y[0] - c0) * (x[0] ** b0))
        c_upper = float(np.min(y) * 0.999999)
        bounds_lower = (1e-12, 1e-12, 0.0)
        bounds_upper = (np.inf, np.inf, c_upper)
        popt, _ = curve_fit(
            power_law,
            x,
            y,
            p0=(a0, b0, c0),
            bounds=(bounds_lower, bounds_upper),
            maxfev=200000,
        )
        return popt
    except (RuntimeError, ValueError):
        return None

batch_sizes_map = {
    "20M": 64,
    "150M": 192,
}

offsets_map = {
    "20M": [4, 3],
    "150M": [5, 5],
}

sequence_length = 2024

def save_plot(path, title, xlabel, ylabel, xscale=None, yscale=None):
    if xscale:
        plt.xscale(xscale)
    if yscale:
        plt.yscale(yscale)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="both", linestyle=":")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {path}")

def plot_last_token_data(family="20M", seed=479, cosine="05", num_coverage_points=8, num_random_sets=50):
    # Create directory for saving plots
    save_dir = f"plots_last_token_data_scaling_{family}"
    os.makedirs(save_dir, exist_ok=True)

    batch_size = batch_sizes_map[family]

    # File setup
    checkpoints_root = "/work/scratch/olmo/checkpoints"
    file_name = "ppl-validation"

    # discover all checkpoint dirs for this (family, seed, cosine) and group by step
    pattern = os.path.join(
        checkpoints_root,
        f"{family}_steps-*"
        f"_cosine-{cosine}"
        f"_seed-{seed}-*"
    )
    matching_ckpt_dirs = [d for d in glob.glob(pattern) if os.path.isdir(d)]

    step_to_dirs = {}
    # Example: 20M_steps-6453_cosine-05_seed-479-d2ee707418105283
    step_re = re.compile(
        rf"^{re.escape(family)}_steps-(\d+)_cosine-{re.escape(str(cosine))}_seed-{re.escape(str(seed))}-"
    )

    for ckpt_dir in matching_ckpt_dirs:
        base = os.path.basename(ckpt_dir.rstrip("/"))
        m = step_re.match(base)
        if not m:
            continue
        step = int(m.group(1))
        step_to_dirs.setdefault(step, []).append(ckpt_dir)

    steps = sorted(step_to_dirs.keys())
    if not steps:
        print(f"Warning: No checkpoints found for family={family}, seed={seed}, cosine={cosine}")
        return

    start_offset = offsets_map[family][0]
    end_offset = offsets_map[family][1]
    steps_range = steps[start_offset:len(steps)-end_offset]
    step_stats = []

    for step in steps_range:
        matching_dirs = step_to_dirs[step]

        token_losses_list = []

        for ckpt_dir in matching_dirs:
            # Construct the path to the "step{step+10}-unsharded" directory
            next_step_dir = os.path.join(ckpt_dir, f"step{step + 10}-unsharded")

            if not os.path.isdir(next_step_dir):
                print(f"Latest saved checkpoint not found: {next_step_dir}")
                continue

            checkpoint_dir = os.path.join(ckpt_dir, "latest-unsharded")

            # Load losses
            try:
                losses = torch.load(f"{checkpoint_dir}/{file_name}_losses.pt")  # N x (T-1)
            except Exception:
                print(f"No eval loss calculated for step {step}")
                continue

            # Get token losses (last column)
            token_losses = losses[:, -1].numpy()
            token_losses_list.append(token_losses)

        if not token_losses_list:
            continue

        token_losses_all = np.concatenate(token_losses_list, axis=0)
        num_tokens = batch_size * sequence_length * step
        avg_loss = float(np.mean(token_losses_all))

        step_stats.append(
            {
                "step": step,
                "num_tokens": num_tokens,
                "token_losses": token_losses_all,
                "avg_loss": avg_loss,
            }
        )

    if not step_stats:
        print("No usable evaluation losses found for the selected steps.")
        return

    # Plot 1: average loss vs tokens with power law fit (log-log).
    tokens = np.array([s["num_tokens"] for s in step_stats], dtype=np.float64)
    avg_losses = np.array([s["avg_loss"] for s in step_stats], dtype=np.float64)

    popt = fit_power_law(tokens, avg_losses)
    if popt is None:
        print("Power law fit failed for average losses.")
        return

    x_fit = np.linspace(tokens.min(), tokens.max(), 200)
    y_fit = power_law(x_fit, *popt)

    plt.figure(figsize=FIGSIZE)
    plt.scatter(tokens, avg_losses, color=COLOR_POINTS, marker="x", label="Avg Loss")
    plt.plot(x_fit, y_fit, color=COLOR_FIT, linestyle="--", label="Power Law Fit")
    plt.text(
        0.02,
        0.05,
        f"a={popt[0]:.3g}\nb={popt[1]:.3g}\nc={popt[2]:.3g}",
        transform=plt.gca().transAxes,
        fontsize=9,
        verticalalignment="bottom",
    )
    plt.legend()
    plot_path = os.path.join(save_dir, "average_loss_power_law.png")
    save_plot(
        plot_path,
        "Average Loss Scaling (Power Law Fit)",
        "Tokens",
        "Average Token-level CE Loss",
        xscale="log",
        yscale="log",
    )
    print("Tokens vs avg loss values:")
    for num_tokens, avg_loss in zip(tokens, avg_losses):
        print(f"tokens={int(num_tokens)} avg_loss={avg_loss:.6f}")

    # Plot 2: power law fit error vs number of tokens used in the average loss.
    rng = np.random.default_rng(479)
    val_count = min(len(s["token_losses"]) for s in step_stats)
    num_points = num_coverage_points
    k_values = np.unique(
        np.clip(
            np.logspace(0, np.log10(val_count), num=num_points).astype(int),
            1,
            val_count,
        )
    )
    if k_values[-1] != val_count:
        k_values = np.append(k_values, val_count)

    m = num_random_sets
    fit_errors = []
    fit_r2 = []
    fit_errors_full = []

    for k in k_values:
        trial_errors = []
        trial_r2 = []
        trial_errors_full = []
        for _ in range(m):
            sampled_avg_losses = []
            for s in step_stats:
                sample = rng.choice(s["token_losses"], size=k, replace=False)
                sampled_avg_losses.append(np.mean(sample))

            sampled_avg_losses = np.array(sampled_avg_losses, dtype=np.float64)
            popt_k = fit_power_law(tokens, sampled_avg_losses)
            if popt_k is None:
                continue
            preds = power_law(tokens, *popt_k)
            log_err = np.mean((np.log10(sampled_avg_losses) - np.log10(preds)) ** 2)
            trial_errors.append(log_err)
            log_y = np.log10(sampled_avg_losses)
            log_pred = np.log10(preds)
            ss_res = np.sum((log_y - log_pred) ** 2)
            ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
            trial_r2.append(1.0 - (ss_res / ss_tot if ss_tot > 0 else 0.0))

            preds_full = power_law(tokens, *popt)
            log_err_full = np.mean((np.log10(sampled_avg_losses) - np.log10(preds_full)) ** 2)
            trial_errors_full.append(log_err_full)

        if trial_errors:
            fit_errors.append(np.mean(trial_errors))
        else:
            fit_errors.append(np.nan)
        if trial_r2:
            fit_r2.append(np.mean(trial_r2))
        else:
            fit_r2.append(np.nan)
        if trial_errors_full:
            fit_errors_full.append(np.mean(trial_errors_full))
        else:
            fit_errors_full.append(np.nan)

    percent_vals = (k_values / val_count) * 100.0
    plt.figure(figsize=FIGSIZE)
    ax = plt.gca()
    ax.plot(percent_vals, fit_errors, marker="o", color=COLOR_POINTS, label="MSLE (subset fit)")
    ax.set_xscale("log")
    ax.set_xlabel("Percent of Validation Set Used")
    ax.set_ylabel("Mean Squared Log Error (Power Law Fit)")
    ax.grid(True, which="both", linestyle=":")
    ax2 = ax.twinx()
    ax2.plot(percent_vals, fit_r2, marker="s", color=COLOR_FIT, label="R2 (log space)")
    ax2.set_ylabel("R2 (log space)")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="center right")
    plot_path = os.path.join(save_dir, "power_law_fit_error_vs_coverage.png")
    plt.title("Power Law Fit Error vs Validation Set Coverage")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {plot_path}")

    plt.figure(figsize=FIGSIZE)
    plt.plot(percent_vals, fit_errors_full, marker="o", color=COLOR_POINTS)
    plot_path = os.path.join(save_dir, "power_law_full_fit_error_vs_coverage.png")
    save_plot(
        plot_path,
        "Error wrt Full-Set Power Law Fit vs Coverage",
        "Percent of Validation Set Used",
        "Mean Squared Log Error (Full-Set Power Law)",
        xscale="log",
    )

    # Plot 3: per-k loss vs steps with power law fit.
    per_k_dir = os.path.join(save_dir, "subset_power_law_fits")
    os.makedirs(per_k_dir, exist_ok=True)

    steps_array = np.array([s["step"] for s in step_stats], dtype=np.float64)
    tokens_for_steps = batch_size * sequence_length * steps_array

    for k in k_values:
        sampled_avg_losses = []
        for s in step_stats:
            sample = rng.choice(s["token_losses"], size=k, replace=False)
            sampled_avg_losses.append(np.mean(sample))

        sampled_avg_losses = np.array(sampled_avg_losses, dtype=np.float64)
        popt_k = fit_power_law(tokens_for_steps, sampled_avg_losses)
        if popt_k is None:
            print(f"Power law fit failed for k={k}.")
            continue

        x_fit_steps = np.linspace(steps_array.min(), steps_array.max(), 200)
        x_fit_tokens = batch_size * sequence_length * x_fit_steps
        y_fit = power_law(x_fit_tokens, *popt_k)

        plt.figure(figsize=FIGSIZE)
        plt.scatter(tokens_for_steps, sampled_avg_losses, color=COLOR_POINTS, marker="x", label="Avg Loss")
        plt.plot(x_fit_tokens, y_fit, color=COLOR_FIT, linestyle="--", label="Power Law Fit")
        plt.text(
            0.02,
            0.05,
            f"a={popt_k[0]:.3g}\nb={popt_k[1]:.3g}\nc={popt_k[2]:.3g}",
            transform=plt.gca().transAxes,
            fontsize=9,
            verticalalignment="bottom",
        )
        plt.legend()
        plot_path = os.path.join(per_k_dir, f"avg_loss_steps_k_{k}.png")
        save_plot(
            plot_path,
            f"Avg Loss vs Steps (k={k}, {k/val_count:.1%} of val set)",
            "Tokens",
            "Average Token-level CE Loss",
            xscale="log",
            yscale="log",
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot scaling laws for OLMo checkpoints.")
    parser.add_argument(
        "--family",
        default="150M",
        choices=sorted(batch_sizes_map.keys()),
        help="Model family to plot.",
    )
    parser.add_argument("--seed", type=int, default=479, help="Training seed.")
    parser.add_argument("--cosine", default="05", help="Cosine schedule identifier.")
    parser.add_argument(
        "--coverage-points",
        type=int,
        default=12,
        help="Number of coverage points to evaluate in the fit error plot.",
    )
    parser.add_argument(
        "--num-random-sets",
        type=int,
        default=200,
        help="Number of random k-token sets to average per coverage point.",
    )
    args = parser.parse_args()
    plot_last_token_data(
        family=args.family,
        seed=args.seed,
        cosine=args.cosine,
        num_coverage_points=args.coverage_points,
        num_random_sets=args.num_random_sets,
    )
