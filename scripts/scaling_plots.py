import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import random
import json
import pdb

from collections import defaultdict, Counter
from matplotlib.cm import get_cmap

import seaborn as sns

from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

import pandas as pd

from olmo.tokenizer import Tokenizer

cmap = plt.cm.get_cmap("tab20", 100) 

# Model sizes in parameters (approximate)
size_to_params = {
    "20M": 20e6,
    "59M": 59e6,
    "136M": 136e6,
    "267M": 267e6
}

# Seeds
seeds = [132, 479, 865]

# steps = [1000, 1500, 2000, 2500, 3000, 3500, 4000]
# steps = [x for x in range(10, 510, 10)] + [x for x in range(550, 5050, 50)]
steps = [x for x in range(200, 510, 10)] + [x for x in range(550, 2050, 50)]


def ids_to_tokens(ids, tokenizer):
    tokens = []
    for idx in ids:
        tokens.append(tokenizer.base_tokenizer.id_to_token(idx))
    return tokens


tokenizer = Tokenizer.from_file(
    "/home/mila/a/arkil.patel/OLMo/olmo_data/tokenizers/allenai_dolma2.json",
    eos_token_id=100257,
    pad_token_id=100277,
)


def get_data_freq(token_id):
    path = '/network/scratch/m/marius.mosbach/olmo/training_data/olmo2/slim_pajama/dolma/part-0-00000.npy'
    size = os.path.getsize(path)
    data = np.memmap(path, dtype='uint32', mode='r', shape=(size // 4,))

    num_instances = 0

    num_instances += (data==token_id).sum()

    path = '/network/scratch/m/marius.mosbach/olmo/training_data/olmo2/slim_pajama/dolma/part-1-00000.npy'
    size = os.path.getsize(path)
    data = np.memmap(path, dtype='uint32', mode='r', shape=(size // 4,))

    num_instances += (data==token_id).sum()

    path = '/network/scratch/m/marius.mosbach/olmo/training_data/olmo2/slim_pajama/dolma/part-1-00000.npy'
    size = os.path.getsize(path)
    data = np.memmap(path, dtype='uint32', mode='r', shape=(size // 4,))

    num_instances += (data==token_id).sum()

    path = '/network/scratch/m/marius.mosbach/olmo/training_data/olmo2/slim_pajama/dolma/part-2-00000.npy'
    size = os.path.getsize(path)
    data = np.memmap(path, dtype='uint32', mode='r', shape=(size // 4,))

    num_instances += (data==token_id).sum()

    path = '/network/scratch/m/marius.mosbach/olmo/training_data/olmo2/slim_pajama/dolma/part-3-00000.npy'
    size = os.path.getsize(path)
    data = np.memmap(path, dtype='uint32', mode='r', shape=(size // 4,))

    num_instances += (data==token_id).sum()

    path = '/network/scratch/m/marius.mosbach/olmo/training_data/olmo2/slim_pajama/dolma/part-4-00000.npy'
    size = os.path.getsize(path)
    data = np.memmap(path, dtype='uint32', mode='r', shape=(size // 4,))

    num_instances += (data==token_id).sum()

    path = '/network/scratch/m/marius.mosbach/olmo/training_data/olmo2/slim_pajama/dolma/part-5-00000.npy'
    size = os.path.getsize(path)
    data = np.memmap(path, dtype='uint32', mode='r', shape=(size // 4,))

    num_instances += (data==token_id).sum()

    path = '/network/scratch/m/marius.mosbach/olmo/training_data/olmo2/slim_pajama/dolma/part-6-00000.npy'
    size = os.path.getsize(path)
    data = np.memmap(path, dtype='uint32', mode='r', shape=(size // 4,))

    num_instances += (data==token_id).sum()

    path = '/network/scratch/m/marius.mosbach/olmo/training_data/olmo2/slim_pajama/dolma/part-7-00000.npy'
    size = os.path.getsize(path)
    data = np.memmap(path, dtype='uint32', mode='r', shape=(size // 4,))

    num_instances += (data==token_id).sum()

    return num_instances



# Define compute estimation function
def compute_log_flops(size, tokens):
    if tokens == 0:
        tokens = 1
    params = size_to_params[size]
    return np.log10(6 * params * tokens)


def moving_average(data, window_size=3):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')



def exp_decay(x, a, b, c):
    return a * np.exp(-b * (x-17)) + c

def power_law(x, a, b, c):
    return a / (x ** b) + c

def log_linear(x, a, b):
    return -a * x + b


def fit_models_and_classify_2(x, y):
    x = np.array(x)
    y = np.array(y)
    best_r2 = -np.inf
    best_fit = None
    best_label = "fit_failed"
    best_model_name = None

    models = [
        ("exp_decay", exp_decay, [8.0, 2.0, 0.5]),
        # ("power_law", power_law, [1, 1, 1]),
        # ("log_linear", log_linear, [1, 1])
    ]

    for name, func, p0 in models:
        try:
            popt, _ = curve_fit(func, x, y, p0=p0, maxfev=10000)
            fitted = func(x, *popt)
            r2 = r2_score(y, fitted)
            norm_std = np.std(y - fitted) / np.mean(y)

            # Create high-res x range for smoother derivative estimation
            pl_x = np.linspace(min(x), max(x), 100)
            pl_y = func(pl_x, *popt)

            # Estimate slope
            slopes = np.gradient(pl_y, pl_x)
            decreasing_fraction = np.sum(slopes < 0) / len(slopes)

            # Update best if this model is better
            if r2 > best_r2:
                best_r2 = r2
                best_fit = (pl_x, pl_y)
                best_model_name = name

                # Classification based on slope + fit quality
                if decreasing_fraction > 0.9:  # Mostly decreasing
                    if r2 > 0.8 and norm_std < 0.5:
                        best_label = "smooth_decreasing"
                    else:
                        best_label = "oscillating_decreasing"
                else:
                    best_label = "noisy"
        except:
            continue

    return best_label, best_model_name, best_r2, best_fit


def fit_models_and_classify(x, y):
    x = np.array(x)
    y = np.array(y)
    best_r2 = -np.inf
    best_fit = None
    best_label = "fit_failed"
    best_model_name = None

    models = [
        ("exp_decay", exp_decay, [8.0, 2.0, 0.5]),
        # ("power_law", power_law, [1, 1, 1]),
        # ("log_linear", log_linear, [1, 1])
    ]

    for name, func, p0 in models:
        try:
            popt, _ = curve_fit(func, x, y, p0=p0, maxfev=10000)
            fitted = func(x, *popt)
            r2 = r2_score(y, fitted)
            norm_std = np.std(y - fitted) / np.mean(y)

            pl_x = np.linspace(min(x), max(x), 100)
            pl_y = func(pl_x, *popt)

            if r2 > best_r2:
                best_r2 = r2
                best_fit = (pl_x, pl_y)
                best_model_name = name
                if r2 > 0.8 and norm_std < 0.5:
                    best_label = "smooth_decreasing"
                elif r2 > 0.4:
                    best_label = "oscillating_decreasing"
                else:
                    best_label = "noisy"
        except:
            continue

    return best_label, best_model_name, best_r2, best_fit


def plot_last_token_data_functional():
    # Create directory for saving plots
    save_dir = "plots_last_token_data_scaling_functional"
    os.makedirs(save_dir, exist_ok=True)

    # File setup
    checkpoint_template = "/network/scratch/a/arkil.patel/olmo/checkpoints/OLMo_136M_456_indi_{step}-*"
    file_name = "ppl-validation"

    plt.figure(figsize=(8, 6))

    all_example_losses = {}  # Store losses for each example across sizes

    all_avg_logs = []
    all_median_logs = []

    for step in steps:
        checkpoint_pattern = checkpoint_template.format(step=step)
        matching_dirs = glob.glob(checkpoint_pattern)

        if not matching_dirs:
            print(f"Warning: No checkpoint found for step {step}")
            continue  # Skip this size if no matching directory is found

        checkpoint_dir = matching_dirs[0]  # Use the first match
        checkpoint_dir = checkpoint_dir + "/latest-unsharded"
        # Load losses
        try:
            losses = torch.load(f"{checkpoint_dir}/{file_name}_losses.pt")  # N x (T-1)
            labels = torch.load(f"{checkpoint_dir}/{file_name}_labels.pt")  # N x (T-1)
        except:
            print(f"No eval loss calculated for step {step}")
            continue

        # max_duration = 1.25*step
        # tokens_covered = max_duration*1024*1024
        # # Compute log compute
        # log_compute = np.log10(6 * 190335744 * tokens_covered)  # Adding 1 to match full sequence length

        log_compute = np.log10(802698.2491095e9 * step)  # change this number based on wandb training gflops

        # Get token losses (last column)
        token_losses = losses[:, -1].numpy()

        token_losses_2 = losses[:, -2].numpy()
        token_losses_3 = losses[:, -3].numpy()
        token_losses_4 = losses[:, -4].numpy()

        # Store losses per example
        for i, loss in enumerate(token_losses):
            if i not in all_example_losses:
                all_example_losses[i] = {"log_compute": [], "losses": [], "losses2": [], "losses3": [], "losses4": [], "labels": labels[i]}
            all_example_losses[i]["log_compute"].append(log_compute)
            all_example_losses[i]["losses"].append(loss)
            all_example_losses[i]["losses2"].append(token_losses_2[i])
            all_example_losses[i]["losses3"].append(token_losses_3[i])
            all_example_losses[i]["losses4"].append(token_losses_4[i])

        # Store average loss per size
        median_loss = np.median(token_losses)
        mean_loss = np.mean(token_losses)

        all_avg_logs.append((log_compute, mean_loss))
        all_median_logs.append((log_compute, median_loss))

    # Select up to 50 random examples
    example_indices = list(all_example_losses.keys())

    cluster_colors = {
        "smooth_decreasing": "#1b9e77",   # Teal-green
        "oscillating_decreasing": "#d95f02",  # Orange
        # "oscillating_increasing": "#7570b3",  # Purple-blue
        "noisy": "#e7298a",   # Reddish pink
    }

    # Cluster examples
    clusters = {}

    running_sim_perc_2 = 0
    running_sim_perc_3 = 0
    running_sim_perc_4 = 0
    cnt = 0

    # Classify and plot
    for example_id in example_indices:
        example = all_example_losses[example_id]
        logs = example["log_compute"]
        losses = example["losses"]
        losses2 = example["losses2"]
        losses3 = example["losses3"]
        losses4 = example["losses4"]

        # Sort by log_compute
        sorted_pairs = sorted(zip(logs, losses))
        logs_sorted, losses_sorted = zip(*sorted_pairs)

        sorted_pairs2 = sorted(zip(logs, losses2))
        _, losses_sorted2 = zip(*sorted_pairs2)

        sorted_pairs3 = sorted(zip(logs, losses3))
        _, losses_sorted3 = zip(*sorted_pairs3)

        sorted_pairs4 = sorted(zip(logs, losses4))
        _, losses_sorted4 = zip(*sorted_pairs4)

        # Smooth losses before plotting
        smoothed_losses = moving_average(losses_sorted, window_size=5)
        # Also trim logs to match the shorter length
        smoothed_logs = logs_sorted[:len(smoothed_losses)]

        smoothed_losses2 = moving_average(losses_sorted2, window_size=5)
        smoothed_losses3 = moving_average(losses_sorted3, window_size=5)
        smoothed_losses4 = moving_average(losses_sorted4, window_size=5)

        cluster, model_used, r2, best_fit = fit_models_and_classify_2(smoothed_logs, smoothed_losses)

        cluster2, _, _, _ = fit_models_and_classify_2(smoothed_logs, smoothed_losses2)
        cluster3, _, _, _ = fit_models_and_classify_2(smoothed_logs, smoothed_losses3)
        cluster4, _, _, _ = fit_models_and_classify_2(smoothed_logs, smoothed_losses4)

        cnt += 1

        if cluster2 == cluster:
            running_sim_perc_2 += 1
            print("Running similarity 2 percentage: ", running_sim_perc_2/cnt)
        if cluster3 == cluster:
            running_sim_perc_3 += 1
            print("Running similarity 3 percentage: ", running_sim_perc_3/cnt)
        if cluster4 == cluster:
            running_sim_perc_4 += 1
            print("Running similarity 4 percentage: ", running_sim_perc_4/cnt)

        if cluster not in clusters:
            clusters[cluster] = []
        clusters[cluster].append((logs_sorted, losses_sorted))

        # Cluster percentages:
        for clus in clusters:
            print(f"Cluster {clus} Percentage: {len(clusters[clus])/cnt}")
        
        print("---------------------------------------------------------------------")

        plt.figure(figsize=(6, 4))
        plt.plot(smoothed_logs, smoothed_losses, color=cluster_colors.get(cluster, "gray"))

        if best_fit is not None:
            plt.plot(best_fit[0], best_fit[1], color="black")

        plt.title(f"Example {example_id} | {cluster} | {model_used} | R²={r2:.2f}")
        plt.xlabel("Log Compute")
        plt.ylabel("Token Loss")
        plt.grid(True)

        indiv_dir = os.path.join(save_dir, "individual_examples", cluster)
        os.makedirs(indiv_dir, exist_ok=True)
        plt.savefig(os.path.join(indiv_dir, f"example_{example_id}.png"), dpi=200, bbox_inches="tight")
        plt.close()


    cluster_plot_dir = os.path.join(save_dir, "cluster_plots")
    os.makedirs(cluster_plot_dir, exist_ok=True)

    n_per_class = 100  # Adjust based on density

    for cluster, examples in clusters.items():
        plt.figure(figsize=(8, 6))
        selected = random.sample(examples, min(n_per_class, len(examples)))
        for logs, losses in selected:
            smoothed_losses = moving_average(losses, window_size=5)
            smoothed_logs = logs[:len(smoothed_losses)]
            plt.plot(smoothed_logs, smoothed_losses, color=cluster_colors.get(cluster, "gray"), alpha=0.6)

        plt.title(f"{cluster} examples")
        plt.xlabel("Log Compute")
        plt.ylabel("Token Loss")
        plt.grid(True)
        plt.savefig(os.path.join(cluster_plot_dir, f"{cluster}.png"), dpi=300, bbox_inches="tight")
        plt.close()

    
    plt.figure(figsize=(10, 8))
    combined_examples = []

    # Sample up to N examples total across all clusters
    total_to_sample = 50
    all_entries = [(cluster, ex) for cluster, examples in clusters.items() for ex in examples]
    random.shuffle(all_entries)
    selected_entries = all_entries[:total_to_sample]

    for cluster, (logs, losses) in selected_entries:
        smoothed_losses = moving_average(losses, window_size=5)
        smoothed_logs = logs[:len(smoothed_losses)]
        plt.plot(smoothed_logs, smoothed_losses, color=cluster_colors.get(cluster, "gray"), alpha=0.5)

    # Add legend
    for label, color in cluster_colors.items():
        plt.plot([], [], color=color, label=label)

    plt.title("Random Examples from All Clusters")
    plt.xlabel("Log Compute")
    plt.ylabel("Token Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir, "all_clusters.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # Save example counts
    print("Example counts by trend category:")
    for k in clusters:
        print(f"  {k}: {len(clusters[k])}")




def plot_loss_density(all_example_losses, save_dir):
    # Collect all (log_compute, loss) pairs
    all_log_compute_vals = []
    all_loss_vals = []

    for example in all_example_losses.values():
        all_log_compute_vals.extend(example["log_compute"])
        all_loss_vals.extend(example["losses"])

    # Convert to numpy arrays
    x = np.array(all_log_compute_vals)
    y = np.array(all_loss_vals)

    # Create the figure
    plt.figure(figsize=(10, 6))
    sns.kdeplot(
        x=x, y=y,
        cmap="viridis", fill=True, thresh=0.01, levels=100,
    )
    plt.xlabel("Tokens Covered - Log FLOPS")
    plt.ylabel("Token Loss")
    plt.title("Density of Token Losses vs Log Compute")
    plt.grid(True)

    plot_path = os.path.join(save_dir, "loss_density_plot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved density plot: {plot_path}")


def plot_loss_boxplot(all_example_losses, save_dir):
    # Prepare data in long-form for seaborn
    records = []
    for example in all_example_losses.values():
        for log_compute, loss in zip(example["log_compute"], example["losses"]):
            records.append({
                "log_compute": round(log_compute, 2),  # Round for grouping
                "loss": loss
            })

    df = pd.DataFrame(records)

    plt.figure(figsize=(12, 6))
    sns.boxplot(
        x="log_compute", y="loss", data=df, 
        color="skyblue", fliersize=1, linewidth=1
    )
    plt.xlabel("Tokens Covered - Log FLOPS (Rounded)")
    plt.ylabel("Token Loss")
    plt.title("Token Loss Distribution at Each Compute Level")
    plt.xticks(rotation=45)
    plt.grid(True)

    plot_path = os.path.join(save_dir, "loss_boxplot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved box plot: {plot_path}")


def plot_last_token_data():
    # Create directory for saving plots
    save_dir = "plots_last_token_data_scaling"
    os.makedirs(save_dir, exist_ok=True)

    num_examples_to_plot = 100

    # File setup
    checkpoint_template = "/network/scratch/a/arkil.patel/olmo/checkpoints/OLMo_136M_*_indi_{step}-*"
    file_name = "ppl-validation"

    plt.figure(figsize=(8, 6))

    all_example_losses = {}  # Store losses for each example across sizes

    all_avg_logs = []
    all_median_logs = []

    for step in steps:
        # pdb.set_trace()
        checkpoint_pattern = checkpoint_template.format(step=step)
        matching_dirs = glob.glob(checkpoint_pattern)

        if not matching_dirs:
            print(f"Warning: No checkpoint found for step {step}")
            continue  # Skip this size if no matching directory is found

        seed_token_losses = []
        mean_losses = []
        median_losses = []
        latest_labels = None

        for ckpt_dir in matching_dirs:
            # Construct the path to the "step{step+10}-unsharded" directory
            next_step_dir = os.path.join(ckpt_dir, f"step{step + 10}-unsharded")

            if not os.path.isdir(next_step_dir):
                print(f"Latest saved checkpoint not found: {next_step_dir}")
                continue
            
            checkpoint_dir = ckpt_dir + "/latest-unsharded"

            # Load losses
            try:
                losses = torch.load(f"{checkpoint_dir}/{file_name}_losses.pt")  # N x (T-1)
                labels = torch.load(f"{checkpoint_dir}/{file_name}_labels.pt")  # N x (T-1)
            except:
                print(f"No eval loss calculated for step {step}")
                continue

            # max_duration = 1.25*step
            # tokens_covered = max_duration*1024*1024
            # # Compute log compute
            # log_compute = np.log10(6 * 190335744 * tokens_covered)  # Adding 1 to match full sequence length

            log_compute = np.log10(802698.2491095e9 * step)  # change this number based on wandb training gflops

            mean_losses.append(np.mean(losses.numpy()))
            median_losses.append(np.median(losses.numpy()))

            # Get token losses (last column)
            token_losses = losses[:, -1].numpy()
            seed_token_losses.append(token_losses)
            latest_labels = labels

        avg_seed_token_losses = np.mean(seed_token_losses, axis=0)

        # Store losses per example
        for i, loss in enumerate(avg_seed_token_losses):
            if i not in all_example_losses:
                all_example_losses[i] = {"log_compute": [], "losses": [], "labels": latest_labels[i]}
            all_example_losses[i]["log_compute"].append(log_compute)
            all_example_losses[i]["losses"].append(loss)

        # Store average loss per size
        # all_avg_logs.append((log_compute, np.mean(token_losses)))
        median_loss = np.mean(np.array(median_losses))
        mean_loss = np.mean(np.array(mean_losses))

        all_avg_logs.append((log_compute, mean_loss))
        all_median_logs.append((log_compute, median_loss))
    
    # pdb.set_trace()

    # Select up to 50 random examples
    example_indices = list(all_example_losses.keys())
    random.shuffle(example_indices)
    selected_examples = example_indices[:num_examples_to_plot]

    # Plot selected example loss trajectories
    # ci = 0
    # for example_id in selected_examples:
    #     example = all_example_losses[example_id]
    #     plt.plot(example["log_compute"], example["losses"], color=cmap(ci), alpha=0.7)  # Light cyan lines
    #     ci += 1

    # Plot averaged losses as a thick blue line
    # if all_avg_logs:
    #     log_computes, avg_losses = zip(*sorted(all_avg_logs))  # Ensure sorted order for proper plotting
    #     plt.plot(log_computes, avg_losses, 'b-', linewidth=2, label=f"Avg Loss")
    
    # plt.xlabel("Tokens Covered - Log FLOPS")
    # plt.ylabel("Token Loss")
    # plt.title(f"Token-Level Scaling Laws")
    # plt.ylim(0, 7)  # Limit y-axis between 2 and 5
    # plt.legend()
    # plt.grid(True)

    # Define thresholds for classifying trends
    decrease_threshold = -1
    increase_threshold = 0

    # Define color mapping
    cluster_colors = {
        "decreasing": "#1b9e77",
        "not too decreasing": "#7570b3"
    }

    # Cluster examples
    clusters = {
        "decreasing": [],
        "not too decreasing": []
    }

    grouped_toks = {
        "decreasing": [],
        "not too decreasing": []
    }

    freq_toks = {
        "decreasing": [],
        "not too decreasing": []
    }

    # Classify and plot
    for example_id in selected_examples:
        example = all_example_losses[example_id]
        logs = example["log_compute"]
        losses = example["losses"]

        labs = example["labels"]

        tok_freq = 0
        cur_labs = labs[-50:]
        for ll in cur_labs:
            # freq = get_data_freq(ll)
            freq = 0
            tok_freq += freq


        toks = ids_to_tokens(labs, tokenizer)

        # Sort by log_compute
        sorted_pairs = sorted(zip(logs, losses))
        logs_sorted, losses_sorted = zip(*sorted_pairs)

        # Smooth losses before plotting
        smoothed_losses = moving_average(losses_sorted, window_size=5)
        # Also trim logs to match the shorter length
        smoothed_logs = logs_sorted[:len(smoothed_losses)]


        delta = smoothed_losses[-1] - smoothed_losses[0]

        if delta <= decrease_threshold:
            cluster = "decreasing"
        else:
            cluster = "not too decreasing"

        clusters[cluster].append((logs_sorted, losses_sorted))
        grouped_toks[cluster].append("".join(toks[-50:]).replace("\u0120", " "))

        freq_toks[cluster].append(tok_freq)



        plt.plot(smoothed_logs, smoothed_losses, color=cluster_colors[cluster], alpha=0.6)

        # plt.plot(logs_sorted, losses_sorted, color=cluster_colors[cluster], alpha=0.6)

    # Plot averaged losses as a thick blue line
    if all_avg_logs:
        log_computes, avg_losses = zip(*sorted(all_avg_logs))  # Ensure sorted order for proper plotting
        # plt.plot(log_computes, avg_losses, color="black", ls='--', linewidth=3, label="Avg Loss")
        plt.scatter(log_computes, avg_losses, color="black", marker='x', label="Avg Loss")

        # Fit a linear model: y = m * x + b
        log_avg_losses = np.log(avg_losses)
        coeffs = np.polyfit(log_computes, log_avg_losses, deg=1)
        m, b = coeffs

        # Predict in log space and exponentiate to get back to original space
        x_fit = np.linspace(min(log_computes), max(log_computes), 100)
        y_fit = np.exp(m * x_fit + b)

        plt.plot(x_fit, y_fit, color="red", linestyle='--', label="Best Fit Line (log-scaled)")
    
    if all_median_logs:
        log_computes, median_losses = zip(*sorted(all_median_logs))
        # plt.plot(log_computes, median_losses, color="#e7298a", ls='-.', linewidth=3, label="Median Loss")

    # Add legend manually
    for label, color in cluster_colors.items():
        plt.plot([], [], color=color, label=label.capitalize())
    
    plt.yscale("log")

    plt.xlabel("Tokens Covered - Log FLOPS")
    plt.ylabel("Token Loss")
    plt.title(f"Token-Level Scaling Laws")
    plt.ylim(2e0, 10e0)
    # plt.xlim(18.1, 18.4)
    # plt.ylim(0, 12)
    plt.legend()
    plt.grid(True)

    # Save plot
    plot_path = os.path.join(save_dir, f"scaling.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()  # Close to avoid overlapping figures

    print(f"Saved plot: {plot_path}")

    # save grouped tokens as json
    with open(os.path.join(save_dir, "grouped_tokens.json"), "w") as f:
        json.dump(grouped_toks, f, indent=4)

    with open(os.path.join(save_dir, "freq_tokens.json"), "w") as f:
        json.dump(freq_toks, f, indent=4)
    
    # plot_loss_boxplot(all_example_losses, save_dir)
    
    print("Number of decreasing examples:", len(clusters["decreasing"]))
    print("Number of increasing examples:", len(clusters["not too decreasing"]))

        

def plot_last_token_compute():
    # Create directory for saving plots
    save_dir = "plots_last_token_compute_scaling"
    os.makedirs(save_dir, exist_ok=True)

    num_examples_to_plot = 100

    # File setup
    checkpoint_template = "/network/scratch/a/arkil.patel/olmo/checkpoints/OLMo_{size}_{seed}-*"
    file_name = "ppl-validation"

    # Plot for each seed
    for seed in seeds:
        plt.figure(figsize=(8, 6))

        all_example_losses = {}  # Store losses for each example across sizes
        model_size_to_log_compute = {}

        all_avg_logs = []

        for size in size_to_params.keys():
            # pdb.set_trace()
            checkpoint_pattern = checkpoint_template.format(size=size, seed=seed)
            matching_dirs = glob.glob(checkpoint_pattern)

            if not matching_dirs:
                print(f"Warning: No checkpoint found for size {size}, seed {seed}")
                continue  # Skip this size if no matching directory is found

            checkpoint_dir = matching_dirs[0]  # Use the first match
            checkpoint_dir = checkpoint_dir + "/latest-unsharded"
            # Load losses
            losses = torch.load(f"{checkpoint_dir}/{file_name}_losses.pt")  # N x (T-1)

            # Compute log compute
            log_compute = compute_log_flops(size, 5.4e9)  # Adding 1 to match full sequence length
            model_size_to_log_compute[size] = log_compute  # Store for labeling

            # Get token losses (last column)
            token_losses = losses[:, -1].numpy()

            # Store losses per example
            for i, loss in enumerate(token_losses):
                if i not in all_example_losses:
                    all_example_losses[i] = {"log_compute": [], "losses": []}
                all_example_losses[i]["log_compute"].append(log_compute)
                all_example_losses[i]["losses"].append(loss)

            # Store average loss per size
            all_avg_logs.append((log_compute, np.mean(token_losses)))

        # Select up to 100 random examples
        example_indices = list(all_example_losses.keys())
        random.shuffle(example_indices)
        selected_examples = example_indices[:num_examples_to_plot]

        # Plot selected example loss trajectories
        ci = 0
        for example_id in selected_examples:
            example = all_example_losses[example_id]
            plt.plot(example["log_compute"], example["losses"], color=cmap(ci), alpha=0.7)  # Light cyan lines
            ci += 1

        # Plot averaged losses as a thick blue line
        if all_avg_logs:
            log_computes, avg_losses = zip(*sorted(all_avg_logs))  # Ensure sorted order for proper plotting
            plt.plot(log_computes, avg_losses, 'b-', linewidth=2, label=f"Avg Loss (Seed {seed})")
        
        # Sort model sizes by log compute
        sorted_sizes = sorted(model_size_to_log_compute.keys(), key=lambda s: model_size_to_log_compute[s])
        sorted_log_computes = [model_size_to_log_compute[s] for s in sorted_sizes]

        # Set custom ticks
        plt.xticks(sorted_log_computes, sorted_sizes)
        plt.xlabel("Model Size (Params) - Log FLOPS")

        plt.ylabel("Token Loss")
        plt.title(f"Token-Level Scaling Laws (Seed {seed})")
        plt.ylim(0, 7)  # Limit y-axis between 2 and 5
        plt.legend()
        plt.grid(True)

        # Save plot
        plot_path = os.path.join(save_dir, f"scaling_seed_{seed}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()  # Close to avoid overlapping figures

        print(f"Saved plot: {plot_path}")

def plot_random_token_compute():
    # Create directory for saving plots
    save_dir = "plots_random_token_compute_scaling"
    os.makedirs(save_dir, exist_ok=True)

    num_examples_to_plot = 100

    # File setup
    checkpoint_template = "/network/scratch/a/arkil.patel/olmo/checkpoints/OLMo_{size}_{seed}-*"
    file_name = "ppl-validation"

    random_label_indices = random.sample(range(1023), num_examples_to_plot)
    random_example_indices = None

    # Plot for each seed
    for seed in seeds:
        plt.figure(figsize=(8, 6))

        all_example_losses = {}  # Store losses for each example across sizes
        model_size_to_log_compute = {}
        all_label_losses = {}

        all_avg_logs = []

        for size in size_to_params.keys():
            checkpoint_pattern = checkpoint_template.format(size=size, seed=seed)
            matching_dirs = glob.glob(checkpoint_pattern)

            if not matching_dirs:
                print(f"Warning: No checkpoint found for size {size}, seed {seed}")
                continue  # Skip this size if no matching directory is found

            checkpoint_dir = matching_dirs[0]  # Use the first match
            checkpoint_dir = checkpoint_dir + "/latest-unsharded"
            # Load losses
            losses = torch.load(f"{checkpoint_dir}/{file_name}_losses.pt")  # N x (T-1)
            labels = torch.load(f"{checkpoint_dir}/{file_name}_labels.pt")  # N x (T)

            if random_example_indices is None:
                random_example_indices = random.sample(range(labels.shape[0]), num_examples_to_plot)

            # Compute log compute
            log_compute = compute_log_flops(size, 5.4e9)  # Adding 1 to match full sequence length
            model_size_to_log_compute[size] = log_compute  # Store for labeling

            # Store losses for the same 100 labels for all sizes
            avg_loss = 0.0
            for idx in range(len(random_example_indices)):
                label_losses = losses[random_example_indices[idx], random_label_indices[idx]].numpy()
                label_log_compute = log_compute

                avg_loss += label_losses

                str_idx = str(random_example_indices[idx]) + str(random_label_indices[idx])

                if str_idx not in all_label_losses:
                    all_label_losses[str_idx] = {"log_compute": [], "losses": []}

                all_label_losses[str_idx]["log_compute"].append(label_log_compute)
                all_label_losses[str_idx]["losses"].append(label_losses)
            
            avg_loss = avg_loss/len(random_example_indices)

            # Store average loss per size for the selected labels
            # avg_losses = [np.mean([losses[idx, -1].numpy() for idx in random_label_indices])]
            all_avg_logs.append((log_compute, avg_loss))

        ci = 0
        for label_id in all_label_losses:
            label = all_label_losses[label_id]
            plt.plot(label["log_compute"], label["losses"], color=cmap(ci), alpha=0.7)
            ci += 1

        # Plot averaged losses as a thick blue line
        if all_avg_logs:
            log_computes, avg_losses = zip(*sorted(all_avg_logs))  # Ensure sorted order for proper plotting
            plt.plot(log_computes, avg_losses, 'b-', linewidth=2, label=f"Avg Loss (Seed {seed})")

        # Sort model sizes by log compute
        sorted_sizes = sorted(model_size_to_log_compute.keys(), key=lambda s: model_size_to_log_compute[s])
        sorted_log_computes = [model_size_to_log_compute[s] for s in sorted_sizes]

        # Set custom ticks
        plt.xticks(sorted_log_computes, sorted_sizes)
        plt.xlabel("Model Size (Params) - Log FLOPS")

        plt.ylabel("Token Loss")
        plt.title(f"Token-Level Scaling Laws (Labels, Seed {seed})")
        plt.ylim(2, 7)  # Limit y-axis between 2 and 5
        plt.legend()
        plt.grid(True)

        # Save plot
        plot_path = os.path.join(save_dir, f"scaling_seed_{seed}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()  # Close to avoid overlapping figures

        print(f"Saved plot: {plot_path}")


def random_positions(labels):
    # Flatten the tensor and get unique values with their counts
    unique_values, counts = torch.unique(labels, return_counts=True)

    # Filter values that occur at least 10 times
    frequent_values = unique_values[counts >= 100]
    print(len(frequent_values))

    # Randomly sample up to 100 values (without replacement)
    num_samples = min(100, len(frequent_values))  # Handle cases where there are fewer than 100 valid values
    sampled_values = frequent_values[torch.randperm(len(frequent_values))[:num_samples]]

    result = []
    for v in sampled_values:
        # Get positions of occurrences of value "v"
        positions = torch.where(labels == v)
        # Convert positions to list of (x, y) tuples
        result.append(positions)

    return result


def plot_specific_token_compute():
    # Create directory for saving plots
    save_dir = "plots_specific_token_compute_scaling"
    os.makedirs(save_dir, exist_ok=True)

    num_examples_to_plot = 100

    # File setup
    checkpoint_template = "/network/scratch/a/arkil.patel/olmo/checkpoints/OLMo_{size}_{seed}-*"
    file_name = "ppl-validation"

    random_label_indices = random.sample(range(1023), num_examples_to_plot)
    random_example_indices = None

    # Plot for each seed
    for seed in seeds:
        plt.figure(figsize=(8, 6))

        all_example_losses = {}  # Store losses for each example across sizes
        model_size_to_log_compute = {}
        all_label_losses = {}

        all_avg_logs = []

        for size in size_to_params.keys():
            checkpoint_pattern = checkpoint_template.format(size=size, seed=seed)
            matching_dirs = glob.glob(checkpoint_pattern)

            if not matching_dirs:
                print(f"Warning: No checkpoint found for size {size}, seed {seed}")
                continue  # Skip this size if no matching directory is found

            checkpoint_dir = matching_dirs[0]  # Use the first match
            checkpoint_dir = checkpoint_dir + "/latest-unsharded"
            # Load losses
            losses = torch.load(f"{checkpoint_dir}/{file_name}_losses.pt")  # N x (T-1)
            labels = torch.load(f"{checkpoint_dir}/{file_name}_labels.pt")  # N x (T)

            # pdb.set_trace()

            positions = random_positions(labels)

            # Compute log compute
            log_compute = compute_log_flops(size, 5.4e9)  # Adding 1 to match full sequence length
            model_size_to_log_compute[size] = log_compute  # Store for labeling

            # Store losses for the same 100 labels for all sizes
            avg_loss = 0.0
            for idx in range(len(positions)):
                cur_pos = positions[idx]
                label_losses = losses[cur_pos].numpy()
                label_losses = np.mean(label_losses)
                label_log_compute = log_compute

                avg_loss += label_losses

                if idx not in all_label_losses:
                    all_label_losses[idx] = {"log_compute": [], "losses": []}

                all_label_losses[idx]["log_compute"].append(label_log_compute)
                all_label_losses[idx]["losses"].append(label_losses)
            
            avg_loss = avg_loss/len(positions)

            # Store average loss per size for the selected labels
            # avg_losses = [np.mean([losses[idx, -1].numpy() for idx in random_label_indices])]
            all_avg_logs.append((log_compute, avg_loss))

        # Plot the losses of the selected labels
        ci = 0
        for label_id in all_label_losses:
            label = all_label_losses[label_id]
            plt.plot(label["log_compute"], label["losses"], color=cmap(ci), alpha=0.7)
            ci += 1

        # Plot averaged losses as a thick blue line
        if all_avg_logs:
            log_computes, avg_losses = zip(*sorted(all_avg_logs))  # Ensure sorted order for proper plotting
            plt.plot(log_computes, avg_losses, 'b-', linewidth=2, label=f"Avg Loss (Seed {seed})")

        # Sort model sizes by log compute
        sorted_sizes = sorted(model_size_to_log_compute.keys(), key=lambda s: model_size_to_log_compute[s])
        sorted_log_computes = [model_size_to_log_compute[s] for s in sorted_sizes]

        # Set custom ticks
        plt.xticks(sorted_log_computes, sorted_sizes)
        plt.xlabel("Model Size (Params) - Log FLOPS")

        plt.ylabel("Token Loss")
        plt.title(f"Token-Level Scaling Laws (Labels, Seed {seed})")
        plt.ylim(2, 8)  # Limit y-axis between 2 and 5
        plt.legend()
        plt.grid(True)

        # Save plot
        plot_path = os.path.join(save_dir, f"scaling_seed_{seed}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()  # Close to avoid overlapping figures

        print(f"Saved plot: {plot_path}")



# def plot_label_scaling():
#     save_dir = "plots_label_scaling"
#     os.makedirs(save_dir, exist_ok=True)

#     checkpoint_template = "/network/scratch/a/arkil.patel/olmo/checkpoints/OLMo_136M_*_indi_{step}-*"
#     file_name = "ppl-validation"

#     all_avg_logs = []
#     token_loss_by_step = defaultdict(lambda: defaultdict(list))  # token_id -> step -> [losses]
#     token_freq_counter = Counter()


#     # Step 1: Compute token frequencies only once
#     # Load labels from any one checkpoint
#     checkpoint_dirs = glob.glob("/network/scratch/a/arkil.patel/olmo/checkpoints/OLMo_136M_*_indi_*")
#     for ckpt_dir in checkpoint_dirs:
#         latest_dir = os.path.join(ckpt_dir, "latest-unsharded")
#         label_path = os.path.join(latest_dir, "ppl-validation_labels.pt")
#         if os.path.exists(label_path):
#             labels = torch.load(label_path)  # shape: [N, T]
#             break

#     labels_flat = labels.view(-1)
#     token_freq_counter = Counter(labels_flat.tolist())
#     top_50_tokens = [tok for tok, _ in token_freq_counter.most_common(50)]

#     # Save mask indices for each token
#     token_idx_map = {tok: (labels_flat == tok).nonzero(as_tuple=True)[0] for tok in top_50_tokens}


#     for step in steps:
#         checkpoint_pattern = checkpoint_template.format(step=step)
#         matching_dirs = glob.glob(checkpoint_pattern)

#         if not matching_dirs:
#             print(f"Warning: No checkpoint found for step {step}")
#             continue

#         step_losses = []
#         log_compute = np.log10(802698.2491095e9 * step)

#         for ckpt_dir in matching_dirs:
#             next_step_dir = os.path.join(ckpt_dir, f"step{step + 10}-unsharded")
#             if not os.path.isdir(next_step_dir):
#                 continue

#             checkpoint_dir = ckpt_dir + "/latest-unsharded"
#             try:
#                 losses = torch.load(f"{checkpoint_dir}/{file_name}_losses.pt")  # N x T
#                 labels = torch.load(f"{checkpoint_dir}/{file_name}_labels.pt")  # N x T
#             except:
#                 print(f"No eval loss calculated for step {step}")
#                 continue

#             losses_flat = losses.view(-1)
#             step_losses.extend(losses_flat.tolist())

#             for tok in top_50_tokens:
#                 indices = token_idx_map[tok]
#                 selected_losses = losses_flat[indices]
#                 token_loss_by_step[tok][log_compute].append(selected_losses.mean().item())

#         if step_losses:
#             all_avg_logs.append((log_compute, np.mean(step_losses)))

#     # Plot average dataset loss
#     plt.figure(figsize=(10, 7))

#     if all_avg_logs:
#         log_computes, avg_losses = zip(*sorted(all_avg_logs))
#         log_avg_losses = np.log(avg_losses)
#         coeffs = np.polyfit(log_computes, log_avg_losses, deg=1)
#         m, b = coeffs
#         x_fit = np.linspace(min(log_computes), max(log_computes), 100)
#         y_fit = np.exp(m * x_fit + b)

#         plt.plot(x_fit, y_fit, color="black", linestyle="--", label="Dataset Avg (Fit)")
#         plt.scatter(log_computes, avg_losses, color="black", marker='x', label="Dataset Avg")

#     # Plot top 50 most frequent tokens
#     top_tokens = [tok for tok, _ in token_freq_counter.most_common(50)]

#     for tok in top_tokens:
#         log_loss_pairs = token_loss_by_step[tok]
#         logs, losses = [], []
#         for l, vals in sorted(log_loss_pairs.items()):
#             logs.append(l)
#             losses.append(np.mean(vals))
#         if len(logs) >= 2:  # Require at least 2 points to plot a curve
#             plt.plot(logs, losses, alpha=0.5, label=tokenizer.decode([tok]).replace("\u0120", " ").strip())

#     plt.xlabel("Log Compute (FLOPs)")
#     plt.ylabel("Loss")
#     plt.yscale("log")
#     plt.title("Token-wise Scaling Trends (Top 50 Frequent Tokens)")
#     plt.legend(fontsize='small', ncol=2)
#     plt.grid(True)
#     plt.tight_layout()

#     plot_path = os.path.join(save_dir, f"tokenwise_scaling.png")
#     plt.savefig(plot_path, dpi=300, bbox_inches="tight")
#     plt.close()

#     print(f"Saved plot: {plot_path}")



def plot_label_scaling():
    save_dir = "plots_label_scaling"
    os.makedirs(save_dir, exist_ok=True)

    checkpoint_template = "/network/scratch/a/arkil.patel/olmo/checkpoints/OLMo_136M_*_indi_{step}-*"
    file_name = "ppl-validation"

    all_avg_logs = []
    token_loss_by_step = defaultdict(lambda: defaultdict(list))

    # Step 1: Precompute labels and top/bottom token index masks
    # Load labels from any one checkpoint
    checkpoint_dirs = glob.glob("/network/scratch/a/arkil.patel/olmo/checkpoints/OLMo_136M_*_indi_*")
    for ckpt_dir in checkpoint_dirs:
        latest_dir = os.path.join(ckpt_dir, "latest-unsharded")
        label_path = os.path.join(latest_dir, f"{file_name}_labels.pt")
        if os.path.exists(label_path):
            labels = torch.load(label_path)  # shape: [N, T]
            break

    labels_flat = labels.view(-1)
    token_freq_counter = Counter(labels_flat.tolist())
    all_tokens_sorted = [tok for tok, _ in token_freq_counter.most_common()]

    top_20_tokens = all_tokens_sorted[:15]
    bottom_20_tokens = all_tokens_sorted[-15:]

    # Store masks only for those tokens
    token_idx_map = {tok: (labels_flat == tok).nonzero(as_tuple=True)[0] for tok in top_20_tokens + bottom_20_tokens}

    # Step 2: Iterate over steps and collect losses
    for step in steps:
        checkpoint_pattern = checkpoint_template.format(step=step)
        matching_dirs = glob.glob(checkpoint_pattern)

        if not matching_dirs:
            print(f"Warning: No checkpoint found for step {step}")
            continue

        step_losses = []
        log_compute = np.log10(802698.2491095e9 * step)

        for ckpt_dir in matching_dirs:
            checkpoint_dir = os.path.join(ckpt_dir, "latest-unsharded")
            try:
                losses = torch.load(f"{checkpoint_dir}/{file_name}_losses.pt")  # N x T
            except:
                print(f"No eval loss calculated for step {step}")
                continue

            losses_flat = losses.view(-1)
            step_losses.extend(losses_flat.tolist())

            for tok in token_idx_map:
                indices = token_idx_map[tok]
                selected_losses = losses_flat[indices]
                token_loss_by_step[tok][log_compute].append(selected_losses.mean().item())

        if step_losses:
            all_avg_logs.append((log_compute, np.mean(step_losses)))

    # Step 3: Plotting
    plt.figure(figsize=(12, 8))

    # Plot dataset average line and fit
    if all_avg_logs:
        log_computes, avg_losses = zip(*sorted(all_avg_logs))
        log_avg_losses = np.log(avg_losses)
        m, b = np.polyfit(log_computes, log_avg_losses, deg=1)
        x_fit = np.linspace(min(log_computes), max(log_computes), 100)
        y_fit = np.exp(m * x_fit + b)

        plt.plot(x_fit, y_fit, color="black", linestyle="--", label="Dataset Avg (Fit)")
        plt.scatter(log_computes, avg_losses, color="black", marker='x', label="Dataset Avg")

    # Color maps
    top_cmap = get_cmap("Blues")
    bottom_cmap = get_cmap("Oranges")

    # Normalize ranks for color shading
    max_rank = 14
    for i, tok in enumerate(top_20_tokens):
        log_loss_pairs = token_loss_by_step[tok]
        logs, losses = [], []
        for l, vals in sorted(log_loss_pairs.items()):
            logs.append(l)
            losses.append(np.mean(vals))
        if len(logs) < 2:
            continue
        losses = np.array(losses)
        log_losses = np.log(losses)
        coeffs = np.polyfit(logs, log_losses, deg=1)
        x_fit = np.linspace(min(logs), max(logs), 100)
        y_fit = np.exp(coeffs[0] * x_fit + coeffs[1])

        color = top_cmap((max_rank - i) / max_rank)
        plt.scatter(logs, losses, s=10, color=color, label=None)
        plt.plot(x_fit, y_fit, color=color, label=f"{tokenizer.decode([tok]).replace('Ġ',' ').strip()}")
        # plt.plot(logs, losses, color=color, label=f"{tokenizer.decode([tok]).replace('Ġ',' ').strip()}")

    for i, tok in enumerate(bottom_20_tokens):
        log_loss_pairs = token_loss_by_step[tok]
        logs, losses = [], []
        for l, vals in sorted(log_loss_pairs.items()):
            logs.append(l)
            losses.append(np.mean(vals))
        if len(logs) < 2:
            continue
        losses = np.array(losses)
        log_losses = np.log(losses)
        coeffs = np.polyfit(logs, log_losses, deg=1)
        x_fit = np.linspace(min(logs), max(logs), 100)
        y_fit = np.exp(coeffs[0] * x_fit + coeffs[1])

        color = bottom_cmap(i / max_rank)
        plt.scatter(logs, losses, s=10, color=color, label=None)
        plt.plot(x_fit, y_fit, color=color, label=f"{tokenizer.decode([tok]).replace('Ġ',' ').strip()}")
        # plt.plot(logs, losses, color=color, label=f"{tokenizer.decode([tok]).replace('Ġ',' ').strip()}")

    plt.xlabel("Log Compute (FLOPs)")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title("Scaling Trends for Frequent and Rare Tokens")
    plt.legend(fontsize="small", ncol=2)
    plt.grid(True)
    plt.tight_layout()

    plot_path = os.path.join(save_dir, f"tokenwise_scaling_top_bottom_20.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved plot: {plot_path}")




def plot_label_contexts_scaling(k=0, m=50):
    save_dir = "plots_label_contexts_scaling"
    os.makedirs(save_dir, exist_ok=True)

    checkpoint_template = "/network/scratch/a/arkil.patel/olmo/checkpoints/OLMo_136M_*_indi_{step}-*"
    file_name = "ppl-validation"

    # === Step 1: Load labels and get kth most frequent token ===
    checkpoint_dirs = glob.glob("/network/scratch/a/arkil.patel/olmo/checkpoints/OLMo_136M_*_indi_*")
    for ckpt_dir in checkpoint_dirs:
        latest_dir = os.path.join(ckpt_dir, "latest-unsharded")
        label_path = os.path.join(latest_dir, f"{file_name}_labels.pt")
        if os.path.exists(label_path):
            labels = torch.load(label_path)  # shape: [N, T]
            break

    labels_flat = labels.view(-1)
    token_freq_counter = Counter(labels_flat.tolist())
    sorted_tokens = [tok for tok, _ in token_freq_counter.most_common()]
    target_token = sorted_tokens[k]
    token_str = tokenizer.decode([target_token]).replace("Ġ", " ").strip()
    token_indices = (labels_flat == target_token).nonzero(as_tuple=True)[0]

    sampled_indices = token_indices[torch.randperm(len(token_indices))[:m]].tolist()

    print(f"Token ID: {target_token}, String: '{token_str}', Showing {len(sampled_indices)} / {len(token_indices)} contexts")

    # === Step 2: Collect per-step losses ===
    per_context_losses = {i: [] for i in sampled_indices}
    avg_losses = []
    log_computes = []

    # pdb.set_trace()

    for step in steps:
        checkpoint_pattern = checkpoint_template.format(step=step)
        matching_dirs = glob.glob(checkpoint_pattern)
        if not matching_dirs:
            continue

        log_compute = np.log10(802698.2491095e9 * step)

        for ckpt_dir in matching_dirs:
            next_step_dir = os.path.join(ckpt_dir, f"step{step + 10}-unsharded")

            if not os.path.isdir(next_step_dir):
                print(f"Latest saved checkpoint not found: {next_step_dir}")
                continue

            checkpoint_dir = os.path.join(ckpt_dir, "latest-unsharded")
            loss_path = os.path.join(checkpoint_dir, f"{file_name}_losses.pt")

            try:
                losses = torch.load(loss_path).view(-1)  # shape: [N*T]
            except:
                print(f"Skipping missing loss for step {step}")
                continue

            context_vals = losses[sampled_indices]
            for idx, val in zip(sampled_indices, context_vals):
                per_context_losses[idx].append(val.item())
            avg_losses.append(context_vals.mean().item())
            log_computes.append(log_compute)

    # === Step 3: Plot ===
    plt.figure(figsize=(10, 7))

    # Use colorblind-friendly colormap (Set2 or tab20 or a custom palette)
    color_map = plt.get_cmap("tab20")
    num_colors = len(per_context_losses)
    color_cycle = [color_map(i % 20) for i in range(num_colors)]

    # Plot each individual context line
    # for (context_id, loss_seq), color in zip(per_context_losses.items(), color_cycle):
    #     if len(loss_seq) == len(log_computes):
    #         plt.plot(log_computes, loss_seq, lw=1.5, alpha=0.8, color=color)
    #         plt.scatter(log_computes, loss_seq, color=color)
    # for (context_id, loss_seq), color in zip(per_context_losses.items(), color_cycle):
    #     if len(loss_seq) == len(log_computes):
    #         x_vals, y_vals = average_duplicate_x(log_computes, loss_seq)
    #         plt.plot(x_vals, y_vals, lw=1.5, alpha=0.8, color=color)
            # plt.scatter(x_vals, y_vals, color=color, s=10)

    for (context_id, loss_seq), color in zip(per_context_losses.items(), color_cycle):
        if len(loss_seq) == len(log_computes):
            x_vals, y_vals = average_duplicate_x(log_computes, loss_seq)
            if len(y_vals) >= 3:  # ensure enough points for smoothing
                smoothed_y = new_moving_average(y_vals, window_size=3)
                smoothed_x = x_vals[:len(smoothed_y)]  # match length
                plt.plot(smoothed_x, smoothed_y, lw=1.5, alpha=0.8, color=color)


    # Plot average scatter and fit
    if len(avg_losses) == len(log_computes):
        x_vals, y_vals = average_duplicate_x(log_computes, avg_losses)
        plt.scatter(x_vals, y_vals, color="black", marker='x', label="Average Loss")

        log_losses = np.log(y_vals)
        m_fit, b_fit = np.polyfit(x_vals, log_losses, deg=1)
        x_fit = np.linspace(min(x_vals), max(x_vals), 100)
        y_fit = np.exp(m_fit * x_fit + b_fit)

        plt.plot(x_fit, y_fit, color="black", linestyle="--", linewidth=2, label="Avg Fit")

    plt.title(f"Scaling for Token '{token_str}' (ID {target_token}) — {len(sampled_indices)} Contexts")
    plt.xlabel("Log Compute (FLOPs)")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.ylim(0.5, 5)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    fname = f"token_{target_token}_{token_str.replace(' ', '_')}_m{len(sampled_indices)}.png"
    path = os.path.join(save_dir, fname)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {path}")

def average_duplicate_x(x_vals, y_vals):
    """Given lists of x and y values (possibly with duplicate x), average y values per x."""
    grouped = defaultdict(list)
    for x, y in zip(x_vals, y_vals):
        grouped[x].append(y)
    x_unique = sorted(grouped.keys())
    y_avg = [np.mean(grouped[x]) for x in x_unique]
    return x_unique, y_avg

def new_moving_average(y, window_size=5):
    return np.convolve(y, np.ones(window_size)/window_size, mode='valid')


# plot_last_token_compute()
# plot_random_token_compute()
# plot_specific_token_compute()
# plot_last_token_data_functional()
# get_data_freq()


# plot_last_token_data()
# plot_label_scaling()
plot_label_contexts_scaling(k=2, m=30)