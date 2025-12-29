import numpy as np
import matplotlib.pyplot as plt

# Power law parameters
slope = -0.6278
intercept = 12.6875

# Step values from 50 to 5000
steps = np.arange(50, 5001, 50)  # (spacing of 50 to have ~100 points)

# Compute max_duration, tokens_covered, and log_compute
max_duration = 1.25 * steps
tokens_covered = max_duration * 1024 * 1024
log_compute = np.log10(6 * 190335744 * tokens_covered)

# Compute the true loss values from the power law
log_loss = slope * log_compute + intercept
loss = np.exp(log_loss)

# Add multiplicative noise (~±10%)
np.random.seed(42)
noise = np.random.uniform(0.95, 1.05, size=loss.shape)
noisy_loss = loss * noise

# Plot
plt.figure(figsize=(8, 5))
# Plot smooth true power law line (for reference)
log_flops_smooth = np.linspace(log_compute.min(), log_compute.max(), 200)
loss_smooth = np.exp(slope * log_flops_smooth + intercept)
plt.plot(log_flops_smooth, loss_smooth, label='Power law fit', color='blue')

# Plot noisy simulated data
plt.scatter(log_compute, noisy_loss, color='red', s=20, label='data')

plt.xlabel('Log FLOPs')
plt.ylabel('Loss')
plt.yscale('log')
plt.title('Power Law trend')
plt.legend()
plt.grid(True, which="both", ls="--", lw=0.5)
plt.savefig("other_plots/power_law.png", dpi=300, bbox_inches="tight")
