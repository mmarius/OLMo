import os

# Define base SLURM script
base_script = """#!/bin/bash

#SBATCH --partition=long
#SBATCH --cpus-per-task=4                               
#SBATCH --mem=24G
#SBATCH --gres=gpu:l40s:2
#SBATCH --time=72:00:00
#SBATCH -o /network/scratch/a/arkil.patel/olmo/slurm_logs/olmo_train_267M_479_indi-{max_duration}.out

# 1. Load the required modules
module load python/3.10 cuda/12.6.0/cudnn

# 2. Load your environment
source ~/envs/scalenv/bin/activate
cd ~/OLMo

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 scripts/train.py configs/slim_pajama/OLMo-267M_indi.yaml \
    --seed=479 \
    --device_train_microbatch_size=32 \
    --save_folder=/network/scratch/a/arkil.patel/olmo/checkpoints \
    --max_duration={max_duration} \
    --scheduler.t_decay={td} \
    --run_name=OLMo_267M_479_indi_{max_duration}
"""

# Define the range of steps
step_values = range(500, 3250, 250)  # 5200 to include 5150

# Create scripts directory
scripts_dir = "slurm_scripts"
os.makedirs(scripts_dir, exist_ok=True)

# Loop through step values, generate SLURM scripts, and submit them
for steps in step_values:
    max_duration = steps
    td = int(0.2 * steps)
    script_content = base_script.format(max_duration=max_duration, td=td)
    script_filename = os.path.join(scripts_dir, f"full_train_{steps}.sh")
    
    # Write script to file
    with open(script_filename, "w") as f:
        f.write(script_content)
    
    # Make script executable
    os.chmod(script_filename, 0o755)
    
    # Submit the job
    os.system(f"sbatch {script_filename}")

print("All jobs submitted!")
