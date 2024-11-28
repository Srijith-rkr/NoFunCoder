#!/bin/sh 
#SBATCH --gres=gpu:A6000:1
#SBATCH --partition=general
#SBATCH --mem=50Gb
#SBATCH --cpus-per-task=4
#SBATCH -t 0-5:00:00              # time limit:  add - for days (D-HH:MM) 
#SBATCH --job-name=gen_dpo_S0
#SBATCH --error=/home/srijithr/iterative-alignment/SPIN_implementation/ANLP_A4_ECCO/job_outputs/generate_dpo_data/%x__%j.err
#SBATCH --output=/home/srijithr/iterative-alignment/SPIN_implementation/ANLP_A4_ECCO/job_outputs/generate_dpo_data/%x__%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=srijithr@andrew.cmu.edu


source /data/tir/projects/tir7/user_data/srijithr/miniconda3/etc/profile.d/conda.sh
conda activate ecco

export OMP_NUM_THREADS=4
export HF_DATASETS_CACHE=/data/tir/projects/tir7/user_data/srijithr/hf_cache_dir
export HF_TOKEN=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

cd /home/srijithr/iterative-alignment/SPIN_implementation/ANLP_A4_ECCO/

python experiments/generate_edit_data_for_DPO.py --run_name=base_deepseek --frac_samples=0.25 --subset=0
# python experiments/generate_edit_data_for_DPO.py --run_name=base_deepseek --frac_samples=0.25 --subset=1
# python experiments/generate_edit_data_for_DPO.py --run_name=base_deepseek --frac_samples=0.25 --subset=2
# python experiments/generate_edit_data_for_DPO.py --run_name=base_deepseek --frac_samples=0.25 --subset=3
