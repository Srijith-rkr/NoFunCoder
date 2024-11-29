#!/bin/sh 
#SBATCH --gres=gpu:A6000:4
#SBATCH --partition=general
#SBATCH --mem=200Gb
#SBATCH --cpus-per-task=40
#SBATCH -t 0-13:00:00         
#SBATCH --job-name=train_dpo
#SBATCH --error=/home/srijithr/iterative-alignment/SPIN_implementation/ANLP_A4_ECCO/job_outputs/train_dpo/%x__%j.err
#SBATCH --output=/home/srijithr/iterative-alignment/SPIN_implementation/ANLP_A4_ECCO/job_outputs/train_dpo/%x__%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=srijithr@andrew.cmu.edu

## //--SBATCH --export=ALL,CUDA_LAUNCH_BLOCKING=1,TORCH_USE_CUDA_DSA=1

source /data/tir/projects/tir7/user_data/srijithr/miniconda3/etc/profile.d/conda.sh
conda activate spin

export OMP_NUM_THREADS=4 # should be equal to cpus-per-task / num gpus
export NCCL_P2P_DISABLE=1 3 
# export DS_SKIP_CUDA_CHECK=1
export HF_DATASETS_CACHE=/data/tir/projects/tir7/user_data/srijithr/hf_cache_dir

cd /home/srijithr/iterative-alignment/SPIN_implementation/ANLP_A4_ECCO/

accelerate launch --config_file Iterative_DPO/configs/multi_gpu_debug_stage_1.yaml --num_processes=2  Iterative_DPO/spin/run_spin.py Iterative_DPO/configs/config_debug_lora.yaml --run_name=debugging --learning_rate=1e-3
