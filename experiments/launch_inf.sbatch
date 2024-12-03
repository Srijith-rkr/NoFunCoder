#!/bin/sh 
#SBATCH --gres=gpu:6000Ada:1
#SBATCH --partition=general
#SBATCH --mem=50Gb
#SBATCH --cpus-per-task=8
#SBATCH -t 0-1:30:00              # time limit:  add - for days (D-HH:MM) 
#SBATCH --job-name=SR_all_samples_700ckpt_1e-3
#SBATCH --error=/home/srijithr/course_hw/anlp_project/job_outputs/inference_jobs/%x__%j.err
#SBATCH --output=/home/srijithr/course_hw/anlp_project/job_outputs/inference_jobs/%x__%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=srijithr@andrew.cmu.edu


source /data/tir/projects/tir7/user_data/srijithr/miniconda3/etc/profile.d/conda.sh
conda activate ecco

export HF_DATASETS_CACHE=/data/tir/projects/tir7/user_data/srijithr/hf_cache_dir
export HF_HOME=/data/tir/projects/tir7/user_data/srijithr/hf_cache_dir
export NCCL_P2P_DISABLE=1 

# export OMP_NUM_THREADS=10


cd /home/srijithr/iterative-alignment/SPIN_implementation/ANLP_A4_ECCO

# python experiments/inference.py --eval_mode edit --finetuned_weights Srijith-rkr/deepseek_base_1e-3_NO_cot_only_failed_samples_3_epoch  G0
# python experiments/inference.py --eval_mode self-refine --finetuned_weights Srijith-rkr/deepseek_base_1e-3_NO_cot_only_failed_samples_3_epoch G1

# python experiments/inference.py --eval_mode edit --finetuned_weights Srijith-rkr/deepseek_base_1e-3_NO_cot_all_samples_700_ckpt
# python experiments/inference.py --eval_mode self-refine --finetuned_weights Srijith-rkr/deepseek_base_1e-3_NO_cot_all_samples_700_ckpt

# python experiments/inference.py --eval_mode edit --finetuned_weights Srijith-rkr/deepseek_base_1e-3_NO_cot_all_samples_3_epoch
# python experiments/inference.py --eval_mode self-refine --finetuned_weights Srijith-rkr/deepseek_base_1e-3_NO_cot_all_samples_3_epoch

# python experiments/inference.py --eval_mode edit --finetuned_weights Srijith-rkr/deepseek_base_1e-3_NO_cot_all_samples_700_ckpt
# python experiments/inference.py --eval_mode self-refine --finetuned_weights Srijith-rkr/deepseek_base_1e-3_NO_cot_all_samples_700_ckpt

# python experiments/inference.py --eval_mode edit --finetuned_weights Srijith-rkr/deepseek_base_1e-3_YES_cot_only_failed_samples_3_epoch
# python experiments/inference.py --eval_mode self-refine --finetuned_weights Srijith-rkr/deepseek_base_1e-3_YES_cot_only_failed_samples_3_epoch


