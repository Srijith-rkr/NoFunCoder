#!/bin/sh 
#SBATCH --gres=gpu:A100_40GB:8
#SBATCH --partition=general
#SBATCH --mem=500Gb
#SBATCH --cpus-per-task=40
#SBATCH -t 0-13:00:00              # time limit:  add - for days (D-HH:MM)  # 1-10:00:00    for L 40 
#SBATCH --job-name=a100iter_2_with_2e_data_2e_ckpt
#SBATCH --error=/home/srijithr/iterative-alignment/SPIN_implementation/job_outputs/%x__%j.err
#SBATCH --output=/home/srijithr/iterative-alignment/SPIN_implementation/job_outputs/%x__%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=srijithr@andrew.cmu.edu

## //--SBATCH --export=ALL,CUDA_LAUNCH_BLOCKING=1,TORCH_USE_CUDA_DSA=1

# THE 8 GPU run failed with CPU mem error with 256Gb, so trying 512Gb
# The long run started muuch earlier than general run 

source /data/tir/projects/tir7/user_data/srijithr/miniconda3/etc/profile.d/conda.sh
conda activate spin

export OMP_NUM_THREADS=5
# export DS_SKIP_CUDA_CHECK=1
export HF_DATASETS_CACHE=/data/tir/projects/tir7/user_data/srijithr/hf_cache_dir
export HF_TOKEN=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

cd /home/srijithr/iterative-alignment/SPIN_implementation/SPIN

# accelerate launch --config_file configs/multi_gpu_debug_stage_1.yaml --num_processes=4 spin/run_spin.py configs/config_debug.yaml --run_name=beta_1e-2_stage1_1024_lora_only_SPIN_iter0_A100_80GB_1e-5 --learning_rate=1e-5 --beta=0.01

# 
# accelerate launch --config_file configs/multi_gpu_debug_stage_3.yaml --num_processes=8 spin/run_spin.py configs/config_debug_full.yaml --run_name=full_SFT_L40_8_27H

#  - long partition   737948
# accelerate launch --config_file configs/multi_gpu_debug_stage_3.yaml --num_processes=4 spin/run_spin.py configs/config_debug_full.yaml --run_name=4_BS_final_full_SFT_L40_4_54H

# SRIJITH _ I CHANGED GRADIENT ACCUMULATION STEPS TO 4 on gpu config 
# 739008
# accelerate launch --config_file configs/multi_gpu_debug_stage_3.yaml --num_processes=4 spin/run_spin.py configs/config_debug_full.yaml --run_name=4_BS_correctGA_final_full_SFT_L40_4_54H  --gradient_accumulation_steps=4 --per_device_train_batch_size=4

# SRIJITH _ I CHANGED GRADIENT ACCUMULATION STEPS TO 2 on gpu config: BS * did not work -> changed it back to 4
# 
accelerate launch --config_file configs/multi_gpu_debug_stage_3.yaml --num_processes=8 spin/run_spin.py configs/config_debug_full_iter2.yaml --run_name=a100iter_2_with_2e_data_2e_ckpt --gradient_accumulation_steps=4 --per_device_train_batch_size=2
# accelerate launch --config_file configs/multi_gpu_debug_stage_3.yaml --num_processes=8 spin/run_spin.py configs/config_debug_full_iter2.yaml --run_name=2048a100iter_2_with_2e_data_2e_ckpt --gradient_accumulation_steps=4 --per_device_train_batch_size=2 --max_length=2048

# below was for iteartaino 2 
# accelerate launch --config_file configs/multi_gpu_debug_stage_3.yaml --num_processes=8 spin/run_spin.py configs/config_debug_full_iter2.yaml --run_name=iter_2_with_2e_data_L40 --gradient_accumulation_steps=4 --per_device_train_batch_size=2

# accelerate launch --config_file configs/multi_gpu_debug_stage_3.yaml --num_processes=8 spin/run_spin.py configs/config_debug_full_iter1.yaml --run_name=iter_1_8xL40  --gradient_accumulation_steps=2 --per_device_train_batch_size=4

# 1 reducsed num cpus in config file - will proaly increase cpu if it is still stuck 
# i am trying iter 0 now - still stuck 
# I am thinkgin this is a gpu issue - changes to a6000 now - this is taking time to start - 
# so i ll just try the new gpu theory on iter 1 itself and let it run if it works 
#if this works I will try local dataset next and local model next 
# got cuda oom in l40s - added this according to errr mesage #SBATCH --export=ALL,CUDA_LAUNCH_BLOCKING=1,TORCH_USE_CUDA_DSA=1
# also adjustd the batch size to 4 from 2 and gradient accumulation
# still got ooom - going back to iter 0 
# the 40s is probaly a smaaler gpu - still getting oom tryin a40
#  finally A40 is working - 
# now trying the iteration 1 
# WORKING NOW - the probelm was with the gpu - lidia was using all the cpus and hence the proble 
# also A40 seems remarkably fast 
# I checked the memory and it is 33.5/ 40 gb - 2 BS itself is fine

# I reduced num cpus back to 80 for iter 2 - chage this if something doen not work


# /data/tir/projects/tir7/user_data/srijithr/miniconda3/condabin/conda /data/tir/projects/tir7/user_data/srijithr/miniconda3/lib/


# l40 is 128 only | happens next month only 
# a6000 is 64 and 128 - good here 
# 6000Ada is 64 only | 