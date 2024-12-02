# Training Setup Instructions

## 1. Environment Setup
Build the Spin environment using the provided `spin.yml` file:

```bash
conda env create -f spin.yml


## 2. Data:
You can find the data in : https://drive.google.com/drive/folders/1sAc1Ei9Z8wmDdswjviuYPkeVxH0NK4nN?usp=sharing


## 3. Configuration File: configs/config_debug_lora.yaml

Edit the following parameters in the configuration file:


Data path:
    •	Update path to you local dataset
    local_dataset_mixer:
  - ANLP_A4_ECCO/data/judge_base_deepseek_0_0.jsonl

Model Selection
	•	To use the base model:
    model_name_or_path: deepseek-ai/deepseek-coder-7b-instruct-v1.5
    •	To use the SFT model trained on A3:
    model_name_or_path: Srijith-rkr/deepseek_SFT_history

Negative Samples
	•	Include negative samples with incorrect reasoning:
    use_cot: true
    Note: Positive samples do not have reasoning steps.

Failed Sample Training
	•	To train only on samples your model predicted incorrectly:
    use_only_failed_samples: true
    All samples will be used if this is not set.

Learning Rate
	•	Ensure the learning rate is set as a float, not a string:
    learning_rate: 5.0e-3

Output Directory
	•	Specify the directory to save results (e.g., the data directory):
    output_dir: ./data

Run Name
	•	Set an identifiable run name. The model will be saved under this folder:
    run_name: DPO_iteration_1_base_model_use_cot_use_negative_samples_1e-3

gradient_accumulation_steps
    •	Set this such that num_gpus * gradient_accumulation_steps * gradient_accumulation_steps = 32
    since gradient_accumulation_steps = 1 in most cases 
    # for 2 GPUS: gradient_accumulation_steps = 16
    # for 4 GPUS: gradient_accumulation_steps = 8

Do not forget to updated the  --num_processes=2  in the sbatch script to the number of GPUs finally 

accelerate launch --config_file Iterative_DPO/configs/multi_gpu_debug_stage_1.yaml --num_processes=2  Iterative_DPO/spin/run_spin.py Iterative_DPO/configs/config_debug_lora.yaml --run_name=debugging --learning_rate=1e-3
