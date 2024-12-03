import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
os.environ["HF_DATASETS_CACHE"] = "/data/tir/projects/tir7/user_data/srijithr/hf_cache_dir"

from datetime import datetime
import sys
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM #,  TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import PeftModel



# for local testing
BASE_MODEL = 'deepseek-ai/deepseek-coder-7b-instruct-v1.5'
OUTPUT_DIR = '/data/tir/projects/tir7/user_data/srijithr/anlp4_checkpoints/path_to_save_models'
# CHECKPOINT ='/data/tir/projects/tir7/user_data/srijithr/anlp4_checkpoints/deepseek_base_1e-3_NO_cot_only_failed_samples/checkpoint-1000'
# RUN_NAME = 'deepseek_base_1e-3_NO_cot_only_failed_samples_1000_ckpt'

# CHECKPOINT ='/data/tir/projects/tir7/user_data/srijithr/anlp4_checkpoints/deepseek_base_1e-3_NO_cot_all_samples/checkpoint-700'
# RUN_NAME = 'deepseek_base_1e-3_NO_cot_all_samples_700_ckpt'

# CHECKPOINT ='/data/tir/projects/tir7/user_data/srijithr/anlp4_checkpoints/deepseek_base_1e-3_YES_cot_only_failed_samples/checkpoint-4700'
# RUN_NAME = 'deepseek_base_1e-3_YES_cot_only_failed_samples_3_epoch'

# CHECKPOINT = '/data/tir/projects/tir7/user_data/srijithr/anlp4_checkpoints/deepseek_base_1e-4_NO_cot_only_failed_samples/checkpoint-4700'
# RUN_NAME = 'deepseek_base_1e-4_NO_cot_only_failed_samples_3_epoch'

CHECKPOINT = '/data/tir/projects/tir7/user_data/srijithr/anlp4_checkpoints/deepseek_base_1e-4_NO_cot_all_samples/checkpoint-8900'
RUN_NAME = 'deepseek_base_1e-4_NO_cot_all_samples_3_epoch'

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map= 'auto',
)
print('Model loaded successfully')


model = PeftModel.from_pretrained(model, CHECKPOINT) 
merged_model = model.merge_and_unload()
print('Model merged successfully')
merged_model.push_to_hub(RUN_NAME)
tokenizer.push_to_hub(RUN_NAME)
print('Model pushed to hub successfully')
print('X'*1000)
# if you want to save it locally
# merged_model.save_pretrained(save_path)
# print('Model saved successfully')
# tokenizer.save_pretrained(save_path)
# print('Tokenizer saved successfully')

# print('Saving final model to', save_path)
# torch.save(merged_model.state_dict(), save_path)
# print('Model saved successfully')
# merged_model.push_to_hub(args.hub_model_name)
