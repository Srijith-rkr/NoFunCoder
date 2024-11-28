import json
import os
from datasets import load_dataset
import pyarrow.parquet as pq
import random
random.seed(42)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--num_fracs', type=int, default=1)
parser.add_argument('--input_dir', type=str, default='data/gen_using_iter0_full_train')
parser.add_argument('--output_dir', type=str, default='data/gen_using_iter0_full_train_parquet')

args = parser.parse_args()
num_fracs = args.num_fracs
input_dir = args.input_dir
output_dir = args.output_dir

dataset = load_dataset('UCLA-AGI/SPIN_iter0', split='train')
dataset = list(dataset)


save_path = f'{input_dir}/synthetic_train_iter0_from_repo.json'
with open(save_path, 'w') as f:
    json.dump(dataset, f, indent=4)

with open(save_path, 'r') as f:
    dataset = json.load(f)

print(len(dataset))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
print('Checking if HF can load the datsaet')

dataset = load_dataset('json', data_files=save_path ,split='train')
