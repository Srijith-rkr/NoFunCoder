path = '/Users/srijith/courses/11711-ANLP/anlp_project/temp/ANLP_A4_ECCO/inference_generations/gen_dpo_data/base_deepseek_0_temp'
import os 
from tqdm import tqdm
import json
files = os.listdir(path)
json_data = []
for file in files:
        with open(os.path.join(path, file), 'r') as f:
            for line in f:
                data = json.loads(line)
                json_data.append(data)

print('for debug')
with open(os.path.join(path,'base_deepseek_0_0.jsonl'), 'w') as f:
    for data in json_data:
        json.dump(data, f)
        f.write('\n')
# write code to save this as jsonl file 