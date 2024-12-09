import os, sys, json


A4_DIR = "/home/anmola/assignments_hw/anlp_final_proj/outputs_A4"
# find all json files recursively in the directory
def find_json_files_in_dir(directory):
    json_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            # print(file)
            if file.endswith('.jsonl'):
                json_files.append(os.path.join(root, file))
    return json_files

json_files = find_json_files_in_dir(A4_DIR)

json_files = [x for x in json_files if "do_not_use" not in x]
json_files = [x for x in json_files if "DPO_NO" not in x]
print("Total number of json files: ", len(json_files))

# here is the filename: generated_codes_deepseek_base_1e-4_NO_cot_speed_incorrect_samples_10_epoch_edit_deepseek_instruct_nrowsNone_tokens1024_temp0.0_fewshotex0_samples1_2024-12-08_00:33:24.jsonl 

# remove timestamp from the filename BUT retain the extension
def remove_timestamp_from_filename(filename):
    return filename.rsplit('_', 2)[0] + ".jsonl"


# read all the json files and map the filename to the data
inference_outputs_df = {}
judge_results_df = {}

for file in json_files:
    print(f"{file=}")
    file_category = "infer" if "inference_outputs" in file else "judge"

    if file_category == "infer":
        with open(file, 'r') as f:
            data = f.readlines()
            inference_outputs_df[remove_timestamp_from_filename(os.path.basename(file))] = data
    else:
        with open(file, 'r') as f:
            data = json.load(f)
            _use_filename = remove_timestamp_from_filename(os.path.basename(file))
            _use_filename = _use_filename.replace("generated_codes_1_", "")
            _use_filename = _use_filename.replace("generated_codes_", "")
            judge_results_df[_use_filename] = data

print("Total number of inference files: ", len(inference_outputs_df))
print("Total number of judge files: ", len(judge_results_df))
assert(len(inference_outputs_df) == len(judge_results_df))

# get sorted base filenames
sorted_base_filenames = sorted(list(inference_outputs_df.keys()) + list(judge_results_df.keys()))
print("Total number of sorted base filenames: ", len(sorted_base_filenames))
print("Example sorted base filenames: ", *sorted_base_filenames, sep='\n')
# number of unique elements print
print("Number of unique elements in sorted_base_filenames: ", len(set(sorted_base_filenames)))
assert(len(set(sorted_base_filenames)) == len(inference_outputs_df))