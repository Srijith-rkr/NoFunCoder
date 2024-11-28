# reference: https://medium.com/@geronimo7/llms-multi-gpu-inference-with-accelerate-5a8333e4c5db

from accelerate import Accelerator
from accelerate.utils import gather_object
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

import argparse
import torch, time, json, os
from pathlib import Path
from tqdm import tqdm
from datetime import timedelta
from accelerate.utils import InitProcessGroupKwargs

import warnings
warnings.filterwarnings("ignore")

kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=36000))
accelerator = Accelerator(kwargs_handlers=[kwargs])

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    # iter 0 
    # parser.add_argument('--model', type=str, default='/data/tir/projects/tir7/user_data/srijithr/spin_outputs/geneal_8_gpu_wSAVECKPT_final_full_SFT_L40/')
    # at epoch 2 ; /data/tir/projects/tir7/user_data/srijithr/spin_outputs/geneal_8_gpu_wSAVECKPT_final_full_SFT_L40/checkpoint-1600/
    # ITER 1 GENERAED BUT 3RD EPCOH: '/data/tir/projects/tir7/user_data/srijithr/spin_outputs/iter_1_8xL40S-5/'
    # ITER 1 generated 2nd epoch: /data/tir/projects/tir7/user_data/srijithr/spin_outputs/iter_1_with_2e_data_L40
    # /data/tir/projects/tir7/user_data/srijithr/spin_outputs/iter_1_with_2e_data_L40'
    # In the above model I used the correct data genearated by the model trained on 2 epochs of data but continued training with the 3 epoch checkpoint - the below model is the corrected version

    
    parser.add_argument('--model', type=str, default='/data/tir/projects/tir7/user_data/srijithr/spin_outputs/iter_1_with_2e_data_2e_checkpoint_2')
    parser.add_argument('--data_frac', type=int, default=0)
    parser.add_argument('--frac_len', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='/home/srijithr/iterative-alignment/SPIN_implementation/data/gen_using_iter1_2e_data_2e_checkpoint')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--input_dir', type=str, default='UCLA-AGI/SPIN_iter0')
    parser.add_argument('--split', type=str, default='train')
    return parser.parse_args()

def prepare_prompts(prompts, tokenizer, batch_size, accelerator):
    """Prepare prompts for tokenization."""
    batches=[prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]  
    batches_tok=[]
    tokenizer.padding_side="left"     
    for prompt_batch in tqdm(batches):
        batches_tok.append(
            tokenizer(
                prompt_batch, 
                return_tensors="pt", 
                padding='longest', 
                truncation=False, 
                pad_to_multiple_of=8,
                add_special_tokens=False).to(accelerator.device) 
            )
    tokenizer.padding_side="right"
    return batches_tok

def main():
    args = parse_arguments()
    model_path = args.model
    data_frac = args.data_frac
    batch_size = args.batch_size
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # load a base model and tokenizer : OLD CODE
    model = AutoModelForCausalLM.from_pretrained(
        model_path,    
        device_map={"": accelerator.process_index},
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)   
    tokenizer.pad_token = tokenizer.eos_token

    # load data
    data = load_dataset(args.input_dir, split=args.split)
    data = data.shuffle(seed=42)
    if args.frac_len > 0:
        sub_len = args.frac_len 
        if sub_len*(data_frac+1) > len(data):
            data = data[sub_len*data_frac:]['real']
        else:
            data = data[sub_len*data_frac:sub_len*(data_frac+1)]['real']
    else:
        data = data[:]['real'] # ofcourse , we use ony real data

    prompts_all = ["### Instruction: " + data[idx][0]['content'] + "\n\n### Response: " for idx in range(len(data))] # Inst + Q
    prompts_old = [data[idx][0]['content'] for idx in range(len(data))] # Q 
    corrects_all = [data[idx][1]['content'] for idx in range(len(data))] # A

    # sync GPUs and start the timer
    accelerator.wait_for_everyone()    
    start=time.time()

    # divide the prompt list onto the available GPUs 
    print(f"Len of prompts_all before split bw process: {len(prompts_all)}")
    with accelerator.split_between_processes(prompts_all) as prompts:
        print(f"Len of prompts after split between process: {len(prompts)}")
        results = []
        prompt_batches=prepare_prompts(prompts, tokenizer, batch_size=args.batch_size, accelerator=accelerator)

        for prompts_tokenized in tqdm(prompt_batches, desc=f"GPU: {accelerator.process_index}"):
            # set max_new_tokens smaller for faster inference
            outputs_tokenized=model.generate(**prompts_tokenized, max_new_tokens=256, pad_token_id=tokenizer.eos_token_id)

            # remove prompt from gen. tokens
            outputs_tokenized=[ tok_out[len(tok_in):] 
                for tok_in, tok_out in zip(prompts_tokenized["input_ids"], outputs_tokenized) ] 
            # decode gen. tokens 
            outputs=tokenizer.batch_decode(outputs_tokenized)
            results.extend(outputs)

    # collect results from all the GPUs and remove paddings
    results_gathered=gather_object(results)
    print(f"results_gathered <MARKER>")
    results = [r.replace("</s>","").lstrip() for r in results_gathered]

    print(f"Writing only in main process <MARKER>")
    if accelerator.is_local_main_process:
        timediff=time.time()-start
        print(f"time elapsed: {timediff}")

        # collecting data
        for idx in tqdm(range(len(corrects_all))):
            d = {"real": [{"role": "user", "content": prompts_old[idx]}, {"role": "assistant", "content": corrects_all[idx]}], "generated": [{"role": "user", "content": prompts_old[idx]}, {"role": "assistant", "content": results[idx]}]}
            if args.split == 'test':
                filename = f"{args.output_dir}/loser_{data_frac}_test.jsonl"
            else:
                filename = f"{args.output_dir}/loser_{data_frac}.jsonl"
            with open(filename, 'a') as f:
                json.dump(d, f)
                f.write('\n')


if __name__ == "__main__":
    main()