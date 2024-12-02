#!/usr/bin/env python
# 
# Adapted from https://github.com/huggingface/alignment-handbook 
import logging
import sys
import os
import yaml 

import torch
import transformers
from transformers import AutoModelForCausalLM, set_seed, TrainingArguments
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState

from accelerate import Accelerator
from alignment import (
    DataArguments,
    SPINConfig,
    H4ArgumentParser,
    ModelArguments,
    get_datasets,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
    is_adapter_model,
)
from datasets import load_dataset, concatenate_datasets
from peft import PeftConfig, PeftModel
from alignment import SPINTrainer
from torch.utils.data import Subset
import re

def create_unique_dir_name(base_dir):
    # If base directory does not exist, return it
    if not os.path.exists(base_dir):
        return base_dir
    else:
        # Find the next available directory name with a suffix
        counter = 2
        new_dir = f"{base_dir}_{counter}"
        while os.path.exists(new_dir):
            counter += 1
            new_dir = f"{base_dir}-{counter}"
        return new_dir
    
def apply_chat_template(
    example, tokenizer, task, assistant_prefix="<|assistant|>\n"
):
    def _strip_prefix(s, pattern):
        # Use re.escape to escape any special characters in the pattern
        return re.sub(f"^{re.escape(pattern)}", "", s)

    if all(k in example.keys() for k in ("real", "generated")):
        # Compared to reward modeling, we filter out the prompt, so the text is everything after the last assistant token
        prompt_messages = [[msg for msg in example["real"] if msg["role"] == "user"][0]]
        # Insert system message
        if example["real"][0]["role"] != "system":
            prompt_messages.insert(0, {"role": "system", "content": ""})
        else:
            prompt_messages.insert(0, example["real"][0])

        real_messages = example["real"][1:]
        generated_messages = example["generated"][1:]
        example["text_real"] = tokenizer.apply_chat_template(real_messages, tokenize=False)
        example["text_generated"] = tokenizer.apply_chat_template(generated_messages, tokenize=False)
        example["text_prompt"] = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        example["text_real"] = _strip_prefix(example["text_real"], assistant_prefix)
        example["text_generated"] = _strip_prefix(example["text_generated"], assistant_prefix)
    else:
        raise ValueError(
            f"Require `[real, generated]` keys but found {list(example.keys())}"
            )
    return example

logger = logging.getLogger(__name__)


def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, SPINConfig))
    model_args, data_args, training_args = parser.parse()

    #######
    # Setup
    #######
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed for reproducibility
    set_seed(training_args.seed)

    # Increase distributed timeout to 3h to enable push to Hub to complete
    accelerator = Accelerator()
    
    if accelerator.is_main_process:
            training_args.output_dir = os.path.join(f'{training_args.output_dir}' , f'{training_args.run_name}')
            training_args.output_dir = create_unique_dir_name(training_args.output_dir)
            os.makedirs(training_args.output_dir, exist_ok=True)


            to_save = {"model_args": model_args, "data_args": data_args, "training_args": training_args}
            for key, value in to_save.items():
                with open(os.path.join(training_args.output_dir, f"{key}_config.yaml"), 'w') as yaml_file:
                    yaml.dump(value, yaml_file)

    ###############
    # Load datasets
    ###############
    # if not data_args.use_local_dataset:
    #     raw_datasets = get_datasets(data_args, splits=data_args.dataset_splits)
    #     logger.info(
    #         f"Training on the following splits: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    #     )
    #     column_names = list(raw_datasets["train"].features)
    # else:   
    #     datafiles = [i for i in data_args.local_dataset_mixer]
    #     raw_datasets = load_dataset("json", data_files=datafiles)
    #     logger.info(f"Training on local dataset with {len(raw_datasets['train'])} samples")
    #     column_names = list(raw_datasets["train"].features)

    #####################################
    # Load tokenizer and process datasets
    #####################################
    data_args.truncation_side = "left"  # Truncate from left to ensure we don't lose labels in final turn
    tokenizer = get_tokenizer(model_args, data_args)

    #####################
    # Apply chat template
    #####################
    # raw_datasets = raw_datasets.map(
    #     apply_chat_template,
    #     fn_kwargs={"tokenizer": tokenizer, "task": "spin"},
    #     num_proc=data_args.preprocessing_num_workers,
    #     remove_columns=column_names,
    #     desc="Formatting comparisons with prompt template",
    # )

    # # Replace column names with what TRL needs, text_real -> real and text_generated -> generated
    # if not data_args.use_local_dataset:
    #     for split in ["train", "test"]:
    #         raw_datasets[split] = raw_datasets[split].rename_columns(
    #             {"text_prompt": "prompt", "text_real": "real", "text_generated": "generated"}
    #         )
    # else:
    # # We do this since I did not generate the test data for iterations
    #     raw_datasets['train'] = raw_datasets['train'].rename_columns(
    #         {"text_prompt": "prompt", "text_real": "real", "text_generated": "generated"})
        
    #     raw_datasets['test'] = raw_datasets['train'].select(range(0, 500))

    #######################################
    # ECCO data stuff here
    tokenizer.add_eos_token = True
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    if data_args.use_local_dataset:
        train_dataset  = load_dataset('json', data_files=data_args.local_dataset_mixer[0], split='train')

    else:
        dataset = load_dataset('EfficientCode/ECCO', 'edit')
        train_dataset = dataset['train']
        eval_dataset = dataset['val']





    def generate_and_tokenize_prompt(data_point, instruct=True, markdown=True, exec= False, local_dataset=data_args.use_local_dataset, use_cot = data_args.use_cot):
        wrap_string = "```" if not markdown else "```python"

        if exec:
            full_prompt =f"""Optimize the python program below to be functionally equivalent but run faster and use less memory. Wrap the optimized code in a block of 3 backticks (```).\n
## Program:
{data_point["input"]}\n
## Program's Execution results:\n
{data_point['input_exec_feedback']}\n
## Optimized (Runtime and Space) version of Program above:\n
"""     
            response = f"{wrap_string}\n{data_point['target']}\n```\n\n## Optimized version's execution results:\n{data_point['output_exec_feedback']}"
            
        else:
            full_prompt =f"""Optimize the python program below to be functionally equivalent but run faster and use less memory. Wrap the optimized code in a block of 3 backticks (```).\n
## Program:
{data_point["input"]}\n
## Optimized (Runtime and Space) version of Program above:\n### Response:\n
"""
            if local_dataset:
                if use_cot:
                    generated = data_point['full_generations'][0]
                else :
                    generated = f"{wrap_string}\n{data_point['generated_codes'][0]}\n```"
            else: 
                generated = f"{wrap_string}\n{data_point['target']}\n```" # does not make sense to do this - have it for testing purposes

            real =  f"{wrap_string}\n{data_point['target']}\n```"
            

        # if not instruct:
        #     full_seq = full_prompt + response
        
        # else: # If chat template to be used 
        #     messages = [
        #         {'role': 'user', 'content': full_prompt},
        #         {'role': 'assistant', 'content': response}
        #     ]
            
            # if 'codellama' in model_args.model_name_or_path:
            #     full_seq = '[INST] ' + full_prompt[:-2] + '[/INST]\n' + response
            # else: # code llama does not have tokenizer.apply_chat_template implemented in huggingface 
            #     full_seq = tokenizer.apply_chat_template(messages, tokenize=False)
        
        use = False
        if data_point['judge_results']['passed_all_test'] == False: # We only want to use the failed samples as the loser samples in DPO
            use = True

        data_point['use'] = use
        data_point['prompt'] = tokenizer.apply_chat_template([{'role': 'user', 'content': full_prompt}], tokenize=False)
        data_point['real'] =  real 
        data_point['generated'] = generated 


        # data_point['template'] = full_seq
        return data_point

    orig_columns = [i for i in train_dataset.features if i != 'prompt']
    train_dataset = train_dataset.map(generate_and_tokenize_prompt, remove_columns=orig_columns) 
    if data_args.use_only_failed_samples:
        train_dataset = train_dataset.filter(lambda example: example['use'] == True)
    val_dataset = train_dataset.select(range(0, 500))

    #########################################

    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        use_flash_attention_2=model_args.use_flash_attention_2,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    model = model_args.model_name_or_path

    ref_model = model
    ref_model_kwargs = model_kwargs

    if model_args.use_peft is True:
        ref_model = None
        ref_model_kwargs = None
        
    class MyCallBacks(TrainerCallback):
        def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
            """
            Event called after logging the last logs.
            """
            # print("Beep Boop!!")
            pass
        def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
            """
            Event called after logging the last logs.
            """
            # print("Beep Boop Again!!")
            pass
        
        

    #########################
    # Instantiate spin trainer
    #########################
    spin_trainer = SPINTrainer(
        model,
        ref_model,
        model_init_kwargs=model_kwargs,
        ref_model_init_kwargs=ref_model_kwargs,
        args=training_args,
        beta=training_args.beta,
        # train_dataset=raw_datasets["train"],
        # eval_dataset=raw_datasets["test"],
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        max_length=training_args.max_length,
        max_prompt_length=training_args.max_prompt_length,
        peft_config=get_peft_config(model_args),
        callbacks=[MyCallBacks()]
    )

    ###############
    # Training loop
    ###############
    train_result = spin_trainer.train()
    metrics = train_result.metrics
    max_train_samples = (
        data_args.max_train_samples if data_args.max_train_samples is not None else len(raw_datasets["train"])
    )
    metrics["train_samples"] = min(max_train_samples, len(raw_datasets["train"]))
    spin_trainer.log_metrics("train", metrics)
    spin_trainer.save_metrics("train", metrics)
    spin_trainer.save_state()

    logger.info("*** Training complete ***")

    ##################################
    # Save model and create model card
    ##################################
    spin_trainer.save_model(training_args.output_dir)
    # Save everything else on main process
    if accelerator.is_main_process:
        kwargs = {
            "finetuned_from": model_args.model_name_or_path,
            "dataset": list(data_args.dataset_mixer.keys()),
            "dataset_tags": list(data_args.dataset_mixer.keys()),
            "tags": ["alignment-handbook"],
        }
        spin_trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        spin_trainer.model.config.use_cache = True
        spin_trainer.model.config.save_pretrained(training_args.output_dir)

    # Ensure we don't timeout on model save / push to Hub
    logger.info("*** Waiting for all processes to finish ***")
    accelerator.wait_for_everyone()

    logger.info("*** Run complete! ***")


if __name__ == "__main__":
    main()
