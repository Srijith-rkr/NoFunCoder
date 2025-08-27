# NoFunCoder
**Enhancing code optimization for non-functional requirements while maintaining correctness**

This repository contains the source code for implementing **preference learning** and **profiler-based inference** for the NoFunCoder project, built on top of **ECCO**.

## Dashboard of Inference and Evaluation Outputs (Reimplemented Baselines and Our Method)

The outputs from our method are located in the `outputs_A4/` directory. A typical layout is:

```bash
./outputs_A4
├── calculate_scores.py
├── profiler
│   ├── inference_outputs
│   │   ├── No_Finetune_edit_codellama_13b_instruct_nrowsNone_tokens1024_temp0.0_fewshotex0_samples1_profiled_keep_only_profiled_2024-12-08_22:39:14.jsonl
│   │   ├── No_Finetune_edit_codellama_13b_instruct_nrowsNone_tokens1024_temp0.0_fewshotex2_samples1_profiled_keep_only_profiled_2024-12-08_22:22:03.jsonl
│   │   └── No_Finetune_self-refine_codellama_13b_instruct_nrowsNone_tokens1024_temp0.0_fewshotex0_samples1_profiled_keep_only_profiled_2024-12-09_01:34:27.jsonl
│   └── judge_outputs
│       ├── generated_codes_1_No_Finetune_self-refine_codellama_13b_instruct_nrowsNone_tokens1024_temp0.0_fewshotex0_samples1_profiled_keep_only_profiled_2024-12-09_01:34:27.jsonl
│       ├── generated_codes_No_Finetune_edit_codellama_13b_instruct_nrowsNone_tokens1024_temp0.0_fewshotex0_samples1_profiled_keep_only_profiled_2024-12-08_22:39:14.jsonl
│       └── generated_codes_No_Finetune_edit_codellama_13b_instruct_nrowsNone_tokens1024_temp0.0_fewshotex2_samples1_profiled_keep_only_profiled_2024-12-08_22:22:03.jsonl
├── DPO_COT_do_not_use
│   ├── inference_outputs
│   │   ├── deepseek_base_1e-3_YES_cot_only_failed_samples_3_epoch_edit_deepseek_instruct_nrowsNone_tokens1024_temp0.0_fewshotex2_samples1_2024-12-08_14:45:05.jsonl
│   │   └── deepseek_base_1e-3_YES_cot_only_failed_samples_checkpoint_300_edit_deepseek_instruct_nrowsNone_tokens1024_temp0.0_fewshotex2_samples1_2024-12-08_14:45:09.jsonl
│   └── judge_outputs
│       └── generated_codes_deepseek_base_1e-3_YES_cot_only_failed_samples_checkpoint_300_edit_deepseek_instruct_nrowsNone_tokens1024_temp0.0_fewshotex2_samples1_2024-12-08_14:45:09.jsonl
├── DPO_NO_FILTERING
│   ├── inference_outputs
│   │   ├── deepseek_base_1e-4_NO_cot_all_samples_100_ckpt_edit_deepseek_instruct_nrowsNone_tokens1024_temp0.0_fewshotex0_samples1_2024-12-04_02:17:27.jsonl
│   │   ├── deepseek_base_1e-4_NO_cot_all_samples_100_ckpt_edit_deepseek_instruct_nrowsNone_tokens1024_temp0.0_fewshotex2_samples1_2024-12-04_00:26:29.jsonl
│   │   └── deepseek_base_1e-4_NO_cot_all_samples_100_ckpt_self-refine_deepseek_instruct_nrowsNone_tokens1024_temp0.0_fewshotex0_samples1_2024-12-04_02:36:15.jsonl
│   └── judge_outputs
│       ├── generated_codes_1_deepseek_base_1e-4_NO_cot_all_samples_100_ckpt_self-refine_deepseek_instruct_nrowsNone_tokens1024_temp0.0_fewshotex0_samples1_2024-12-04_02:36:15.jsonl
│       ├── generated_codes_deepseek_base_1e-4_NO_cot_all_samples_100_ckpt_edit_deepseek_instruct_nrowsNone_tokens1024_temp0.0_fewshotex0_samples1_2024-12-04_02:17:27.jsonl
│       └── generated_codes_deepseek_base_1e-4_NO_cot_all_samples_100_ckpt_edit_deepseek_instruct_nrowsNone_tokens1024_temp0.0_fewshotex2_samples1_2024-12-04_00:26:29.jsonl
├── DPO_PASS1_FILTERING
│   ├── inference_outputs
│   │   ├── deepseek_base_1e-4_NO_cot_only_failed_samples_100_ckpt_edit_deepseek_instruct_nrowsNone_tokens1024_temp0.0_fewshotex0_samples1_2024-12-04_02:12:24.jsonl
│   │   ├── deepseek_base_1e-4_NO_cot_only_failed_samples_100_ckpt_edit_deepseek_instruct_nrowsNone_tokens1024_temp0.0_fewshotex2_samples1_2024-12-03_23:56:32.jsonl
│   │   └── deepseek_base_1e-4_NO_cot_only_failed_samples_100_ckpt_self-refine_deepseek_instruct_nrowsNone_tokens1024_temp0.0_fewshotex0_samples1_2024-12-04_02:34:40.jsonl
│   └── judge_outputs
│       ├── generated_codes_1_deepseek_base_1e-4_NO_cot_only_failed_samples_100_ckpt_self-refine_deepseek_instruct_nrowsNone_tokens1024_temp0.0_fewshotex0_samples1_2024-12-04_02:34:40.jsonl
│       ├── generated_codes_deepseek_base_1e-4_NO_cot_only_failed_samples_100_ckpt_edit_deepseek_instruct_nrowsNone_tokens1024_temp0.0_fewshotex0_samples1_2024-12-04_02:12:24.jsonl
│       └── generated_codes_deepseek_base_1e-4_NO_cot_only_failed_samples_100_ckpt_edit_deepseek_instruct_nrowsNone_tokens1024_temp0.0_fewshotex2_samples1_2024-12-03_23:56:32.jsonl
├── DPO_SPEED_AND_PASS1_FILTERING
│   ├── inference_outputs
│   │   ├── deepseek_base_1e-4_NO_cot_speed_incorrect_samples_10_epoch_edit_deepseek_instruct_nrowsNone_tokens1024_temp0.0_fewshotex0_samples1_2024-12-08_00:33:24.jsonl
│   │   ├── deepseek_base_1e-4_NO_cot_speed_incorrect_samples_10_epoch_edit_deepseek_instruct_nrowsNone_tokens1024_temp0.0_fewshotex2_samples1_2024-12-07_22:43:54.jsonl
│   │   └── deepseek_base_1e-4_NO_cot_speed_incorrect_samples_10_epoch_self-refine_deepseek_instruct_nrowsNone_tokens1024_temp0.0_fewshotex0_samples1_2024-12-07_23:21:55.jsonl
│   └── judge_outputs
│       ├── generated_codes_1_deepseek_base_1e-4_NO_cot_speed_incorrect_samples_10_epoch_self-refine_deepseek_instruct_nrowsNone_tokens1024_temp0.0_fewshotex0_samples1_2024-12-07_23:21:55.jsonl
│       ├── generated_codes_deepseek_base_1e-4_NO_cot_speed_incorrect_samples_10_epoch_edit_deepseek_instruct_nrowsNone_tokens1024_temp0.0_fewshotex0_samples1_2024-12-08_00:33:24.jsonl
│       └── generated_codes_deepseek_base_1e-4_NO_cot_speed_incorrect_samples_10_epoch_edit_deepseek_instruct_nrowsNone_tokens1024_temp0.0_fewshotex2_samples1_2024-12-07_22:43:54.jsonl
├── DPO_SPEED_FILTERING
│   ├── inference_outputs
│   │   ├── deepseek_base_1e-4_NO_cot_only_speed_samples_10_epoch_edit_deepseek_instruct_nrowsNone_tokens1024_temp0.0_fewshotex0_samples1_2024-12-08_00:32:39.jsonl
│   │   ├── deepseek_base_1e-4_NO_cot_only_speed_samples_10_epoch_edit_deepseek_instruct_nrowsNone_tokens1024_temp0.0_fewshotex2_samples1_2024-12-07_22:40:47.jsonl
│   │   └── deepseek_base_1e-4_NO_cot_only_speed_samples_10_epoch_self-refine_deepseek_instruct_nrowsNone_tokens1024_temp0.0_fewshotex0_samples1_2024-12-07_23:17:26.jsonl
│   └── judge_outputs
│       ├── generated_codes_1_deepseek_base_1e-4_NO_cot_only_speed_samples_10_epoch_self-refine_deepseek_instruct_nrowsNone_tokens1024_temp0.0_fewshotex0_samples1_2024-12-07_231726.jsonl
│       ├── generated_codes_deepseek_base_1e-4_NO_cot_only_speed_samples_10_epoch_edit_deepseek_instruct_nrowsNone_tokens1024_temp0.0_fewshotex0_samples1_2024-12-08_003239.jsonl
│       └── generated_codes_deepseek_base_1e-4_NO_cot_only_speed_samples_10_epoch_edit_deepseek_instruct_nrowsNone_tokens1024_temp0.0_fewshotex2_samples1_2024-12-07_224047.jsonl

```


The outputs from our baseline repoduction can be found in the outputs_A3 directory with the structure shown below. We have 2 subdirectories for the 2 models we use to reproduce the baselines. Each of these subdirectories are further divided based on inference and evaluation outputs and then by task type at the next level. Outputs corresponding to codellama can be found in `dashboard/codellama_13b` and those corresponding to deepseek can be found in `dashboard/deepseek`. Within the directories for each of these models, the outputs are further divided based on the task type and the setting of the task. Also, `inference_results` contains the LLM outputs, whereas `judge_results` contains the judgement/verdict on the code generated by the LLMs.

```bash
./outputs_A3
├── calculate_scores.py
├── codellama_13b
│   ├── inference_outputs
│   │   ├── history_based
│   │   │   ├── edit_codellama_13b_instruct_nrowsNone_tokens1024_temp0.4_fewshotex0_samples1_some_date.jsonl
│   │   │   ├── edit_codellama_13b_instruct_nrowsNone_tokens1024_temp0.4_fewshotex2_samples1_some_date.jsonl
│   │   │   ├── exec-refine_codellama_13b_nrowsNone_tokens1024_temp0.4_samples1_numrefine1_2024-11-12_14:12:42.jsonl
│   │   │   ├── nl-exec-refine_codellama_13b_nrowsNone_tokens1024_temp0.4_samples1_numrefine1_2024-11-12_14:53:42.jsonl
│   │   │   └── self-refine_codellama_13b_nrowsNone_tokens1024_temp0.4_samples1_numrefine1_some_date.jsonl
│   │   └── nl-based
│   │       ├── generated_codes_1_nl2code-exec-refine_codellama_13b_nrowsNone_tokens1024_temp0.4_samples1_numrefine1_2024-11-12_15_34_30.jsonl
│   │       ├── generated_codes_1_nl2code-nl-exec-refine_codellama_13b_nrowsNone_tokens1024_temp0.4_samples1_numrefine1_2024-11-12_17_03_12.jsonl
│   │       ├── generated_codes_1_nl2code-self-refine_codellama_13b_nrowsNone_tokens1024_temp0.4_samples1_numrefine1_some_date.jsonl
│   │       └── generated_codes_nl2code_codellama_13b_instruct_nrowsNone_tokens1024_temp0.4_fewshotex0_samples1_some_date.jsonl
│   └── judge_outputs
│       ├── history_based
│       │   ├── generated_codes_1_exec-refine_codellama_13b_nrowsNone_tokens1024_temp0.4_samples1_numrefine1_2024-11-12_14:12:42.jsonl
│       │   ├── generated_codes_1_nl-exec-refine_codellama_13b_nrowsNone_tokens1024_temp0.4_samples1_numrefine1_2024-11-12_14:53:42.jsonl
│       │   ├── generated_codes_1_self-refine_codellama_13b_nrowsNone_tokens1024_temp0.4_samples1_numrefine1_some_date.jsonl
│       │   ├── generated_codes_edit_codellama_13b_instruct_nrowsNone_tokens1024_temp0.4_fewshotex0_samples1_some_date.jsonl
│       │   └── generated_codes_edit_codellama_13b_instruct_nrowsNone_tokens1024_temp0.4_fewshotex2_samples1_some_date.jsonl
│       └── nl-based
│           ├── generated_codes_1_nl2code-exec-refine_codellama_13b_nrowsNone_tokens1024_temp0.4_samples1_numrefine1_2024-11-12_15_34_30.jsonl
│           ├── generated_codes_1_nl2code-nl-exec-refine_codellama_13b_nrowsNone_tokens1024_temp0.4_samples1_numrefine1_2024-11-12_17_03_12.jsonl
│           ├── generated_codes_1_nl2code-self-refine_codellama_13b_nrowsNone_tokens1024_temp0.4_samples1_numrefine1_some_date.jsonl
│           └── generated_codes_nl2code_codellama_13b_instruct_nrowsNone_tokens1024_temp0.4_fewshotex0_samples1_some_date.jsonl
└── deepseek
    ├── inference_outputs
    │   ├── history_based
    │   │   ├── edit_deepseek_instruct_nrowsNone_tokens1024_temp0.4_fewshotex0_samples1_2024-11-10_08:21:27.jsonl
    │   │   ├── edit_deepseek_instruct_nrowsNone_tokens1024_temp0.4_fewshotex2_samples1_2024-11-10_08:26:33.jsonl
    │   │   ├── exec-refine_deepseek_nrowsNone_tokens1024_temp0.4_samples1_numrefine1_2024-11-10_08:42:21.jsonl
    │   │   ├── nl-exec-refine_deepseek_nrowsNone_tokens1024_temp0.4_samples1_numrefine1_2024-11-11_01:24:24.jsonl
    │   │   └── self-refine_deepseek_nrowsNone_tokens1024_temp0.4_samples1_numrefine1_2024-11-10_08:31:40.jsonl
    │   ├── nl2
    │   │   ├── nl2code_deepseek_instruct_nrowsNone_tokens1024_temp0.4_fewshotex0_samples1_2024-11-10_06:54:38.jsonl
    │   │   ├── nl2code-exec-refine_deepseek_nrowsNone_tokens1024_temp0.4_samples1_numrefine1_2024-11-11_07:18:13.jsonl
    │   │   ├── nl2code-nl-exec-refine_deepseek_nrowsNone_tokens1024_temp0.4_samples1_numrefine1_2024-11-11_07:12:31.jsonl
    │   │   └── nl2code-self-refine_deepseek_nrowsNone_tokens1024_temp0.4_samples1_numrefine1_2024-11-10_06:57:15.jsonl
    │   └── sft
    │       ├── DO_NOT_USE_EXEC-SFT-exec-refine_deepseek_nrowsNone_tokens1024_temp0.4_samples1_numrefine1_2024-11-12_11:45:45.jsonl
    │       ├── DO_NOT_USE_Trajectory_SFT_FS_edit_deepseek_instruct_nrowsNone_tokens1024_temp0.4_fewshotex2_samples1_2024-11-12_11:09:13.jsonl
    │       ├── EXEC-SFT-edit_deepseek_instruct_nrowsNone_tokens1024_temp0.4_fewshotex0_samples1_2024-11-12_16:04:34.jsonl
    │       ├── SFT_FS_edit_deepseek_instruct_nrowsNone_tokens1024_temp0.4_fewshotex2_samples1_2024-11-11_15:31:38.jsonl
    │       └── TRAJECTORY-SFT_edit_deepseek_instruct_nrowsNone_tokens1024_temp0.4_fewshotex0_samples1_2024-11-12_17:21:34.jsonl
    └── judge_outputs
        ├── history_based
        │   ├── generated_codes_1_exec-refine_deepseek_nrowsNone_tokens1024_temp0.4_samples1_numrefine1_2024-11-10_08:42:21.jsonl
        │   ├── generated_codes_1_nl-exec-refine_deepseek_nrowsNone_tokens1024_temp0.4_samples1_numrefine1_2024-11-11_01:24:24.jsonl
        │   ├── generated_codes_1_self-refine_deepseek_nrowsNone_tokens1024_temp0.4_samples1_numrefine1_2024-11-10_08:31:40.jsonl
        │   ├── generated_codes_edit_deepseek_instruct_nrowsNone_tokens1024_temp0.4_fewshotex0_samples1_2024-11-10_08:21:27.jsonl
        │   └── generated_codes_edit_deepseek_instruct_nrowsNone_tokens1024_temp0.4_fewshotex2_samples1_2024-11-10_08:26:33.jsonl
        ├── nl2
        │   ├── generated_codes_1_nl2code-exec-refine_deepseek_nrowsNone_tokens1024_temp0.4_samples1_numrefine1_2024-11-11_07:18:13.jsonl
        │   ├── generated_codes_1_nl2code-nl-exec-refine_deepseek_nrowsNone_tokens1024_temp0.4_samples1_numrefine1_2024-11-11_07:12:31.jsonl
        │   ├── generated_codes_1_nl2code-self-refine_deepseek_nrowsNone_tokens1024_temp0.4_samples1_numrefine1_2024-11-10_06:57:15.jsonl
        │   └── generated_codes_nl2code_deepseek_instruct_nrowsNone_tokens1024_temp0.4_fewshotex0_samples1_2024-11-10_06:54:38.jsonl
        └── sft
            ├── generated_codes_EXEC-SFT-edit_deepseek_instruct_nrowsNone_tokens1024_temp0.4_fewshotex0_samples1_2024-11-12_16:04:34.jsonl
            ├── generated_codes_SFT_FS_edit_deepseek_instruct_nrowsNone_tokens1024_temp0.4_fewshotex2_samples1_2024-11-11_15:31:38.jsonl
            └── generated_codes_TRAJECTORY-SFT_edit_instruct_nrowsNone_tokens1024_temp0.4_fewshotex0_samples1_2024-11-12_17:21:34.jsonl
```



## Code structure

1. `evaluation` consists of scripts to run evaluation of model generated code on the Judge0 environment server hosted on AWS. Please see instructions to setup the evaluation server.

   - `edit_eval.py` is the script for evaluating code generated on the metrics for the history-based editing setting
   - `generate_eval.py` is the script for evaluating code generated on the metrics for the NL-instructed generation setting
2. `experiments` consists of the scripts to run modelling experiment.

   - `model_classes.py` consists of the Inference Engine Classes for each model that is benchmarked.
   - `inference.py` is the entrypoint for running the experiments
   - `prompt_formats.py` and `utils.py` cotains utilities for prompt building and execution feedback formatting
3. `judge` conatins code for running a local instance of a judge and the profiler code in a docker environment.
- `run_judge_multi_process_multi_thread.py` is the code used to evaluate pass rates and relative speedup in between DPO runs as the AWS judge would be too slow for the iterations.
- `run_profiler.py` is the code used to run the profiler on the ECCO dataset code.

### Starting up the evaluation setup
We followed the evaluation setup with the guide in the [evaluation README](./evaluation/README.md) as used AWS credits allotted to us as a part of the course.

## Dataset

The original paper's dataset is available on Huggingface at: [CodeEff/ECCO](https://huggingface.co/datasets/CodeEff/ECCO).

It consists of 2 subsets `edit` and `generate` each with 3 splits (`train`, `val` and `test`).

### Loading the dataset

```python
dataset = load_dataset('CodeEff/ECCO', 'edit') # For history-based editing setting
dataset = load_dataset('CodeEff/ECCO', 'generate') # For nl-instructed generation setting
```

The profiled code dataset is saved in the `data/annotated_dataset` directory.

## Instructions to reproduce our results

### Environment setup

```bash
conda env create -f environment.yml
conda activate ecco
```

### Download the test cases

```sh
mkdir data && cd data
wget https://huggingface.co/datasets/CodeEff/resolve/main/test_cases.zip
unzip test_cases.zip
```

### Setup a judge on AWS

   Guide can be found in the evaluation README

### Running inference

NL-Instructed tasks have 'eval-mode's that begin with nl2code.

To used the profiled dataser, use the flags `--use_profiler` and `--keep_only_profiled_data`.

#### Pre-Refine

* DeepSeek

```sh
   python experiments/inference.py --eval_mode <nl2code|edit> --few_shot_example <Number of in-context examples> --judge_url http://<PUBLIC_URL>:2358 --model deepseek
```

* CodeLLaMa-13b

```sh
   python experiments/inference.py --eval_mode <nl2code|edit> --few_shot_example <Number of in-context examples> --judge_url http://<PUBLIC_URL>:2358 --model codellama_13b --num_gpus 2
```

#### Self-Refine (with Natural Language Feedback)

* DeepSeek

```sh
   python experiments/inference.py --eval_mode <nl2code-self-refine|self-refine> --judge_url http://<PUBLIC_URL>:2358 --model deepseek
```

* CodeLLaMa-13b

```sh
   python experiments/inference.py --eval_mode <nl2code-self-refine|self-refine> --judge_url http://<PUBLIC_URL>:2358 --model codellama_13b --num_gpus 2
```

#### Refine with Interpreter Feedback

* DeepSeek

```sh
   python experiments/inference.py --eval_mode <nl2code-exec-refine|exec-refine> --judge_url http://<PUBLIC_URL>:2358 --model deepseek
```

* CodeLLaMa-13b

```sh
   python experiments/inference.py --eval_mode <nl2code-exec-refine|exec-refine>  --judge_url http://<PUBLIC_URL>:2358 --model codellama_13b --num_gpus 2
```

#### Refine with Interpreter Feedback and Natural Language

* DeepSeek

```sh
   python experiments/inference.py --eval_mode <nl2code-nl-exec-refine|nl-exec-refine> --judge_url http://<PUBLIC_URL>:2358 --model deepseek
```

* CodeLLaMa-13b

```sh
   python experiments/inference.py --eval_mode <nl2code-nl-exec-refine|nl-exec-refine> --judge_url http://<PUBLIC_URL>:2358 --model codellama_13b --num_gpus 2
```

### Running Evaluation

#### For NL-Instructed Tasks

```sh
   python evaluation/generate_eval.py --judge_url http://<PUBLIC_URL>:2358 --input_path <path_to_jsonl_generated_in_inference> --code_col_name <generated_codes[_<Last iteration for refinement tasks>]>
```

#### For History-based Editing

```sh
   python evaluation/edit_eval.py --judge_url http://<PUBLIC_URL>:2358 --input_path <path_to_jsonl_generated_in_inference> --code_col_name <generated_codes[_<Last iteration for refinement tasks>]>
```
