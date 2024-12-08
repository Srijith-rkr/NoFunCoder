import os
import tempfile
from line_profiler import LineProfiler
import importlib.util
import sys
import numpy as np
from datasets import load_dataset
import logging
from tqdm import tqdm
from timeout_decorator import timeout, TimeoutError
TEST_PATH = 'data/codenet/public_test_cases'
OUTPUT_DIR = 'data/annotated_dataset'
NUM_TESTS = 5

def load_module(filename):
    try:
        if not os.path.exists(filename):
            print(f"File does not exist: {filename}")
            return None

        spec = importlib.util.spec_from_file_location("module", filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        print(f"Error loading module from {filename}: {e}")
        return None

def profile_module(code_file, input_file):
    return_stats = {}

    try:
        # Replace stdin with the input file
        sys.stdin = open(input_file, 'r')

        module = load_module(code_file)
        
        # Create profiler
        profiler = LineProfiler()

        for name, obj in module.__dict__.items():
            if callable(obj):
                # Only profile functions defined in this module
                if hasattr(obj, '__code__') and obj.__code__.co_filename == code_file:
                    profiler.add_function(obj)

        if hasattr(module, 'execute_for_profiling'):
            # Wrap the profiler execution with a 10 second timeout
            @timeout(10)  # 10 seconds timeout
            def run_with_timeout():
                profiler.runctx('module.execute_for_profiling()', globals(), {'module': module})
            
            run_with_timeout()
            
    except Exception as e:
        print(f"Error in profile_module: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Restore original stdin
        sys.stdin.close()
    
    # Get profiler stats if execution succeeded
    try:
        stats = profiler.get_stats()
        for key, timings in stats.timings.items():
            for line_no, nhits, total_time in timings:
                return_stats[line_no] = {'nhits': nhits, 'total_time': total_time}
    except Exception as e:
        print(f"Error getting profiler stats: {str(e)}")
        
    return return_stats



def run_profiler(code, problem_id, test_cases_path, number_of_tests=None):
    """Submit code to judge server and get results"""

    if 'execute_for_profiling' in code or '__main__' in code or 'exit(' in code or 'quit(' in code:
        print(f'Did not annotate code*****************')
        return None

    stats = {}
    problem_input_folder = os.path.join(test_cases_path, problem_id)
    
    ###### Sanity checks ######
    if not os.path.exists(problem_input_folder):
        print(f"Does not exist {problem_input_folder}")
        return None 
    
    data = os.listdir(problem_input_folder)
    file_count = len(data)
    if file_count < 2: # 1 test case has one input and output (2 files min). If < 2 then 1 test case not present 
        print(f"Not enough test files for {problem_id}")

    #############################

    # Get input files
    input_files = sorted([f for f in os.listdir(problem_input_folder) 
                         if f.startswith('input')])
    
    if number_of_tests:
        input_files = input_files[:number_of_tests]

    with tempfile.TemporaryDirectory() as temp_dir:
        # Wrap the code in a function before saving
        wrapped_code = f"def execute_for_profiling():\n"
        for line in code.split('\n'):
            wrapped_code += f"    {line}\n"

        # Write the wrapped code to a temporary file
        code_file = os.path.join(temp_dir, 'submission.py')
        with open(code_file, 'w') as f:
            f.write(wrapped_code)

        # Run tests
        for input_file in input_files:
            input_path = os.path.join(problem_input_folder, input_file)
        
            # Run the code with test cases
            result = profile_module(code_file, input_path)
            for line_no, stat in result.items():
                if line_no not in stat:
                    stats[line_no] = {'nhits':[], 'total_time':[]}
                stats[line_no]['nhits'].append(stat['nhits'])
                stats[line_no]['total_time'].append(stat['total_time'])
    
    avg_stats = {}
    for line_no, stats in stats.items():
        avg_stats[line_no] = {'nhits':np.mean(stats['nhits']), 'total_time':np.mean(stats['total_time'])}

    annotated_code = str(code).split('\n')
    
    for line_no, avg_stat in avg_stats.items():
        # Adjust line number for wrapped code
        actual_line = line_no - 1 if 'execute_for_profiling' not in code else line_no
        annotated_code[actual_line-1] = annotated_code[actual_line-1] + f" # Hits:{avg_stat['nhits']}, Total execution time: {avg_stat['total_time'] / 1000} ms, Average time per hit: {(avg_stat['total_time'] / avg_stat['nhits']) / 1000 if avg_stat['nhits'] > 0 else '-' } ms \n"
    print(f'Annotated code for {problem_id}')
    return '\n'.join(annotated_code)

if __name__ == '__main__':

    # SAMPLE_INPUT = 'data/dpo/dpo_outputs/0_to12096_edit_deepseek_instruct_nrowsNone_tokens1024_temp0.4_fewshotex0_samples1_2024-11-28_19_26_43.jsonl'

    # with open(SAMPLE_INPUT, 'r') as f:
    #     for line in f:
    #         data = json.loads(line)
    #         #print(data)
    #         print(run_profiler(data['input'], data['problem_id'],'data/codenet/public_test_cases',1))
    #         break


    dataset = load_dataset('EfficientCode/ECCO', 'edit')
    test = dataset['test'].to_pandas()

    input_annotated = []
    target_annotated = []
    is_annotated = []
    error_rows = 0


    for index, row in tqdm(test.iterrows()):
        inp_is_annotated = False
        tar_is_annotated = False
        print(f"Processing row {index}")
        try:
            inp_ann = run_profiler(row['input'], row['problem_id'],TEST_PATH,NUM_TESTS)
            if inp_ann is None:
                raise Exception("Input not annotated")
            inp_is_annotated = True
        except Exception as e:
            print(f"(Input) Error in row {index} : {e}")
            error_rows += 1
            inp_ann = row['input']
        
        input_annotated.append(inp_ann)

        try:
            tar_ann = run_profiler(row['target'], row['problem_id'],TEST_PATH,NUM_TESTS)
            tar_is_annotated = True
        except Exception as e:
            print(f"Error in row {index} : {e}")
            error_rows += 1
            tar_ann = row['target']

        target_annotated.append(tar_ann)
        is_annotated.append(inp_is_annotated and tar_is_annotated)
    print(f"(Target) Error rows: {error_rows}")


    test_annotated = test.copy()
    test_annotated['input'] = input_annotated
    test_annotated['target'] = target_annotated
    test_annotated['codes_annotated'] = is_annotated


    test_annotated.to_csv(os.path.join(OUTPUT_DIR,'test_annotated.csv'), index=False)


    train = dataset['train'].to_pandas()[:150]

    input_annotated = []
    target_annotated = []
    is_annotated = []
    error_rows = 0


    for index, row in tqdm(train.iterrows()):
        inp_is_annotated = False
        tar_is_annotated = False
        print(f"Processing row {index}")
        try:
            inp_ann = run_profiler(row['input'], row['problem_id'],TEST_PATH,NUM_TESTS)
            if inp_ann is None:
                raise Exception("Input not annotated")
            inp_is_annotated = True
        except Exception as e:
            print(f"Error in row {index} : {e}")
            error_rows += 1
            inp_ann = row['input']
        
        input_annotated.append(inp_ann)

        try:
            tar_ann = run_profiler(row['target'], row['problem_id'],TEST_PATH,NUM_TESTS)
            tar_is_annotated = True
        except Exception as e:
            print(f"Error in row {index} : {e}")
            error_rows += 1
            tar_ann = row['target']

        target_annotated.append(tar_ann)
        is_annotated.append(inp_is_annotated and tar_is_annotated)
    print(f"Error rows: {error_rows}")


    train_annotated = train.copy()
    train_annotated['input'] = input_annotated
    train_annotated['target'] = target_annotated
    train_annotated['codes_annotated'] = is_annotated


    train_annotated.to_csv(os.path.join(OUTPUT_DIR,'train_annotated.csv'), index=False)

    

    #train = dataset['train'].to_pandas()

    # train['annotated_input'] = train.apply(lambda x: run_profiler(x['input'], x['problem_id'],TEST_PATH,NUM_TESTS), axis=1)
    # train['annotated_target'] = train.apply(lambda x: run_profiler(x['target'], x['problem_id'],TEST_PATH,NUM_TESTS), axis=1)

    # train.to_csv(os.path.join(OUTPUT_DIR,'train_annotated.csv'), index=False)

   