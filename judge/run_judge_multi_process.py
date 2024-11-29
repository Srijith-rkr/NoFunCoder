import os
import json
import tempfile
from tqdm import tqdm
from multiprocessing import Pool
from code_runner import run_code_with_tests
import tempfile

TEST_PATH = 'data/codenet/public_test_cases'
DPO_OUTPUTS = 'data/dpo/judge_output'
INPUT_FILE = 'data/dpo/dpo_outputs/edit_deepseek_instruct_nrowsNone_tokens1024_temp0.4_fewshotex0_samples1_2024-11-10_08:21:27.jsonl'

# local 
# TEST_PATH = 'judge/data/codenet/public_test_cases'
# DPO_OUTPUTS = 'judge/data/dpo/judge_output'
# INPUT_FILE = 'judge/data/dpo/dpo_outputs/edit_deepseek_instruct_nrowsNone_tokens1024_temp0.4_fewshotex0_samples1_2024-11-10_08:21:27.jsonl'


def run_judge(code, problem_id, test_cases_path, num_runs, number_of_tests=None):
    """Submit code to judge server and get results"""


    problem_input_folder = os.path.join(test_cases_path, problem_id)
    
    ###### Sanity checks ######
    if not os.path.exists(problem_input_folder):
        print(f"Does not exist {problem_input_folder}")
        return None 
    
    data = os.listdir(problem_input_folder)
    file_count = len(data)
    if file_count < 2: # 1 test case has one input and output (2 files min). If < 2 then 1 test case not present
        print(f"Not enough test files for {problem_id}")
        return None

    #############################

    # Get input files
    input_files = sorted([f for f in os.listdir(problem_input_folder) 
                         if f.startswith('input')])
    
    if number_of_tests:
        input_files = input_files[:number_of_tests]

    passed = {}
    errors = {}
    runtimes = {}
    memory = {}

    with tempfile.TemporaryDirectory() as temp_dir:
        # Write the code to a temporary file
        code_file = os.path.join(temp_dir, 'submission.py')
        with open(code_file, 'w') as f:
            f.write(code)

        # Run tests
        for input_file in input_files:
            test_id = input_file.split('.')[1]
            input_path = os.path.join(problem_input_folder, input_file)
            output_path = os.path.join(problem_input_folder, f'output.{test_id}.txt')
        
            data = {
                'code': code,
                'input': open(input_path, 'r').read(),
                'output': open(output_path, 'r').read(),
                'num_runs': num_runs
            }

            # Run the code with test cases
            result = run_code_with_tests(code_file, data['input'], data['output'], num_runs)

            passed[test_id] = result['passed']
            errors[test_id] = result['errors']
            runtimes[test_id] = result['runtimes']
            memory[test_id] = result['memory']

    return {
        'passed': passed,
        'errors': errors,
        'runtimes': runtimes,
        'memory': memory,
        'passed_all_test': all(passed.values())
    }

def process_line(args):
    """Process a single line from the input file."""
    i, line, test_path, num_runs, number_of_tests, output_file_path = args
    data = json.loads(line)
    judge_results = run_judge(data['generated_codes'][0], data['problem_id'], test_path, num_runs, number_of_tests)
    if not judge_results:
        print(f"Judge results not found for {data['problem_id']}")
        return None
    judge_results['id'] = i
    # Append result to output file
    with open(output_file_path, 'a') as out_file:
        json.dump(judge_results, out_file)
        out_file.write('\n')  # Ensure newline for JSONL format
    return i, judge_results

def main():
    # Read input lines
    with open(INPUT_FILE, 'r') as f:
        lines = f.readlines()

    # Prepare output file path
    output_file_path = os.path.join(DPO_OUTPUTS, f"judge_{os.path.basename(INPUT_FILE)[:-6]}.jsonl")
    os.makedirs(DPO_OUTPUTS, exist_ok=True)  # Ensure the output directory exists

    # Set up multiprocessing pool
    args = [
        (i, line, TEST_PATH, 1, 5, output_file_path) for i, line in enumerate(lines)
    ]
    results = {}
    with Pool(processes=os.cpu_count()) as pool:
        it = tqdm(pool.imap_unordered(process_line, args), total=len(lines))
        for result in it:
            if result:
                i, judge_results = result
                results[i] = judge_results

    # Calculate pass rate
    all_passed = sum(1 for result in results.values() if result.get('passed_all_test'))
    print(f'*****Pass Rate: {all_passed}/{len(results)}*****')

if __name__ == '__main__':
    main()