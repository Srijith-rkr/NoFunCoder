import os
import json
from code_runner import run_code_with_tests
import tempfile

TEST_PATH = 'data/codenet/public_test_cases'
DPO_OUTPUTS = 'data/dpo/judge_output'
INPUT_FILE = 'data/dpo/dpo_outputs/edit_deepseek_instruct_nrowsNone_tokens1024_temp0.4_fewshotex0_samples1_2024-11-10_08:21:27.jsonl'

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

        with tempfile.TemporaryDirectory() as temp_dir:
            # Write the code to a temporary file
            code_file = os.path.join(temp_dir, 'submission.py')
            with open(code_file, 'w') as f:
                f.write(code)

            # Run the code with test cases
            result = run_code_with_tests(code_file, data['input'], data['output'], num_runs)

        passed[test_id] = result['passed']
        errors[test_id] = result['errors']
        runtimes[test_id] = result['runtimes']
        memory[test_id] = result['memory']



    return {'passed': passed, 'errors': errors, 'runtimes': runtimes, 'memory': memory, 'passed_all_test':all(passed.values())}

if __name__ == '__main__':
    results = {}
    with open(INPUT_FILE, 'r') as f:
        for i,line in enumerate(f):
            print(f'Processing {i}th line')
            data = json.loads(line)
            #print(data)
            judge_results = run_judge(data['generated_codes'][0], data['problem_id'],TEST_PATH,1)
            if not judge_results:
                print(f'Judge results not found for {data["problem_id"]}')
            else:
                results[i] = judge_results
                judge_results['id'] = i
                with open(f"{DPO_OUTPUTS}/judge_{INPUT_FILE.split('/')[-1][:-6]}.jsonl", 'a') as out_file:
                    json.dump(judge_results,out_file)

    all_passed = 0
    for id, result in results.items():
        if result['passed_all_test']:
            all_passed += 1
    print(f'*****Pass Rate: {all_passed}/{len(results)}*****')
    





