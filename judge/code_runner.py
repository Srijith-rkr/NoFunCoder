import os
import resource
import subprocess
import time
import psutil

def run_with_timeout(cmd, timeout=15):
    """Run a command with timeout and memory limits"""
    start_time = time.time()
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=lambda: resource.setrlimit(
            resource.RLIMIT_AS,
            (1024 * 1024 * 512, 1024 * 1024 * 512)  # 512MB memory limit
        )
    )
    
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        run_time = time.time() - start_time
        
        # Get memory usage
        process = psutil.Process(process.pid)
        memory_info = process.memory_info()
        memory_used = memory_info.rss / 1024 / 1024  # Convert to MB
        
        return {
            'stdout': stdout.decode(),
            'stderr': stderr.decode(),
            'runtime': run_time,
            'memory': memory_used,
            'timeout': False
        }
    
    except subprocess.TimeoutExpired:
        process.kill()
        return {
            'stdout': '',
            'stderr': 'Timeout',
            'runtime': timeout,
            'memory': 0,
            'timeout': True
        }

def run_code_with_tests(code_file, problem_id, test_cases_path, num_runs, number_of_tests=None):
    """Run code against test cases and return results"""
    problem_input_folder = os.path.join(test_cases_path, problem_id)
    
    if not os.path.exists(problem_input_folder):
        return {
            'valid': False,
            'error': f'Test cases not found for problem {problem_id}'
        }

    # Get input files
    input_files = sorted([f for f in os.listdir(problem_input_folder) 
                         if f.startswith('input')])
    
    if number_of_tests:
        input_files = input_files[:number_of_tests]

    results = {
        'valid': True,
        'passed_tests': set(),
        'errors': {},
        'runtimes': {},
        'memory': {}
    }

    # Run tests
    for input_file in input_files:
        test_id = input_file.split('.')[1]
        input_path = os.path.join(problem_input_folder, input_file)
        output_path = os.path.join(problem_input_folder, f'output.{test_id}.txt')

        # Run multiple times to get average performance
        test_times = []
        test_memory = []
        
        for _ in range(num_runs):
            run_result = run_with_timeout(
                ['python', code_file],
                input=open(input_path, 'r').read()
            )
            
            if run_result['timeout']:
                results['errors'][test_id] = 'Timeout'
                break
                
            if run_result['stderr']:
                results['errors'][test_id] = run_result['stderr']
                break
                
            # Compare output
            expected_output = open(output_path, 'r').read().strip()
            actual_output = run_result['stdout'].strip()
            
            if actual_output == expected_output:
                results['passed_tests'].add(test_id)
                test_times.append(run_result['runtime'])
                test_memory.append(run_result['memory'])
            else:
                results['errors'][test_id] = 'Wrong Answer'
                break

        if test_times:  # Only add metrics if test passed - CONSIDER 
            results['runtimes'][test_id] = sum(test_times) / len(test_times)
            results['memory'][test_id] = sum(test_memory) / len(test_memory)

    return results 