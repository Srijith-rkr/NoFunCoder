import os
import resource
import subprocess
import time
import psutil

def run_with_timeout(cmd, input='', timeout=20):
    """Run a command with timeout and memory limits"""
    start_time = time.time()
    max_memory = 0
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.PIPE,
        preexec_fn=lambda: resource.setrlimit(
            resource.RLIMIT_AS,
            (1024 * 1024 * 512, 1024 * 1024 * 512)  # 512MB memory limit
        )
    )
    
    try:
        # Monitor memory usage while process is running
        # while process.poll() is None:
        #     try:
        #         mem = psutil.Process(process.pid).memory_info().rss / 1024 / 1024  # MB
        #         max_memory = max(max_memory, mem)
        #     except (psutil.NoSuchProcess, psutil.AccessDenied):
        #         break
        
        stdout, stderr = process.communicate(input=input.encode(), timeout=timeout)
        run_time = time.time() - start_time
        
        return {
            'stdout': stdout.decode(),
            'stderr': stderr.decode(),
            'runtime': run_time,
            'memory': 0, #HERE
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

def run_code_with_tests(code_file, input, output, num_runs=1):
    """Run code against test case and return result"""

    results = {
        'valid': True,
        'passed': False,
        'errors': None,
        'runtimes': None,
        'memory': None
    }
    test_times = []
    test_memory = []
        
    for _ in range(num_runs):
        run_result = run_with_timeout(
            ['python', code_file],
            input=input
        )
        
        if run_result['timeout']:
            results['errors'] = 'Timeout'
            break
            
        if run_result['stderr']:
            results['errors'] = run_result['stderr']
            break
            
        # Compare output
        expected_output = output.strip()
        actual_output = run_result['stdout'].strip()
        
        if actual_output == expected_output:
            results['passed']=True
            test_times.append(run_result['runtime'])
            test_memory.append(run_result['memory'])
        else:
            results['errors'] = 'Wrong Answer'
            break

    if test_times:  # Only add metrics if test passed - CONSIDER 
        results['runtimes'] = sum(test_times) / len(test_times)
        results['memory'] = sum(test_memory) / len(test_memory)

    # # Run tests
    # for input_file in input_files:
    #     test_id = input_file.split('.')[1]
    #     input_path = os.path.join(problem_input_folder, input_file)
    #     output_path = os.path.join(problem_input_folder, f'output.{test_id}.txt')

        # Run multiple times to get average performance
        

    return results 