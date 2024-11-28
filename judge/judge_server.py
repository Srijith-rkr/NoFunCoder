from flask import Flask, request, jsonify
import tempfile
import os
import json
from code_runner import run_code_with_tests

app = Flask(__name__)

@app.route('/judge', methods=['POST'])
def judge():
    try:
        data = request.get_json()
        code = data['code']
        problem_id = data['problem_id']
        test_cases_path = data['test_cases_path']
        num_runs = data.get('num_runs', 1)
        number_of_tests = data.get('number_of_tests')

        # Create a temporary directory for this submission
        with tempfile.TemporaryDirectory() as temp_dir:
            # Write the code to a temporary file
            code_file = os.path.join(temp_dir, 'submission.py')
            with open(code_file, 'w') as f:
                f.write(code)

            # Run the code with test cases
            result = run_code_with_tests(
                code_file,
                problem_id,
                test_cases_path,
                num_runs,
                number_of_tests
            )

            return jsonify(result)

    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 