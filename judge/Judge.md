To use this system:
1. Build the Docker image:
```
docker build -t judge .
```
2. Run the judge server:
```
docker run -p 5000:5000 -v /path/to/test/cases:/app/test_cases code-judge
```

3. Use the judge_submit_local function to submit code to the judge server. (Code change required)

4. Run the evaluation script:
python generate_eval.py --judge_url http://localhost:5000