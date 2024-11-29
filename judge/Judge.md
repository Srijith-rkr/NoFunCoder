To use this system:
1. Build the Docker image:
```
docker build -t judge .
```
2. Run the judge server:
```
docker run -it -v $(pwd)/data:/app/data judge bash
```

Use run_judge.py to run the judge on a given input file.
Variables to set: TEST_PATH, DPO_OUTPUTS, INPUT_FILE