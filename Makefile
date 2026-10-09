.PHONY: test lint run-server clean build benchmark install

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --maxfail=1

benchmark:
	python3 scripts/run_benchmark.py --methods rome memit mend grace ike ft_l

server:
	python3 -m uvicorn editmind.server.app:app --host 0.0.0.0 --port 8000 --reload

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov dist build *.egg-info
