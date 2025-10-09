PYTHON?=python
PYTHONPATH?=./

.PHONY: vendor run fmt lint test clean

vendor:
	$(PYTHON) -m pip install -r requirements.txt

run:

dev:

app_dirs := industrial_tasks scripts
tests_dir:= tests

fmt:
	isort $(app_dirs)
	black $(app_dirs)

lint:
	@(isok=true; \
	echo "===== black ====="; \
	black --check $(app_dirs) || isok=false; \
	echo "===== flake8 ====="; \
	flake8 $(app_dirs) || isok=false; \
	$$isok && echo "\nLINTERS OK" || echo "\nLINTERS FAILED"; \
	$$isok;)
	
test:
	pytest $(tests_dir)

clean:
	find $(app_dirs) -type d -name "__pycache__" -exec rm -rf {} +
	find $(app_dirs) -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
	find $(app_dirs) -type d -name ".pytest_cache" -exec rm -rf {} +
	echo "Cache cleaned"