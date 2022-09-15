OUTPUT_DIR= .

install-dep:
	poetry install

check-bandit:
	poetry run bandit -lll -iii -r . -x ./tests

check-safety:
	poetry export -f requirements.txt | poetry run safety check --stdin

requirement-file:
	poetry export -f requirements.txt -o $(OUTPUT_DIR)/requirements.txt

format:
	poetry run black ./degas_sci

check-flake8:
	poetry run flake8

build-package:
	poetry build

build-bundle:
	poetry build --format wheel
	pip install ./dist/*.whl --target $(OUTPUT_DIR)/.python_packages/lib/site-packages

unit-tests:
	poetry run pytest tests

unit-tests-cov:
	poetry run pytest tests -s --cov-append --doctest-modules --junitxml=$(OUTPUT_DIR)/junit/unit-tests-results.xml --cov=degas_sci --cov-report=xml:$(OUTPUT_DIR)/coverage.xml --cov-report=html:$(OUTPUT_DIR)/htmlcov

text-unit-tests-cov:
	poetry run pytest --cache-clear --cov=evse -vvv tests/ > $(OUTPUT_DIR)/pytest-coverage.txt
