.PHONY: install lint format typecheck test eval audit ui check

install:
	python -m pip install -e ".[dev,ui,observability,agents]"

lint:
	ruff check src tests eval app.py scripts
	ruff format --check src tests eval app.py scripts

format:
	ruff check --fix src tests eval app.py scripts
	ruff format src tests eval app.py scripts

typecheck:
	mypy

test:
	pytest --cov --cov-report=term-missing -m "not azure"

eval:
	python eval/run_eval.py

audit:
	pip-audit -r requirements.lock --no-deps --disable-pip --ignore-vuln PYSEC-2026-311 --ignore-vuln PYSEC-2026-3813 --ignore-vuln PYSEC-2026-3814 --ignore-vuln PYSEC-2026-3815 --ignore-vuln CVE-2026-112940 --ignore-vuln CVE-2026-112913 --ignore-vuln CVE-2026-112907
	bandit -r src -q

ui:
	streamlit run app.py

check: lint typecheck test eval audit
