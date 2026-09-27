PYTHON ?= python3

.PHONY: check test
check:
	$(PYTHON) scripts/check.py

test:
	$(PYTHON) -m pytest
