.PHONY: test smoke

test:
	pytest -q

smoke:
	python examples/quickstart.py
