.PHONY: ruff
ruff:
	ruff check . --fix
	ruff format .