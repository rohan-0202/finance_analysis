.PHONY: lint format

# Targets for linting and formatting Python code using Ruff

lint:
	uv run ruff check .

format:
	uv run ruff format . 