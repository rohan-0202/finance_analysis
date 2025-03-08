.PHONY: lint format get_data technical_graphs

# Targets for linting and formatting Python code using Ruff

lint:
	uv run ruff check .

get_data:
	uv run python src/data_insertion.py

technical_graphs:
	$(eval ticker=$(word 2,$(MAKECMDGOALS)))
	uv run python src/technical_graphs.py $(ticker)

format:
	uv run ruff format . 
	uv run isort .

%:
	@:
