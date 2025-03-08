.PHONY: lint format get_data get_data_for_ticker technical_graphs

# Targets for linting and formatting Python code using Ruff

lint:
	uv run ruff check .

lint-fix:
	uv run ruff check --fix .	

get_data:
	uv run python src/data_insertion.py

get_data_for_ticker:
	$(eval ticker=$(word 2,$(MAKECMDGOALS)))
	uv run python src/data_insertion.py --ticker $(ticker)

technical_graphs:
	$(eval ticker=$(word 2,$(MAKECMDGOALS)))
	uv run python src/technical_graphs.py $(ticker)

format:
	uv run ruff format . 
	uv run isort .

%:
	@:
