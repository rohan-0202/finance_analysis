.PHONY: lint format get_data get_data_for_ticker technical_graphs news_analysis

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

options_analysis:
	$(eval ticker=$(word 2,$(MAKECMDGOALS)))
	uv run python src/options_analysis.py $(ticker)

news_analysis:
	$(eval ticker=$(word 2,$(MAKECMDGOALS)))
	uv run python -m src.agent.finance_news_agent ticker $(ticker)

combined_signals:
	$(eval ticker=$(word 2,$(MAKECMDGOALS)))
	$(if $(ticker),\
		uv run python src/combined_signals.py -t $(ticker),\
		uv run python src/combined_signals.py)

format:
	uv run ruff format . 
	uv run isort .

%:
	@:
