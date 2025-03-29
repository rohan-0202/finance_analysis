.PHONY: lint format get_data get_data_for_ticker technical_graphs news_analysis

# Targets for linting and formatting Python code using Ruff

lint:
	uv run ruff check .

lint-fix:
	uv run ruff check --fix .	

test:
	uv run pytest

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

# extra args are optional. They can be passed like so:
# make news_analysis ticker AAPL ARGS="--no-llm --no-news"
news_analysis:
	$(eval ticker=$(word 2,$(MAKECMDGOALS)))	
	$(if $(ticker),\
		uv run python src/agent/finance_agent_run.py ticker $(ticker) $(ARGS),\
		uv run python src/agent/finance_agent_run.py all $(ARGS))

combined_signals:
	$(eval ticker=$(word 2,$(MAKECMDGOALS)))
	$(if $(ticker),\
		uv run python src/combined_signals.py -t $(ticker),\
		uv run python src/combined_signals.py)

install_jupyter_kernel:
	uv run python -m ipykernel install --user --name=finance_analysis_kernel --display-name "finance_analysis_kernel"

uninstall_jupyter_kernel:
	uv run python -m ipykernel uninstall finance_analysis_kernel

jupyter_notebook:
	uv run jupyter lab	


format:
	uv run ruff format . 
	uv run isort .

%:
	@:
