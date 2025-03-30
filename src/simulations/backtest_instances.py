import os
from datetime import date
from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel

from simulations.simulation_config import (
    DataSelectionType,
    SimulationConfig,
    TickerFilter,
)
from simulations.ticker_handler import select_random_tickers


class BacktestInstance(BaseModel):
    """Represents a single backtest execution configuration."""

    run_id: str  # Unique ID for this specific backtest run instance
    strategy_name: str  # Name of the strategy class to use
    strategy_params: Dict[str, Any]
    initial_capital: float
    commission: float
    allow_short_selling: bool
    allow_margin_trading: bool
    tickers: List[str]  # Specific list of tickers for this instance
    start_date: date  # Changed from datetime to date
    end_date: date  # Changed from datetime to date
    data_buffer_months: int
    benchmark_ticker: Optional[str]
    db_name: str  # Database name passed from the command line or config
    output_path: str  # Full path to save results for this instance


def _create_backtest_instance_from_dict(
    run_dict: Dict[str, Any], db_name: str, available_tickers: List[str]
) -> Optional[BacktestInstance]:
    """
    Helper function to create a BacktestInstance from a run dictionary.
    Handles data validation and selection logic (specific, random).
    """
    run_id = run_dict.get("id")
    if not run_id:
        logger.warning("Skipping run dictionary with missing 'id'.")
        return None

    # Extract necessary sections, providing defaults where possible
    strategy_info = run_dict.get("strategy", {})
    portfolio_info = run_dict.get("portfolio", {})
    timeframe_info = run_dict.get("timeframe", {})
    data_selection_info = run_dict.get("data_selection", {})
    benchmark_info = run_dict.get("benchmark", {})
    output_info = run_dict.get("output", {})

    # Validate required fields exist
    if not all(
        [
            strategy_info.get("name"),
            data_selection_info.get("type"),
        ]
    ):
        logger.warning(
            f"Skipping run {run_id}: Missing required configuration fields (strategy name, data type)."
        )
        return None

    # --- Data Selection Logic ---
    data_type_str = data_selection_info.get("type")
    tickers: Optional[List[str]] = None

    try:
        data_type = DataSelectionType(data_type_str)
        if data_type == DataSelectionType.SPECIFIC:
            tickers = data_selection_info.get("tickers", [])
            if not tickers:
                logger.warning(
                    f"Skipping run {run_id}: Data type is 'specific' but no tickers provided."
                )
                return None
        elif data_type == DataSelectionType.RANDOM:
            ticker_selection_info = data_selection_info.get("ticker_selection")
            if not ticker_selection_info:
                logger.warning(
                    f"Skipping run {run_id}: Data type is 'random' but 'ticker_selection' config is missing."
                )
                return None

            # Extract parameters for random selection
            count = ticker_selection_info.get("count")
            # Use the first seed if available, otherwise None (will use system time)
            seeds = ticker_selection_info.get("seeds", [None])
            seed = seeds[0] if seeds else None
            # Extract filters (currently unused in selection logic but passed for future)
            filter_dict = ticker_selection_info.get("filter", {})
            filters = TickerFilter(**filter_dict) if filter_dict else None

            if count is None or count <= 0:
                logger.warning(
                    f"Skipping run {run_id}: Invalid or missing 'count' ({count}) in ticker_selection for random type."
                )
                return None

            logger.info(
                f"Selecting {count} random tickers with seed {seed} for run {run_id}."
            )
            tickers = select_random_tickers(
                available_tickers=available_tickers,
                count=count,
                seed=seed,
                filters=filters,
            )
            if not tickers:
                logger.warning(
                    f"Skipping run {run_id}: Failed to select random tickers (count: {count}, seed: {seed})."
                )
                return None
            logger.info(f"Selected tickers for run {run_id}: {tickers}")

        # Add elif for INDEX, SECTOR etc. when implemented
        else:
            logger.warning(
                f"Skipping run {run_id}: Data selection type '{data_type_str}' is not yet supported."
            )
            return None
    except ValueError:
        logger.warning(
            f"Skipping run {run_id}: Invalid configuration value (e.g., data selection type '{data_type_str}')."
        )
        return None
    except Exception as e:
        logger.error(f"Error during data selection for run {run_id}: {e}")
        return None

    if not tickers:  # Should be caught above, but double-check
        logger.warning(f"Skipping run {run_id}: Failed to determine tickers.")
        return None

    # --- Use date objects directly ---
    start_date_obj: Optional[date] = timeframe_info.get("start_date")
    end_date_obj: Optional[date] = timeframe_info.get("end_date")

    # Pydantic should guarantee these are valid date objects if present
    if not start_date_obj or not end_date_obj:
        logger.warning(
            f"Skipping run {run_id}: Missing start or end date in timeframe."
        )
        return None

    benchmark_ticker = (
        benchmark_info.get("ticker") if benchmark_info.get("use_benchmark") else None
    )

    # Determine output path - use the structure from SimulationRun.get_full_output_path
    base_output_path = output_info.get("results_path", "./results/")
    # Ensure the run_id is part of the path if not already handled by the generator functions
    # The generator functions (get_parameter_sweep_runs etc.) seem to set results_path correctly already.
    # Let's trust that for now, but if not, we'd use os.path.join(base_output_path, run_id)
    output_path = output_info.get(
        "results_path", os.path.join(base_output_path, run_id)
    )

    try:
        instance = BacktestInstance(
            run_id=run_id,
            strategy_name=strategy_info.get("name"),
            strategy_params=strategy_info.get("parameters", {}),
            initial_capital=portfolio_info.get("initial_capital", 10000.0),
            commission=portfolio_info.get("commission", 0.001),
            allow_short_selling=portfolio_info.get("allow_short_selling", False),
            allow_margin_trading=portfolio_info.get("allow_margin_trading", False),
            tickers=tickers,
            start_date=start_date_obj,
            end_date=end_date_obj,
            data_buffer_months=timeframe_info.get("data_buffer_months", 2),
            benchmark_ticker=benchmark_ticker,
            db_name=db_name,  # Use the provided db_name
            output_path=output_path,
        )
        return instance
    except Exception as e:  # Catch potential Pydantic validation errors or others
        logger.error(f"Failed to create BacktestInstance for run {run_id}: {e}")
        return None


def get_backtest_instances(
    config: SimulationConfig, db_name: str, available_tickers: List[str]
) -> list[BacktestInstance]:
    """
    Generates a list of BacktestInstance configurations based on the SimulationConfig.

    This function processes standard runs, parameter sweeps, random tests, and multi-strategy tests
    defined in the configuration file. It consolidates them into a flat list of specific
    backtests to be executed.

    Args:
        config: The loaded SimulationConfig object.
        db_name: The name of the database to use for fetching historical data.
        available_tickers: Full list of tickers available for random selection.

    Returns:
        A list of BacktestInstance objects, each ready to be passed to a backtesting engine.
    """
    all_instances: list[BacktestInstance] = []
    processed_run_ids = set()  # Keep track of generated IDs to avoid duplicates if base runs are also used in sweeps etc.

    # 1. Process direct runs defined in 'runs'
    logger.info(f"Processing {len(config.runs)} direct simulation runs...")
    for run in config.runs:
        if run.id in processed_run_ids:
            logger.warning(f"Skipping duplicate run ID from direct runs: {run.id}")
            continue
        instance = _create_backtest_instance_from_dict(
            run.model_dump(), db_name, available_tickers
        )
        if instance:
            all_instances.append(instance)
            processed_run_ids.add(instance.run_id)

    # 2. Process parameter sweeps
    logger.info(f"Processing {len(config.parameter_sweeps)} parameter sweeps...")
    for sweep in config.parameter_sweeps:
        logger.info(f"Generating runs for parameter sweep: {sweep.id}")
        try:
            sweep_run_dicts = config.get_parameter_sweep_runs(sweep.id)
            logger.info(f"Generated {len(sweep_run_dicts)} runs from sweep {sweep.id}.")
            for run_dict in sweep_run_dicts:
                run_id = run_dict.get("id")
                if run_id in processed_run_ids:
                    logger.warning(
                        f"Skipping duplicate run ID from parameter sweep {sweep.id}: {run_id}"
                    )
                    continue
                instance = _create_backtest_instance_from_dict(
                    run_dict, db_name, available_tickers
                )
                if instance:
                    all_instances.append(instance)
                    processed_run_ids.add(instance.run_id)
        except ValueError as e:
            logger.error(f"Error generating runs for parameter sweep {sweep.id}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error processing parameter sweep {sweep.id}: {e}")

    # 3. Process random testing
    logger.info(f"Processing {len(config.random_testing)} random tests...")
    for test in config.random_testing:
        logger.info(f"Generating runs for random test: {test.id}")
        try:
            test_run_dicts = config.get_random_test_runs(test.id)
            logger.info(
                f"Generated {len(test_run_dicts)} runs from random test {test.id}."
            )
            for run_dict in test_run_dicts:
                run_id = run_dict.get("id")
                if run_id in processed_run_ids:
                    logger.warning(
                        f"Skipping duplicate run ID from random test {test.id}: {run_id}"
                    )
                    continue
                instance = _create_backtest_instance_from_dict(
                    run_dict, db_name, available_tickers
                )
                if instance:
                    all_instances.append(instance)
                    processed_run_ids.add(instance.run_id)
                else:
                    # Log that it was skipped due to implementation limits if run_id exists
                    if run_id:
                        logger.debug(
                            f"Run {run_id} from random test {test.id} skipped (likely due to unimplemented data selection type)."
                        )

        except ValueError as e:
            logger.error(f"Error generating runs for random test {test.id}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error processing random test {test.id}: {e}")

    # 4. Process multi-strategy testing
    logger.info(
        f"Processing {len(config.multi_strategy_testing)} multi-strategy tests..."
    )
    for multi_strat in config.multi_strategy_testing:
        logger.info(f"Generating runs for multi-strategy test: {multi_strat.id}")
        try:
            multi_run_dicts = config.get_multi_strategy_runs(multi_strat.id)
            logger.info(
                f"Generated {len(multi_run_dicts)} runs from multi-strategy test {multi_strat.id}."
            )
            for run_dict in multi_run_dicts:
                run_id = run_dict.get("id")
                if run_id in processed_run_ids:
                    logger.warning(
                        f"Skipping duplicate run ID from multi-strategy test {multi_strat.id}: {run_id}"
                    )
                    continue
                instance = _create_backtest_instance_from_dict(
                    run_dict, db_name, available_tickers
                )
                if instance:
                    all_instances.append(instance)
                    processed_run_ids.add(instance.run_id)
        except ValueError as e:
            logger.error(
                f"Error generating runs for multi-strategy test {multi_strat.id}: {e}"
            )
        except Exception as e:
            logger.error(
                f"Unexpected error processing multi-strategy test {multi_strat.id}: {e}"
            )

    # TODO: Add processing for walk-forward testing if needed.
    # This would involve generating instances for each walk-forward window (training/testing).

    logger.info(f"Total backtest instances generated: {len(all_instances)}")
    return all_instances
