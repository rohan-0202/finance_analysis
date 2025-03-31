from datetime import datetime, time, date

import click
from loguru import logger
import os
import json
import pathlib # Import pathlib

# --- Imports for Backtesting ---
from backtesting.backtest_strategy import Backtest
from backtesting.strategies.strategy_factory import StrategyFactory
from simulations.backtest_instances import get_backtest_instances
from simulations.simulation_config import (
    SimulationConfig,
)
from simulations.ticker_handler import load_tickers_from_file

# -----------------------------


@click.command()
@click.option(
    "--config",
    type=click.Path(exists=True),
    required=True,  # Make config path required
    help="Path to the simulation configuration file",
)
@click.option(
    "--tickers-file",
    default="nyse_tickers.txt",
    help="Path to file containing one ticker per line for random selection.",
)
def run_simulation(config, tickers_file):
    """
    Run simulations based on the given configuration file.
    """
    # --- Determine Base Output Directory from Config Path ---
    config_path = pathlib.Path(config)
    # Get filename without extension (e.g., "example_config")
    config_name = config_path.stem 
    # Create base results path (e.g., "./results/example_config")
    base_output_dir = os.path.join(".", "results", config_name)
    logger.info(f"Base output directory set to: {base_output_dir}")
    os.makedirs(base_output_dir, exist_ok=True)
    # -------------------------------------------------------

    # Load the list of all available tickers first
    logger.info(f"Loading available tickers from: {tickers_file}")
    available_tickers = load_tickers_from_file(tickers_file)
    if not available_tickers:
        logger.error(
            "No available tickers loaded. Cannot proceed with random selections."
        )
        # Decide if you want to exit or continue (maybe only specific runs work)
        # For now, let's exit if the file is needed but empty/missing
        # Consider adding a check later if random selection is actually requested in config
        return

    # Load the configuration file
    logger.info(f"Loading simulation config from: {config}")
    try:
        sim_config = SimulationConfig.from_file(config)
    except (FileNotFoundError, ValueError, Exception) as e:
        logger.error(f"Failed to load or parse configuration file: {e}")
        return  # Exit if config fails

    logger.info(f"Simulation Name: {sim_config.name}")

    # Now given the simulation config, determine the instances of backtest that need to be run
    # Pass the loaded tickers list to the generator
    backtest_instances = get_backtest_instances(
        sim_config, sim_config.data_source.db_name, available_tickers
    )

    if not backtest_instances:
        logger.warning(
            "No backtest instances were generated. Check configuration and logs."
        )
        return

    logger.info(f"Generated {len(backtest_instances)} backtest instances to run.")

    # Run the actual backtests
    for i, instance in enumerate(backtest_instances):
        logger.info(
            f"--- Running Backtest Instance {i + 1}/{len(backtest_instances)} ---"
        )
        logger.info(f"Run ID: {instance.run_id}")
        logger.info(f"Strategy: {instance.strategy_name}")
        logger.info(f"Tickers: {instance.tickers}")
        logger.info(f"Period: {instance.start_date} to {instance.end_date}")
        logger.info(f"DB: {instance.db_name}")

        try:
            # --- Directory creation is now handled above by base_output_dir ---

            # 1. Get the strategy class from the factory
            strategy_class = StrategyFactory.get_strategy_class(instance.strategy_name)

            # 2. Instantiate the Backtest class
            backtester = Backtest(
                strategy_class=strategy_class,
                strategy_params=instance.strategy_params,
                initial_capital=instance.initial_capital,
                commission=instance.commission,
                allow_short_selling=instance.allow_short_selling,
                allow_margin_trading=instance.allow_margin_trading,
            )

            # 3. Convert date objects to datetime objects (start of the day)
            # Backtest.run expects datetime objects.
            # Backtest internally handles timezone conversion to UTC.
            start_dt = datetime.combine(instance.start_date, time.min)
            end_dt = datetime.combine(instance.end_date, time.min)

            # 4. Run the backtest
            logger.info(f"Executing backtest for {instance.run_id}...")
            results = backtester.run(
                tickers=instance.tickers,
                start_date=start_dt,
                end_date=end_dt,
                db_name=instance.db_name,
                benchmark_ticker=instance.benchmark_ticker,
                # Note: data_buffer_months is handled within _prepare_data using the start_date
            )
            # logger.info(f"Results: {results}") # Optional: remove if too verbose

            # 5. Process results 
            logger.success(f"Backtest {instance.run_id} completed successfully.")
            # Optional: Print summary to console still?
            # backtester.print_results() 

            # --- Get results as string and save to TXT file ---
            results_string = backtester.get_results_as_string()
            if results_string:
                 # Sanitize run_id for use in filename if needed (replace slashes, etc.)
                 safe_run_id = instance.run_id.replace(os.sep, '_').replace('/', '_') 
                 results_filename = f"{safe_run_id}.txt"
                 results_filepath = os.path.join(base_output_dir, results_filename)
                 try:
                     with open(results_filepath, 'w') as f:
                         f.write(results_string)
                     logger.info(f"Results saved to: {results_filepath}")
                 except Exception as save_e:
                     logger.error(f"Failed to save results for {instance.run_id} to {results_filepath}: {save_e}")
            else:
                logger.warning(f"Could not get results string for {instance.run_id}")
            # ---------------------------------------------------

            # TODO: Optionally save plots using backtester.plot_results() (needs modification to save to file)

        except ValueError as ve:  # Catch strategy not found errors specifically
            logger.error(
                f"Configuration error for backtest instance {instance.run_id}: {ve}"
            )
        except Exception as e:
            logger.error(
                f"Error running backtest instance {instance.run_id}: {e}", exc_info=True
            )  # Log traceback

    logger.info("Simulation run finished.")


if __name__ == "__main__":
    run_simulation()
