import click
from loguru import logger

from simulations.backtest_instances import get_backtest_instances
from simulations.simulation_config import (
    SimulationConfig,
)
from simulations.ticker_handler import load_tickers_from_file


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

    # Placeholder for running the actual backtests
    for i, instance in enumerate(backtest_instances):
        logger.info(
            f"--- Preparing Backtest Instance {i + 1}/{len(backtest_instances)} ---"
        )
        logger.info(f"Run ID: {instance.run_id}")
        logger.info(f"Strategy: {instance.strategy_name}")
        logger.info(f"Tickers: {instance.tickers}")
        logger.info(f"Period: {instance.start_date} to {instance.end_date}")
        logger.info(f"Output Path: {instance.output_path}")
        # TODO: Instantiate Backtest class from backtesting.backtest_strategy
        #       using the instance details and run it.
        #       Ensure the Backtest class can handle `date` objects or convert them
        #       to `datetime` just before calling its run method if needed.
        # Example (needs adjustment based on Backtest class requirements):
        # try:
        #     strategy_class = StrategyFactory.get_strategy_class(instance.strategy_name)
        #     backtester = Backtest(
        #         # ... other params
        #     )
        #     # If Backtest requires datetime, convert here:
        #     # start_dt = datetime.combine(instance.start_date, time.min)
        #     # end_dt = datetime.combine(instance.end_date, time.min)
        #     results = backtester.run(
        #         tickers=instance.tickers,
        #         start_date=instance.start_date, # or start_dt
        #         end_date=instance.end_date,     # or end_dt
        #         db_name=instance.db_name,
        #         benchmark_ticker=instance.benchmark_ticker
        #     )
        #     # ... save results ...
        # except Exception as e:
        #     logger.error(f"Error running backtest instance {instance.run_id}: {e}")

    logger.info("Simulation run finished.")


if __name__ == "__main__":
    run_simulation()
