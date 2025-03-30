"""
Strategy Factory

This module provides a factory for creating strategy instances based on strategy names.
It handles the dynamic loading of strategy classes from their respective files.
"""

import importlib.util
import inspect
import os
from typing import Dict, List, Type

from backtesting.portfolio import Portfolio
from backtesting.strategy import Strategy


class StrategyFactory:
    """Factory for creating strategy instances."""

    @staticmethod
    def get_strategies_dir() -> str:
        """Find the path to the strategies directory."""
        # Try different possible locations for the strategies directory
        possible_paths = [
            os.path.join("src", "backtesting", "strategies"),
            os.path.join("backtesting", "strategies"),
            os.path.join(os.getcwd(), "src", "backtesting", "strategies"),
            os.path.join(os.getcwd(), "backtesting", "strategies"),
        ]

        for path in possible_paths:
            if os.path.exists(path) and os.path.isdir(path):
                return path

        raise FileNotFoundError("Could not find strategies directory")

    @staticmethod
    def list_available_strategies() -> List[str]:
        """
        List all available strategy files in the strategies directory.

        Returns:
        --------
        List[str]
            List of strategy names (without the .py extension)
        """
        try:
            strategies_dir = StrategyFactory.get_strategies_dir()
            strategy_files = []

            # List all Python files in the directory
            for file in os.listdir(strategies_dir):
                if file.endswith(".py") and not file.startswith("__"):
                    # Remove the .py extension
                    strategy_name = file[:-3]
                    strategy_files.append(strategy_name)

            return sorted(strategy_files)

        except Exception as e:
            print(f"Error listing strategies: {e}")
            return []

    @staticmethod
    def get_strategy_class(strategy_name: str) -> Type[Strategy]:
        """
        Get a strategy class by name.

        Parameters:
        -----------
        strategy_name : str
            Name of the strategy (without .py extension)

        Returns:
        --------
        Type[Strategy]
            The strategy class

        Raises:
        -------
        ValueError
            If the strategy file doesn't exist or doesn't contain a Strategy subclass
        """
        # Clean up the strategy name input
        if strategy_name.endswith(".py"):
            strategy_name = strategy_name[:-3]

        # Get the strategies directory
        strategies_dir = StrategyFactory.get_strategies_dir()

        # Construct the full path to the strategy file
        strategy_path = os.path.join(strategies_dir, f"{strategy_name}.py")

        if not os.path.exists(strategy_path):
            raise ValueError(
                f"Strategy file '{strategy_name}.py' not found in '{strategies_dir}'"
            )

        try:
            # Import the module
            spec = importlib.util.spec_from_file_location(strategy_name, strategy_path)
            if spec is None or spec.loader is None:
                raise ValueError(f"Could not load spec for {strategy_name}")

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Find the strategy class in the module
            strategy_class = None
            for _, obj in inspect.getmembers(module):
                if (
                    inspect.isclass(obj)
                    and issubclass(obj, Strategy)
                    and obj != Strategy
                    and obj.__module__ == module.__name__
                ):
                    strategy_class = obj
                    break

            if strategy_class is None:
                raise ValueError(f"No Strategy subclass found in '{strategy_name}.py'")

            return strategy_class

        except Exception as e:
            raise ValueError(f"Error loading strategy '{strategy_name}': {e}") from e

    @staticmethod
    def create_strategy(strategy_name: str, portfolio: Portfolio, **kwargs) -> Strategy:
        """
        Create a strategy instance by name.

        Parameters:
        -----------
        strategy_name : str
            Name of the strategy (without .py extension)
        portfolio : Portfolio
            Portfolio instance to use with the strategy
        **kwargs
            Additional parameters to pass to the strategy

        Returns:
        --------
        Strategy
            An instance of the requested strategy

        Raises:
        -------
        ValueError
            If the strategy cannot be created
        """
        strategy_class = StrategyFactory.get_strategy_class(strategy_name)

        # Create a new instance of the strategy
        strategy = strategy_class(portfolio)

        # Set parameters if any were provided
        if kwargs:
            strategy.set_parameters(**kwargs)

        return strategy

    @staticmethod
    def get_default_parameters(strategy_name: str) -> Dict:
        """
        Get default parameters for a strategy.

        Parameters:
        -----------
        strategy_name : str
            Name of the strategy (without .py extension)

        Returns:
        --------
        Dict
            Dictionary of default parameters for the strategy

        Notes:
        ------
        Strategy classes should implement a class method `get_default_parameters()`
        to provide their default parameters.
        """
        strategy_class = StrategyFactory.get_strategy_class(strategy_name)

        # Check if the strategy has a get_default_parameters class method
        if hasattr(strategy_class, "get_default_parameters") and callable(
            strategy_class.get_default_parameters
        ):
            return strategy_class.get_default_parameters()

        # If no defaults are provided, return an empty dict
        print(
            f"Warning: Strategy '{strategy_name}' doesn't implement get_default_parameters(). Using empty defaults."
        )
        return {}
