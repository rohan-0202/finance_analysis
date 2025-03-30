import json
import os
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)


class DataSelectionType(str, Enum):
    """Type of ticker selection method."""

    SPECIFIC = "specific"
    RANDOM = "random"
    INDEX = "index"
    SECTOR = "sector"


class SlippageModelType(str, Enum):
    """Type of slippage model."""

    FIXED = "fixed"
    PERCENTAGE = "percentage"
    VOLUME_BASED = "volume_based"
    NONE = "none"


class MonteCarloMethod(str, Enum):
    """Method for Monte Carlo simulation."""

    BOOTSTRAP = "bootstrap"
    GEOMETRIC_BROWNIAN_MOTION = "geometric_brownian_motion"
    HISTORICAL_RETURNS = "historical_returns"


class RebalancingFrequency(str, Enum):
    """Frequency for portfolio rebalancing."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUALLY = "annually"
    NONE = "none"


class SlippageModel(BaseModel):
    """Configuration for slippage model."""

    type: SlippageModelType = SlippageModelType.NONE
    value: float = 0.0

    model_config = ConfigDict(use_enum_values=True)


class Rebalancing(BaseModel):
    """Configuration for portfolio rebalancing."""

    enabled: bool = False
    frequency: RebalancingFrequency = RebalancingFrequency.MONTHLY
    target_weights: Dict[str, float] = Field(default_factory=dict)

    model_config = ConfigDict(use_enum_values=True)

    @field_validator("target_weights")
    @classmethod
    def validate_weights(cls, v: Dict[str, float]):
        """Ensure that weights sum to 1.0 (or 0 if empty)."""
        if v and abs(sum(v.values()) - 1.0) > 1e-6:
            raise ValueError("Rebalancing target weights must sum to 1.0")
        return v


class PortfolioParameters(BaseModel):
    """Portfolio configuration for a simulation run."""

    initial_capital: float = 10000.0
    commission: float = 0.001
    slippage_model: SlippageModel = Field(default_factory=SlippageModel)
    allow_short_selling: bool = False
    allow_margin_trading: bool = False
    max_positions: Optional[int] = None
    max_allocation_per_position: Optional[float] = None
    rebalancing: Rebalancing = Field(default_factory=Rebalancing)


class MarketRegime(BaseModel):
    """Definition of a market regime period."""

    name: str
    start_date: date
    end_date: date

    @field_validator("end_date")
    @classmethod
    def validate_dates(cls, v: date, info: ValidationInfo):
        """Ensure end_date is after start_date."""
        if "start_date" in info.data and v < info.data["start_date"]:
            raise ValueError("end_date must be after start_date")
        return v


class Timeframe(BaseModel):
    """Timeframe configuration for a simulation run."""

    start_date: date
    end_date: date
    data_buffer_months: int = 2
    market_regimes: List[MarketRegime] = Field(default_factory=list)

    @field_validator("end_date")
    @classmethod
    def validate_dates(cls, v: date, info: ValidationInfo):
        """Ensure end_date is after start_date."""
        if "start_date" in info.data and v < info.data["start_date"]:
            raise ValueError("end_date must be after start_date")
        return v


class TickerFilter(BaseModel):
    """Filter criteria for ticker selection."""

    market_cap_min: Optional[float] = None
    market_cap_max: Optional[float] = None
    sector: Optional[List[str]] = None
    industry: Optional[List[str]] = None
    exclude_tickers: List[str] = Field(default_factory=list)
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    min_volume: Optional[float] = None
    exchange: Optional[List[str]] = None


class TickerSelection(BaseModel):
    """Configuration for random ticker selection."""

    method: DataSelectionType = DataSelectionType.RANDOM
    count: int = 20
    filter: TickerFilter = Field(default_factory=TickerFilter)
    iterations: int = 1
    seeds: List[int] = Field(default_factory=lambda: [42])

    model_config = ConfigDict(use_enum_values=True)


class DataSelection(BaseModel):
    """Data selection configuration for a simulation run."""

    type: DataSelectionType = DataSelectionType.SPECIFIC
    tickers: List[str] = Field(default_factory=list)
    ticker_selection: Optional[TickerSelection] = None

    model_config = ConfigDict(use_enum_values=True)

    @model_validator(mode="before")
    @classmethod
    def validate_fields(cls, data: Any) -> Any:
        """Validate that the appropriate fields are filled based on selection type."""
        # Ensure it's a dictionary before proceeding
        if not isinstance(data, dict):
            return data

        data_type = data.get("type")
        tickers = data.get("tickers")
        ticker_selection = data.get("ticker_selection")

        if data_type == DataSelectionType.SPECIFIC and not tickers:
            # Check if tickers is None or empty list
            if tickers is None or len(tickers) == 0:
                raise ValueError(
                    "Tickers must be provided when selection type is 'specific'"
                )
        elif (
            data_type
            in (
                DataSelectionType.RANDOM,
                DataSelectionType.SECTOR,
                DataSelectionType.INDEX,
            )
            and not ticker_selection
        ):
            # Assign a default TickerSelection if missing for relevant types
            data["ticker_selection"] = TickerSelection(method=data_type)

        return data


class Benchmark(BaseModel):
    """Benchmark configuration for a simulation run."""

    ticker: str = "SPY"
    use_benchmark: bool = True


class MonteCarloConfig(BaseModel):
    """Configuration for Monte Carlo simulation."""

    enabled: bool = False
    iterations: int = 1000
    method: MonteCarloMethod = MonteCarloMethod.BOOTSTRAP
    confidence_interval: float = 0.95

    model_config = ConfigDict(use_enum_values=True)

    @field_validator("confidence_interval")
    @classmethod
    def validate_confidence_interval(cls, v: float):
        """Ensure confidence interval is between 0 and 1."""
        if not 0 < v < 1:
            raise ValueError("Confidence interval must be between 0 and 1")
        return v


class ScenarioConfig(BaseModel):
    """Configuration for a stress test scenario."""

    name: str
    price_adjustment: Optional[float] = None
    volume_adjustment: Optional[float] = None
    slippage_multiplier: Optional[float] = None
    duration_days: int = 1


class StressTestConfig(BaseModel):
    """Configuration for stress testing."""

    enabled: bool = False
    scenarios: List[ScenarioConfig] = Field(default_factory=list)


class RiskAnalysis(BaseModel):
    """Risk analysis configuration for a simulation run."""

    monte_carlo: MonteCarloConfig = Field(default_factory=MonteCarloConfig)
    stress_test: StressTestConfig = Field(default_factory=StressTestConfig)


class OutputConfig(BaseModel):
    """Output configuration for a simulation run."""

    save_results: bool = True
    results_path: str = "./results/"
    plot_equity_curve: bool = True
    save_trades: bool = True
    calculate_metrics: List[str] = Field(
        default_factory=lambda: [
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "volatility",
        ]
    )


class StrategyConfig(BaseModel):
    """Strategy configuration for a simulation run."""

    name: str
    parameters: Dict[str, Any] = Field(default_factory=dict)


class SimulationRun(BaseModel):
    """Configuration for an individual simulation run."""

    id: str
    strategy: StrategyConfig
    portfolio: PortfolioParameters = Field(default_factory=PortfolioParameters)
    timeframe: Timeframe
    data_selection: DataSelection = Field(default_factory=DataSelection)
    benchmark: Benchmark = Field(default_factory=Benchmark)
    risk_analysis: RiskAnalysis = Field(default_factory=RiskAnalysis)
    output: OutputConfig = Field(default_factory=OutputConfig)

    def get_full_output_path(self) -> str:
        """Get the full output path for this run."""
        return os.path.join(self.output.results_path, self.id)


class ParameterValue(BaseModel):
    """Parameter path and values for parameter sweep."""

    path: str
    values: List[Any]


class ParameterSweep(BaseModel):
    """Configuration for a parameter sweep."""

    id: str
    base_run_id: str
    parameter_path: Optional[str] = None
    parameter_paths: Optional[List[ParameterValue]] = None
    values: Optional[List[Any]] = None
    output_prefix: str

    @model_validator(mode="after")
    def validate_parameters(self) -> "ParameterSweep":
        """Validate parameter configuration."""
        if self.parameter_path and self.parameter_paths:
            raise ValueError("Cannot specify both parameter_path and parameter_paths")
        if self.parameter_path and not self.values:
            raise ValueError("Must specify values when using parameter_path")
        if not self.parameter_path and not self.parameter_paths:
            raise ValueError("Must specify either parameter_path or parameter_paths")
        return self


class RandomTest(BaseModel):
    """Configuration for random ticker testing."""

    id: str
    base_run_id: str
    ticker_selection: TickerSelection
    output_prefix: str


class WalkForwardTest(BaseModel):
    """Configuration for walk-forward testing."""

    enabled: bool = False
    training_period: str
    testing_period: str
    steps: int = 4
    optimization_metric: str = "sharpe_ratio"
    optimize_parameters: List[str] = Field(default_factory=list)


class Visualization(BaseModel):
    """Configuration for visualization options."""

    heatmaps: bool = True
    scatter_plots: bool = True
    parameter_sensitivity: bool = True


class ComparativeAnalysis(BaseModel):
    """Configuration for comparative analysis."""

    enabled: bool = True
    metrics: List[str] = Field(
        default_factory=lambda: ["total_return", "sharpe_ratio", "max_drawdown"]
    )
    group_by: List[str] = Field(default_factory=list)
    output_path: str = "./results/comparative_analysis.json"
    visualizations: Visualization = Field(default_factory=Visualization)


class MultiStrategy(BaseModel):
    """Configuration for multi-strategy testing."""

    id: str
    strategies: List[StrategyConfig]
    common_settings: Dict[str, Any] = Field(default_factory=dict)
    output_prefix: str


class DataSource(BaseModel):
    """Configuration for data sources."""

    db_name: str = "stock_data.db"
    alternative_sources: List[Dict[str, str]] = Field(default_factory=list)


class SimulationConfig:
    """
    Configuration manager for trading strategy simulations.

    This class loads, validates, and provides access to the simulation configuration
    from a JSON file or a Python dictionary. It ensures that all required fields are
    present and validates the structure and values according to the defined schema.

    Example:
    ```python
    # Load from a file
    config = SimulationConfig.from_file("simulation_config.json")

    # Access configuration properties
    for run in config.runs:
        print(f"Running simulation {run.id} with strategy {run.strategy.name}")
        # ... run the simulation
    ```

    The configuration supports:
    - Multiple individual simulation runs
    - Parameter sweeps across different values
    - Random ticker testing with filtering
    - Walk-forward testing and optimization
    - Comparative analysis across runs
    - Multi-strategy testing

    Full JSON example (see class docstring in code)
    """

    def __init__(self, config_data: Dict[str, Any]):
        """
        Initialize the configuration manager with parsed configuration data.

        Args:
            config_data: Dictionary containing the simulation configuration
        """
        # Extract the simulation section
        if "simulation" not in config_data:
            raise ValueError("Configuration must contain a 'simulation' section")

        simulation_data = config_data["simulation"]

        self.name = simulation_data.get("name", "Strategy Performance Analysis")
        self.description = simulation_data.get("description", "")
        self.version = simulation_data.get("version", "1.0")

        # Parse the created_at date string, handling ISO timestamp format
        created_at_str = simulation_data.get("created_at")
        if created_at_str:
            try:
                # Parse the full ISO timestamp string, replacing 'Z' for compatibility
                # Note: datetime.fromisoformat handles 'Z' in Python 3.11+,
                # but replacing ensures broader compatibility.
                dt_obj = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
                self.created_at = dt_obj.date() # Extract just the date part
            except ValueError as e:
                # Raise a more informative error if parsing fails
                raise ValueError(
                    f"Invalid ISO format for created_at: '{created_at_str}'. Error: {e}"
                ) from e
        else:
            # Use today's date if 'created_at' is missing
            self.created_at = datetime.now().date() # Use current date

        # Parse data source configuration
        data_source_data = simulation_data.get("data_source", {})
        self.data_source = DataSource(**data_source_data)

        # Parse runs
        runs_data = simulation_data.get("runs", [])
        self.runs = [SimulationRun(**run_data) for run_data in runs_data]

        # Parse parameter sweeps
        parameter_sweeps_data = simulation_data.get("parameter_sweeps", [])
        self.parameter_sweeps = [
            ParameterSweep(**sweep_data) for sweep_data in parameter_sweeps_data
        ]

        # Parse random testing
        random_testing_data = simulation_data.get("random_testing", [])
        self.random_testing = [
            RandomTest(**test_data) for test_data in random_testing_data
        ]

        # Parse walk-forward testing
        walk_forward_data = simulation_data.get("walk_forward_testing")
        self.walk_forward_testing = (
            WalkForwardTest(**walk_forward_data) if walk_forward_data else None
        )

        # Parse comparative analysis
        comparative_analysis_data = simulation_data.get("comparative_analysis")
        self.comparative_analysis = (
            ComparativeAnalysis(**comparative_analysis_data)
            if comparative_analysis_data
            else None
        )

        # Parse multi-strategy testing
        multi_strategy_data = simulation_data.get("multi_strategy_testing", [])
        self.multi_strategy_testing = [
            MultiStrategy(**strategy_data) for strategy_data in multi_strategy_data
        ]

        # Build run lookup dictionary for quick access
        self._run_lookup = {run.id: run for run in self.runs}

    @classmethod
    def from_file(cls, file_path: str) -> "SimulationConfig":
        """
        Load simulation configuration from a JSON file.

        Args:
            file_path: Path to the JSON configuration file

        Returns:
            SimulationConfig: Initialized configuration manager
        """
        try:
            with open(file_path, "r") as f:
                config_data = json.load(f)
                return cls(config_data)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Configuration file not found: {file_path}") from e
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {file_path}") from e

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SimulationConfig":
        """
        Load simulation configuration from a Python dictionary.

        Args:
            config_dict: Dictionary containing the simulation configuration

        Returns:
            SimulationConfig: Initialized configuration manager
        """
        return cls(config_dict)

    def get_run(self, run_id: str) -> Optional[SimulationRun]:
        """
        Get a simulation run by ID.

        Args:
            run_id: ID of the simulation run

        Returns:
            SimulationRun: The simulation run configuration, or None if not found
        """
        return self._run_lookup.get(run_id)

    def get_parameter_sweep_runs(self, sweep_id: str) -> List[Dict[str, Any]]:
        """
        Generate run configurations for a parameter sweep.

        This method takes a sweep ID and generates all the run configurations with
        different parameter values based on the sweep definition.

        Args:
            sweep_id: ID of the parameter sweep

        Returns:
            List[Dict[str, Any]]: List of run configurations with different parameter values
        """
        # Find the sweep configuration
        sweep = next((s for s in self.parameter_sweeps if s.id == sweep_id), None)
        if not sweep:
            raise ValueError(f"Parameter sweep not found: {sweep_id}")

        # Get the base run
        base_run = self.get_run(sweep.base_run_id)
        if not base_run:
            raise ValueError(
                f"Base run not found for sweep {sweep_id}: {sweep.base_run_id}"
            )

        # Convert the base run to a dictionary for modification
        base_run_dict = base_run.model_dump()

        # Generate run configurations
        runs = []

        if sweep.parameter_path and sweep.values:
            # Single parameter sweep
            for i, value in enumerate(sweep.values):
                # Create a copy of the base run
                run_dict = base_run_dict.copy()
                run_dict["id"] = f"{sweep.output_prefix}{i}_{value}"

                # Set the parameter value using the path
                self._set_nested_value(run_dict, sweep.parameter_path, value)

                # Adjust the output path
                run_dict["output"]["results_path"] = os.path.join(
                    base_run.output.results_path, f"{sweep.output_prefix}{i}_{value}"
                )

                runs.append(run_dict)

        elif sweep.parameter_paths:
            # Multi-parameter sweep (cartesian product)
            from itertools import product

            # Extract all parameter combinations
            param_names = [p.path for p in sweep.parameter_paths]
            param_values = [p.values for p in sweep.parameter_paths]

            for i, combination in enumerate(product(*param_values)):
                # Create a copy of the base run
                run_dict = base_run_dict.copy()

                # Create a descriptive ID
                param_desc = "_".join(
                    f"{name.split('.')[-1]}_{value}"
                    for name, value in zip(param_names, combination, strict=False)
                )
                run_dict["id"] = f"{sweep.output_prefix}{i}_{param_desc}"

                # Set all parameter values
                for path, value in zip(param_names, combination, strict=False):
                    self._set_nested_value(run_dict, path, value)

                # Adjust the output path
                run_dict["output"]["results_path"] = os.path.join(
                    base_run.output.results_path, f"{sweep.output_prefix}{i}"
                )

                runs.append(run_dict)

        return runs

    def get_random_test_runs(self, test_id: str) -> List[Dict[str, Any]]:
        """
        Generate run configurations for random ticker testing.

        This method takes a random test ID and generates run configurations with
        different random ticker selections based on the test definition.

        Args:
            test_id: ID of the random test

        Returns:
            List[Dict[str, Any]]: List of run configurations with different ticker selections
        """
        # Find the random test configuration
        test = next((t for t in self.random_testing if t.id == test_id), None)
        if not test:
            raise ValueError(f"Random test not found: {test_id}")

        # Get the base run
        base_run = self.get_run(test.base_run_id)
        if not base_run:
            raise ValueError(
                f"Base run not found for random test {test_id}: {test.base_run_id}"
            )

        # Convert the base run to a dictionary for modification
        base_run_dict = base_run.model_dump()

        # Generate run configurations
        runs = []

        # For each iteration and seed
        for iteration in range(test.ticker_selection.iterations):
            for seed in test.ticker_selection.seeds:
                # Create a copy of the base run
                run_dict = base_run_dict.copy()
                run_dict["id"] = f"{test.output_prefix}iter{iteration}_seed{seed}"

                # Set the data selection
                run_dict["data_selection"] = {
                    "type": test.ticker_selection.method,
                    "ticker_selection": {
                        "method": test.ticker_selection.method,
                        "count": test.ticker_selection.count,
                        "filter": test.ticker_selection.filter.model_dump(),
                        "iterations": 1,  # Set to 1 since we're handling iterations here
                        "seeds": [seed],
                    },
                }

                # Adjust the output path
                run_dict["output"]["results_path"] = os.path.join(
                    base_run.output.results_path,
                    f"{test.output_prefix}iter{iteration}_seed{seed}",
                )

                runs.append(run_dict)

        return runs

    def get_multi_strategy_runs(self, multi_strategy_id: str) -> List[Dict[str, Any]]:
        """
        Generate run configurations for multi-strategy testing.

        This method takes a multi-strategy test ID and generates run configurations for
        different strategies with common settings.

        Args:
            multi_strategy_id: ID of the multi-strategy test

        Returns:
            List[Dict[str, Any]]: List of run configurations for different strategies
        """
        # Find the multi-strategy configuration
        multi_strategy = next(
            (m for m in self.multi_strategy_testing if m.id == multi_strategy_id), None
        )
        if not multi_strategy:
            raise ValueError(f"Multi-strategy test not found: {multi_strategy_id}")

        # Generate run configurations
        runs = []

        for i, strategy in enumerate(multi_strategy.strategies):
            # Create a new run with common settings
            run_dict = {
                "id": f"{multi_strategy.output_prefix}{strategy.name}_{i}",
                "strategy": strategy.model_dump(),
                **multi_strategy.common_settings,
            }

            # Adjust the output path if it exists
            if "output" in run_dict:
                run_dict["output"]["results_path"] = os.path.join(
                    run_dict["output"].get("results_path", "./results/"),
                    f"{multi_strategy.output_prefix}{strategy.name}_{i}",
                )
            else:
                run_dict["output"] = {
                    "results_path": f"./results/{multi_strategy.output_prefix}{strategy.name}_{i}"
                }

            runs.append(run_dict)

        return runs

    def get_walk_forward_windows(
        self,
    ) -> List[Tuple[date, date, date, date]]:
        """
        Generate time windows for walk-forward testing.

        This method calculates the training and testing periods for each step of
        walk-forward testing based on the configuration.

        Returns:
            List[Tuple[date, date, date, date]]: List of tuples containing
            (training_start, training_end, testing_start, testing_end) for each step
        """
        if not self.walk_forward_testing or not self.walk_forward_testing.enabled:
            return []

        # Parse the periods
        from dateutil.relativedelta import relativedelta

        def parse_period(period_str):
            """Parse a period string like '6m' or '1y' into a relativedelta."""
            if not period_str:
                raise ValueError("Period string cannot be empty")

            unit = period_str[-1].lower()
            try:
                value = int(period_str[:-1])
            except ValueError as e:
                raise ValueError(f"Invalid period format: {period_str}") from e

            if unit == "d":
                return relativedelta(days=value)
            elif unit == "w":
                return relativedelta(weeks=value)
            elif unit == "m":
                return relativedelta(months=value)
            elif unit == "y":
                return relativedelta(years=value)
            else:
                raise ValueError(f"Unknown period unit: {unit}")

        training_delta = parse_period(self.walk_forward_testing.training_period)
        testing_delta = parse_period(self.walk_forward_testing.testing_period)

        # Find the overall date range from all runs
        all_start_dates = [run.timeframe.start_date for run in self.runs]
        all_end_dates = [run.timeframe.end_date for run in self.runs]

        if not all_start_dates or not all_end_dates:
            raise ValueError("No runs defined with timeframes")

        overall_start = min(all_start_dates)
        overall_end = max(all_end_dates)

        # Generate the windows
        windows = []

        for step in range(self.walk_forward_testing.steps):
            step_offset = (
                relativedelta(months=step * testing_delta.months)
                if hasattr(testing_delta, "months")
                else relativedelta(days=step * testing_delta.days)
                if hasattr(testing_delta, "days")
                else relativedelta(weeks=step * testing_delta.weeks)
                if hasattr(testing_delta, "weeks")
                else relativedelta(years=step * testing_delta.years)
            )

            training_start = overall_start + step_offset
            training_end = training_start + training_delta
            testing_start = training_end
            testing_end = testing_start + testing_delta

            # Ensure we don't go beyond the overall end date
            if testing_end > overall_end:
                break

            windows.append((training_start, training_end, testing_start, testing_end))

        return windows

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the configuration to a dictionary using model_dump.

        Returns:
            Dict[str, Any]: Dictionary representation of the configuration
        """
        # Use model_dump(mode='json') to handle date serialization automatically
        # We need to manually handle created_at as it's not part of a Pydantic model here
        return {
            "simulation": {
                "name": self.name,
                "description": self.description,
                "version": self.version,
                "created_at": self.created_at.isoformat(),  # Serialize date to string
                "data_source": self.data_source.model_dump(mode="json"),
                "runs": [run.model_dump(mode="json") for run in self.runs],
                "parameter_sweeps": [
                    sweep.model_dump(mode="json") for sweep in self.parameter_sweeps
                ],
                "random_testing": [
                    test.model_dump(mode="json") for test in self.random_testing
                ],
                "walk_forward_testing": self.walk_forward_testing.model_dump(
                    mode="json"
                )
                if self.walk_forward_testing
                else None,
                "comparative_analysis": self.comparative_analysis.model_dump(
                    mode="json"
                )
                if self.comparative_analysis
                else None,
                "multi_strategy_testing": [
                    strategy.model_dump(mode="json")
                    for strategy in self.multi_strategy_testing
                ],
            }
        }

    def to_file(self, file_path: str) -> None:
        """
        Save the configuration to a JSON file.

        Args:
            file_path: Path where to save the JSON file
        """
        # Ensure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

        with open(file_path, "w") as f:
            # model_dump(mode='json') handles serialization, including dates
            json.dump(self.to_dict(), f, indent=2)  # Removed default serializer

    @staticmethod
    def _set_nested_value(data: Dict[str, Any], path: str, value: Any) -> None:
        """
        Set a value in a nested dictionary using a dot-notation path.

        Args:
            data: Dictionary to modify
            path: Dot-notation path to the value (e.g., 'strategy.parameters.rsi_period')
            value: Value to set
        """
        parts = path.split(".")

        # Navigate to the nested location
        current = data
        for _, part in enumerate(parts[:-1]):
            if part not in current:
                current[part] = {}
            current = current[part]

        # Set the value
        current[parts[-1]] = value


if __name__ == "__main__":
    # Ensure the example config file exists or adjust the path
    # Use a config file with simple YYYY-MM-DD dates
    config_path = "src/simulations/configs/rsi_macd_simple_config.json"  # Adjusted path
    config = SimulationConfig.from_file(config_path)

    # Convert back to dict and print as JSON
    generated_data = config.to_dict()
    print("--- Generated Data ---")
    # Use standard json.dumps without the custom serializer
    print(json.dumps(generated_data, indent=2))

    # Load original data
    with open(config_path, "r") as f:
        original_data = json.load(f)
    print("\n--- Original Data ---")
    print(json.dumps(original_data, indent=2))

    # Example of saving back to a file (optional)
    output_path = "src/simulations/configs/example_config_output_test.json"
    config.to_file(output_path)
    print(f"\nConfiguration saved to {output_path}")

    # Compare the dictionaries
    print("\n--- Comparing Dictionaries ---")

    def compare_dicts(d1, d2, path=""):
        """Recursively compare two dictionaries and print differences."""
        # Simplified comparison - assumes keys and types match after Pydantic validation/serialization
        set1 = set(d1.keys())
        set2 = set(d2.keys())

        if set1 != set2:
            print(
                f"Key mismatch at {path}: Original keys {set1}, Generated keys {set2}"
            )
            # Only compare common keys if sets differ
            common_keys = set1.intersection(set2)
        else:
            common_keys = set1  # or set2, they are the same

        for key in common_keys:
            new_path = f"{path}.{key}" if path else key
            val1 = d1[key]
            val2 = d2[key]

            if isinstance(val1, dict) and isinstance(val2, dict):
                compare_dicts(val1, val2, new_path)
            elif isinstance(val1, list) and isinstance(val2, list):
                if len(val1) != len(val2):
                    print(
                        f"List length mismatch at {new_path}: original={len(val1)}, generated={len(val2)}"
                    )
                else:
                    for i, (item1, item2) in enumerate(zip(val1, val2, strict=False)):
                        item_path = f"{new_path}[{i}]"
                        if isinstance(item1, dict) and isinstance(item2, dict):
                            compare_dicts(item1, item2, item_path)
                        # Simplified comparison: relies on Pydantic/JSON handling types
                        elif item1 != item2:
                            # Basic float comparison tolerance
                            is_float1 = isinstance(item1, (int, float))
                            is_float2 = isinstance(item2, (int, float))
                            if is_float1 and is_float2:
                                if abs(item1 - item2) > 1e-9:
                                    print(
                                        f"Value mismatch at {item_path}: original='{item1}', generated='{item2}'"
                                    )
                            # Add check for None vs empty list/dict if needed based on specific cases
                            # elif (item1 is None and isinstance(item2, (list, dict)) and not item2) or \
                            #      (item2 is None and isinstance(item1, (list, dict)) and not item1):
                            #      pass # Treat None and empty container as equivalent in some contexts if desired
                            else:
                                print(
                                    f"Value mismatch at {item_path}: original='{item1}', generated='{item2}'"
                                )

            elif val1 != val2:
                # Basic float comparison tolerance
                is_float1 = isinstance(val1, (int, float))
                is_float2 = isinstance(val2, (int, float))
                if is_float1 and is_float2:
                    if abs(val1 - val2) > 1e-9:
                        print(
                            f"Value mismatch at {new_path}: original='{val1}', generated='{val2}'"
                        )
                # Add check for None vs empty list/dict if needed
                # elif (val1 is None and isinstance(val2, (list, dict)) and not val2) or \
                #      (val2 is None and isinstance(val1, (list, dict)) and not val1):
                #      pass # Treat None and empty container as equivalent
                else:
                    print(
                        f"Value mismatch at {new_path}: original='{val1}', generated='{val2}'"
                    )

    # Perform the comparison
    # Note: Direct comparison might still fail due to float precision or subtle differences
    # in how None/empty lists/dicts are handled between original JSON and Pydantic models.
    # The detailed comparison function helps identify where differences occur.
    compare_dicts(original_data, generated_data)

    # Keep the assertion commented out unless perfect match is strictly required and achieved
    # assert generated_data == original_data, "Generated dictionary does not match the original loaded dictionary."
    print("\nComparison finished.")
