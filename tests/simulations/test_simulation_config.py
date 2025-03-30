# Test Plan for SimulationConfig
# -------------------------------
#
# 1.  **Basic Initialization & Loading:**
#     - Test loading from a valid minimal dictionary.
#     - Test loading from a valid comprehensive dictionary (like the example).
#     - Test loading from a valid minimal JSON file.
#     - Test loading from a valid comprehensive JSON file.
#     - Test error handling for missing 'simulation' key in input dict/JSON.
#     - Test error handling for invalid JSON format.
#     - Test error handling for non-existent file path (`FileNotFoundError`).
#     - Test basic attribute access (name, description, version, created_at, data_source).
#
# 2.  **Default Value Assignment:**
#     - Load a minimal config and verify that default values are correctly applied for:
#         - `PortfolioParameters` (initial_capital, commission, slippage_model, etc.)
#         - `Rebalancing` (enabled, frequency)
#         - `SlippageModel` (type, value)
#         - `Benchmark` (ticker, use_benchmark)
#         - `RiskAnalysis` (monte_carlo.enabled, stress_test.enabled)
#         - `OutputConfig` (save_results, paths, metrics)
#         - `DataSelection` (type, tickers)
#         - `DataSource` (db_name, alternative_sources)
#         - `WalkForwardTest` (enabled)
#         - `ComparativeAnalysis` (enabled)
#
# 3.  **Pydantic Model Validation Tests (Field & Model Level):**
#     - Test `Rebalancing`: ensure `validate_weights` raises ValueError for incorrect sums.
#     - Test `MarketRegime`: ensure `validate_dates` raises ValueError if end_date < start_date.
#     - Test `Timeframe`: ensure `validate_dates` raises ValueError if end_date < start_date.
#     - Test `MonteCarloConfig`: ensure `validate_confidence_interval` raises ValueError for out-of-range values (0, 1, <0, >1).
#     - Test `ParameterSweep`: ensure `validate_parameters` raises ValueError for invalid combinations (both path/paths set, path without values, neither set).
#     - Test `DataSelection`: ensure `validate_fields` raises ValueError if type='specific' but no tickers provided. Ensure default `TickerSelection` is created for random/sector/index types if missing.
#     - Test Enum validation: provide invalid strings for Enum fields (e.g., `DataSelectionType`, `SlippageModelType`) and expect validation errors.
#     - Test type validation: provide incorrect types (e.g., string for a float field) and expect validation errors.
#     - Test datetime parsing: ensure ISO format strings (with/without 'Z') are parsed correctly. Test invalid datetime strings.
#
# 4.  **`get_run` Method:**
#     - Test retrieving a run config using a valid, existing `run_id`.
#     - Test retrieving a run config using a non-existent `run_id` (should return `None`).
#
# 5.  **`get_parameter_sweep_runs` Method:**
#     - Test generating runs for a single parameter sweep (`parameter_path` and `values`). Verify correct IDs, output paths, and nested parameter value updates.
#     - Test generating runs for a multi-parameter sweep (`parameter_paths`). Verify correct IDs, output paths, and nested parameter value updates (Cartesian product).
#     - Test error handling for non-existent `sweep_id`.
#     - Test error handling for non-existent `base_run_id` referenced in the sweep.
#
# 6.  **`get_random_test_runs` Method:**
#     - Test generating runs based on a `RandomTest` configuration.
#     - Verify correct run IDs and output paths based on iterations and seeds.
#     - Verify the `data_selection` block in generated runs is correctly configured (type, ticker_selection details, single seed).
#     - Test error handling for non-existent `test_id`.
#     - Test error handling for non-existent `base_run_id` referenced in the test.
#
# 7.  **`get_multi_strategy_runs` Method:**
#     - Test generating runs based on a `MultiStrategy` configuration.
#     - Verify correct run IDs, strategy configurations, and application of `common_settings`.
#     - Verify correct output path generation.
#     - Test error handling for non-existent `multi_strategy_id`.
#
# 8.  **`get_walk_forward_windows` Method:**
#     - Test generating windows when WFT is enabled. Verify correct start/end dates for training/testing based on config periods ('6m', '1y', '30d', etc.).
#     - Test the number of generated windows matches `steps` (unless constrained by overall dates).
#     - Test behavior when calculated windows extend beyond the overall timeframe end date.
#     - Test returning an empty list when WFT is disabled (`enabled: false`).
#     - Test error handling for invalid period strings (e.g., '6x', 'abc').
#     - Test behavior when no runs are defined (should likely raise an error or return empty list gracefully).
#
# 9.  **Serialization (`to_dict`, `to_file`):**
#     - Test `to_dict()` output structure matches the expected format, including default values and parsed types (e.g., datetimes).
#     - Test `to_file()` successfully creates a JSON file.
#     - Test that loading the file created by `to_file()` using `from_file()` results in an equivalent `SimulationConfig` object.
#     - Specifically check `datetime` serialization to ISO format strings in the output dict/file.
#
# 10. **Edge Cases:**
#     - Test loading a config with empty lists for `runs`, `parameter_sweeps`, `random_testing`, `multi_strategy_testing`.
#     - Test config with more deeply nested parameters within `strategy.parameters`.
#     - Test config with zero or empty string values where semantically valid (e.g., `initial_capital = 0`, `commission = 0`).
#
# 11. **Comprehensive Integration Test (using example_config.json):**
#     - Load `example_config.json`.
#     - Call `to_dict()` and compare the output dictionary deeply with the original loaded JSON data (accounting for potential minor differences like datetime formatting/parsing if necessary, and added defaults).
#     - Potentially generate runs from sweeps/tests defined in the example config and verify their structure.


import json
import os
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

# Assuming the SimulationConfig class is in src.simulations.simulation_config
from src.simulations.simulation_config import (
    ComparativeAnalysis,
    DataSelection,
    DataSelectionType,
    DataSource,
    MonteCarloConfig,
    MonteCarloMethod,
    MultiStrategy,
    OutputConfig,
    ParameterSweep,
    PortfolioParameters,
    RandomTest,
    Rebalancing,
    RebalancingFrequency,
    RiskAnalysis,
    ScenarioConfig,
    SimulationConfig,
    SimulationRun,
    SlippageModel,
    SlippageModelType,
    StrategyConfig,
    StressTestConfig,
    TickerFilter,
    TickerSelection,
    Timeframe,
    Visualization,
    WalkForwardTest,
)

# Define a path for temporary test files
TEST_CONFIG_DIR = Path("tests/temp_configs")
TEST_CONFIG_DIR.mkdir(parents=True, exist_ok=True)


class TestSimulationConfig(unittest.TestCase):
    # --- Test Fixtures ---

    def setUp(self):
        """Set up basic configuration data for tests."""
        self.minimal_config_dict = {
            "simulation": {
                "name": "Minimal Test",
                "runs": [
                    {
                        "id": "run_minimal",
                        "strategy": {"name": "dummy_strategy"},
                        "timeframe": {
                            "start_date": "2024-01-01T00:00:00Z",
                            "end_date": "2024-01-31T00:00:00Z",
                        },
                    }
                ],
            }
        }
        # Add more complex fixtures as needed for specific tests

    def tearDown(self):
        """Clean up any created files."""
        for item in TEST_CONFIG_DIR.iterdir():
            if item.is_file():
                item.unlink()
        # Remove the directory itself if it's empty
        try:
            TEST_CONFIG_DIR.rmdir()
        except OSError:
            pass  # Directory might not be empty if other tests use it

    # --- Test Cases (Implement based on plan above) ---

    # Example Test Case (Step 1)
    def test_load_from_minimal_dict(self):
        """1. Basic Initialization: Test loading from a valid minimal dictionary."""
        config = SimulationConfig.from_dict(self.minimal_config_dict)
        self.assertEqual(config.name, "Minimal Test")
        self.assertEqual(len(config.runs), 1)
        self.assertEqual(config.runs[0].id, "run_minimal")
        self.assertEqual(config.runs[0].strategy.name, "dummy_strategy")
        self.assertIsInstance(config.runs[0].timeframe.start_date, datetime)
        self.assertIsInstance(
            config.runs[0].portfolio, PortfolioParameters
        )  # Check defaults assigned

    # TODO: Add more test methods based on the test plan above


if __name__ == "__main__":
    unittest.main()
