import os
import random
from typing import Optional

from loguru import logger

from simulations.simulation_config import TickerFilter


def load_tickers_from_file(file_path: str) -> list[str]:
    """Loads a list of tickers from a file, one ticker per line."""
    possible_paths = [
        file_path,
        os.path.join("src", file_path),
        os.path.join(os.getcwd(), file_path),
        os.path.join(os.getcwd(), "src", file_path),
    ]
    found_path = None
    for path in possible_paths:
        if os.path.exists(path):
            found_path = path
            break

    if not found_path:
        logger.error(f"Tickers file not found at any expected location: {file_path}")
        return []

    try:
        with open(found_path, "r") as f:
            tickers = [
                line.strip()
                for line in f
                if line.strip() and not line.startswith("#")  # Allow comments
            ]
        logger.info(f"Loaded {len(tickers)} tickers from {found_path}")
        return tickers
    except Exception as e:
        logger.error(f"Error reading tickers file {found_path}: {e}")
        return []


def select_random_tickers(
    available_tickers: list[str],
    count: int,
    seed: Optional[int] = None,
    filters: Optional[TickerFilter] = None,
) -> list[str]:
    """
    Selects a random sample of tickers from the available list, using a seed.
    Filters out tickers starting with '^'.
    Other filters are not yet implemented.
    """
    if not available_tickers:
        logger.warning("Cannot select random tickers: No available tickers provided.")
        return []

    # Filter out tickers starting with '^' (indices)
    filtered_tickers = [t for t in available_tickers if not t.startswith("^")]
    if len(filtered_tickers) < len(available_tickers):
        logger.info(
            f"Filtered out {len(available_tickers) - len(filtered_tickers)} index tickers (starting with '^')."
        )

    # TODO: Implement filtering logic based on TickerFilter (market cap, sector, etc.)
    # For now, we use the filtered list.
    eligible_tickers = filtered_tickers  # Use the filtered list

    if not eligible_tickers:
        logger.warning("No eligible tickers remaining after filtering.")
        return []

    if count >= len(eligible_tickers):
        logger.warning(
            f"Requested count ({count}) is >= available eligible tickers ({len(eligible_tickers)}). Returning all eligible tickers."
        )
        return sorted(eligible_tickers)  # Return sorted list for consistency

    # Use the provided seed for reproducibility
    if seed is not None:
        random.seed(seed)

    selected_tickers = random.sample(eligible_tickers, count)
    return sorted(selected_tickers)  # Return sorted list for consistency
