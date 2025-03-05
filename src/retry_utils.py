import random
import time
from functools import wraps


def retry_on_rate_limit(max_retries=3, base_delay=5):
    """
    Decorator to retry functions when rate limit is hit.
    Uses exponential backoff with jitter for delays.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries <= max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if "Too Many Requests" in str(e) and retries < max_retries:
                        retries += 1
                        # Calculate delay with exponential backoff and jitter
                        delay = base_delay * (2**retries) + random.uniform(0, 3)
                        print(
                            f"Rate limited. Retrying in {delay:.1f} seconds... (Attempt {retries}/{max_retries})"
                        )
                        time.sleep(delay)
                    else:
                        # If max retries reached or other error, raise it
                        raise

        return wrapper

    return decorator 