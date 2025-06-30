import os
from dotenv import load_dotenv

def load_configuration() -> dict:
    """
    Loads and validates required configuration from environment variables.

    Returns:
        A dictionary with LLM configuration parameters.

    Raises:
        ValueError: If a required environment variable is not set.
    """
    load_dotenv()
    
    config = {
        "api_key": os.getenv("LLM_API_KEY"),
        "base_url": os.getenv("LLM_BASE_URL"),
        "model_name": os.getenv("LLM_MODEL_NAME")
    }

    if not all([config["api_key"], config["base_url"], config["model_name"]]):
        raise ValueError(
            "FATAL: One or more LLM environment variables are missing. "
            "Please check LLM_API_KEY, LLM_BASE_URL, and LLM_MODEL_NAME in your .env file."
        )

    return config
