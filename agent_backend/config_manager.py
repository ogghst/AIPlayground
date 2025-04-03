# config_manager.py

import os
import logging
from typing import Optional, Any, List
from dotenv import load_dotenv

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ConfigManager:
    """
    Manages application configuration, primarily from environment variables.
    Loads variables from a .env file if present.
    """
    def __init__(self, dotenv_path: Optional[str] = None):
        """
        Initializes the ConfigManager and loads environment variables.

        Args:
            dotenv_path: Path to the .env file. If None, searches in standard locations.
        """
        self._load_environment(dotenv_path)
        logging.info("ConfigManager initialized and environment variables loaded.")

    def _load_environment(self, dotenv_path: Optional[str]):
        """Loads environment variables from .env file."""
        try:
            loaded = load_dotenv(dotenv_path=dotenv_path, override=False) # override=False -> existing env vars take precedence
            if loaded:
                logging.info(f"Loaded environment variables from: {dotenv_path or '.env file found'}")
            else:
                logging.info("No .env file found or loaded. Using system environment variables.")
        except Exception as e:
            logging.error(f"Error loading .env file: {e}", exc_info=True)

    def get(self, key: str, default: Optional[Any] = None) -> Optional[str]:
        """
        Retrieves a configuration value (environment variable).

        Args:
            key: The name of the environment variable (case-sensitive).
            default: The value to return if the key is not found.

        Returns:
            The value of the environment variable as a string, or the default value.
        """
        value = os.getenv(key, default)
        # Log access, potentially masking sensitive keys
        log_key = key.upper()
        display_value = "****" if "API_KEY" in log_key or "SECRET" in log_key or "PASSWORD" in log_key else value
        logging.debug(f"Configuration access: key='{key}', returning='{display_value}'")
        return value

    def get_bool(self, key: str, default: bool = False) -> bool:
        """Retrieves a boolean configuration value."""
        value_str = self.get(key)
        if value_str is None:
            return default
        return value_str.lower() in ('true', '1', 't', 'y', 'yes')

    def get_int(self, key: str, default: Optional[int] = None) -> Optional[int]:
        """Retrieves an integer configuration value."""
        value_str = self.get(key)
        if value_str is None:
            return default
        try:
            return int(value_str)
        except ValueError:
            logging.warning(f"Configuration value for key '{key}' ('{value_str}') is not a valid integer. Returning default.")
            return default

    def get_list(self, key: str, default: Optional[List[str]] = None, delimiter: str = ',') -> Optional[List[str]]:
        """Retrieves a list configuration value (comma-separated string)."""
        value_str = self.get(key)
        if value_str is None:
            return default if default is not None else []
        # Split and strip whitespace
        return [item.strip() for item in value_str.split(delimiter) if item.strip()]

    # --- Specific Configuration Properties (Examples) ---

    @property
    def openai_api_key(self) -> Optional[str]:
        return self.get("OPENAI_API_KEY")

    @property
    def log_level(self) -> str:
        return self.get("LOG_LEVEL", "INFO").upper()

    @property
    def ollama_host(self) -> Optional[str]:
        return self.get("OLLAMA_HOST") # Returns None if not set

    @property
    def persistence_dir(self) -> str:
        return self.get("PERSISTENCE_DIR", "./conversation_states")

    @property
    def feature_flag_new_ui(self) -> bool:
        return self.get_bool("FEATURE_FLAG_NEW_UI", False)


# Example Usage (Illustrative)
if __name__ == "__main__":
    # Create a dummy .env file for testing (optional)
    try:
        with open(".env_example_config", "w") as f:
            f.write("OPENAI_API_KEY=sk-dummykeyfromfile\n")
            f.write("LOG_LEVEL=DEBUG\n")
            f.write("FEATURE_FLAG_NEW_UI=true\n")
            f.write("ALLOWED_ORIGINS=http://localhost:3000,https://example.com\n")
            f.write("MAX_RETRIES=5\n")
        # Set an environment variable that should take precedence
        os.environ["OPENAI_API_KEY"] = "sk-dummykeyfromenvvar"

        config = ConfigManager(dotenv_path=".env_example_config")

        print("\n--- Configuration Values ---")
        print(f"OpenAI API Key: {config.openai_api_key}") # Should be from env var
        print(f"Log Level: {config.log_level}")
        print(f"Ollama Host: {config.ollama_host}") # Should be None
        print(f"Persistence Dir: {config.persistence_dir}") # Should be default
        print(f"Feature Flag New UI: {config.feature_flag_new_ui}")
        print(f"Allowed Origins: {config.get_list('ALLOWED_ORIGINS')}")
        print(f"Max Retries: {config.get_int('MAX_RETRIES')}")
        print(f"Non-existent Key (Int): {config.get_int('NON_EXISTENT_KEY', default=99)}")

    finally:
        # Clean up dummy file and env var
        if os.path.exists(".env_example_config"):
            os.remove(".env_example_config")
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"] 