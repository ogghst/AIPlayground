import os
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import asyncio

# Assuming state_management module exists in the same directory or PYTHONPATH
from state_management import ConversationState

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Abstract Base Class for LLM Providers ---

class LLMProvider(ABC):
    """Abstract base class for different LLM API providers."""

    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """
        Generates text based on the provided prompt and parameters.

        Args:
            prompt: The input prompt for the language model.
            **kwargs: Provider-specific parameters (e.g., temperature, max_tokens).

        Returns:
            The generated text response from the language model.

        Raises:
            Exception: If the generation process fails.
        """
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Returns the name of the LLM provider (e.g., 'openai', 'anthropic', 'ollama')."""
        pass


# --- Concrete Implementations ---

# --- OpenAI Provider ---
# NOTE: Requires 'openai' library and OPENAI_API_KEY env var.
class OpenAIProvider(LLMProvider):
    """LLM Provider implementation for OpenAI's API (GPT models)."""

    def __init__(self, model: str = "gpt-4o", api_key: Optional[str] = None):
        """Initializes the OpenAIProvider."""
        try:
            from openai import AsyncOpenAI
        except ImportError:
            logging.error("OpenAI library not found. Please install it: pip install openai")
            raise ImportError("OpenAI library is required for OpenAIProvider.")

        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided or found in environment variables (OPENAI_API_KEY).")

        try:
            self.client = AsyncOpenAI(api_key=self.api_key)
        except Exception as e:
            logging.error(f"Failed to initialize OpenAI client: {e}", exc_info=True)
            raise ConnectionError(f"Failed to initialize OpenAI client: {e}") from e

    @property
    def provider_name(self) -> str:
        return "openai"

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generates text using the configured OpenAI model."""
        logging.debug(f"Generating response from OpenAI model {self.model} with prompt: {prompt[:100]}...")
        system_prompt_content = kwargs.pop("system_prompt", None)
        messages = []
        if system_prompt_content:
            messages.append({"role": "system", "content": system_prompt_content})
        messages.append({"role": "user", "content": prompt})

        params = {
            "temperature": kwargs.pop("temperature", 0.7), # Extract known params
            "max_tokens": kwargs.pop("max_tokens", 1024),
            **kwargs # Pass remaining specific params
        }

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **params
            )
            if response.choices and len(response.choices) > 0:
                generated_text = response.choices[0].message.content
                logging.debug(f"OpenAI response received: {generated_text[:100]}...")
                return generated_text.strip() if generated_text else ""
            else:
                logging.warning(f"OpenAI response did not contain expected choices structure: {response}")
                return ""
        except Exception as e:
            logging.error(f"OpenAI API call failed: {e}", exc_info=True)
            raise ConnectionError(f"Failed to generate response from OpenAI: {e}") from e


# --- Ollama Provider ---
# NOTE: Requires 'ollama' library and a running Ollama instance.
#       `pip install ollama`
class OllamaProvider(LLMProvider):
    """LLM Provider implementation for Ollama."""

    def __init__(self, model: str = "llama3", host: Optional[str] = None):
        """
        Initializes the OllamaProvider.

        Args:
            model: The name of the Ollama model to use (e.g., "llama3", "mistral").
                   Ensure this model is pulled in your Ollama instance.
            host: The host URL of the Ollama instance (e.g., "http://localhost:11434").
                  If None, uses the default from the ollama library (usually env var or default).
        """
        try:
            import ollama
        except ImportError:
            logging.error("Ollama library not found. Please install it: pip install ollama")
            raise ImportError("Ollama library is required for OllamaProvider.")

        self.model = model
        self.host = host
        # Initialize the async Ollama client
        try:
            # Pass host only if provided, otherwise let the library use defaults
            client_args = {}
            if self.host:
                client_args['host'] = self.host
            self.client = ollama.AsyncClient(**client_args)
            # Consider adding a check here to see if the host is reachable or model exists
            # e.g., by calling self.client.list() or similar, though might slow down init.
            logging.info(f"Ollama client initialized for model '{self.model}'"
                         f"{(' targeting host ' + self.host) if self.host else ' (using default host)'}.")
        except Exception as e:
            logging.error(f"Failed to initialize Ollama client: {e}", exc_info=True)
            raise ConnectionError(f"Failed to initialize Ollama client: {e}") from e

    @property
    def provider_name(self) -> str:
        return "ollama"

    async def generate(self, prompt: str, **kwargs) -> str:
        """
        Generates text using the configured Ollama model.

        Args:
            prompt: The input prompt.
            **kwargs: Additional parameters for the Ollama API call, such as:
                      - system_prompt (str, Optional): A system message.
                      - template (str, Optional): The full prompt template.
                      - options (dict, Optional): Ollama-specific runtime parameters (e.g., temperature).

        Returns:
            The generated text content.

        Raises:
            ConnectionError: If the API call fails.
        """
        logging.debug(f"Generating response from Ollama model {self.model} with prompt: {prompt[:100]}...")

        system_prompt = kwargs.pop("system_prompt", None)
        ollama_template = kwargs.pop("template", None) # Ollama's specific template parameter
        ollama_options = kwargs.pop("options", {}) # Ollama specific options (temp, top_p, etc.)

        # Allow overriding default options via kwargs directly if not in 'options'
        ollama_options['temperature'] = kwargs.pop('temperature', ollama_options.get('temperature', 0.7))
        # Add other common parameters if needed, mapping them to ollama_options

        # Build message list for Ollama's chat endpoint (usually preferred)
        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': prompt})

        try:
            response = await self.client.chat(
                model=self.model,
                messages=messages,
                stream=False, # Get full response at once
                options=ollama_options if ollama_options else None, # Pass options if provided
                template=ollama_template if ollama_template else None # Pass template if provided
                # format: Optional[str] = None # e.g., 'json' if needed
            )

            # Accessing the response content correctly based on Ollama's library structure
            if response and 'message' in response and 'content' in response['message']:
                generated_text = response['message']['content']
                logging.debug(f"Ollama response received: {generated_text[:100]}...")
                return generated_text.strip()
            else:
                logging.warning(f"Ollama response did not contain expected structure: {response}")
                return ""

        except Exception as e:
            logging.error(f"Ollama API call failed: {e}", exc_info=True)
            # Check for common connection errors
            if "connection refused" in str(e).lower():
                 error_msg = f"Connection refused. Ensure Ollama is running{(' at ' + self.host) if self.host else ''} and the model '{self.model}' is available."
                 raise ConnectionError(error_msg) from e
            raise ConnectionError(f"Failed to generate response from Ollama model '{self.model}': {e}") from e


# --- LLM Integrator Class ---

class LLMIntegrator:
    """
    Handles interaction with a selected LLM provider, including prompt formatting.
    """

    def __init__(self, provider: LLMProvider):
        """Initializes the LLMIntegrator."""
        if not isinstance(provider, LLMProvider):
            raise TypeError("Provider must be an instance of LLMProvider")
        self.provider = provider
        logging.info(f"LLMIntegrator initialized with provider: {self.provider.provider_name}")

    def _format_prompt(self, state: ConversationState, prompt_template: str) -> str:
        """Formats the prompt using a template and the current conversation state."""
        history_str = "\n".join([f"{msg.sender}: {msg.content}" for msg in state.get_messages()])
        last_user_message = next((msg.content for msg in reversed(state.get_messages()) if msg.sender != 'assistant' and msg.sender != 'tool'), "")

        try:
            formatted_prompt = prompt_template.format(
                history=history_str,
                user_input=last_user_message,
                short_term_memory=str(state.memory.short_term),
                conversation_id=state.conversation_id
            )
            return formatted_prompt
        except KeyError as e:
            logging.error(f"Missing key in prompt template: {e}. Template: '{prompt_template}'")
            return prompt_template # Fallback
        except Exception as e:
            logging.error(f"Error formatting prompt: {e}", exc_info=True)
            raise ValueError(f"Failed to format prompt: {e}")

    async def generate_response(
        self,
        state: ConversationState,
        prompt_template: str,
        model_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """Formats the prompt, calls the LLM provider, and returns the response."""
        if model_params is None:
            model_params = {}

        try:
            # Formatting should ideally return just the core user prompt for most models
            # System prompts and history are often handled differently by providers
            # Let's assume format_prompt returns the main content/question
            core_prompt_content = self._format_prompt(state, prompt_template)
            logging.debug(f"Formatted core prompt content generated for provider {self.provider.provider_name}.")

            # Pass state or extracted parts if needed by the provider's generate method directly
            # For OpenAI/Ollama chat, system prompt and history are often passed via specific args
            # Consider adjusting _format_prompt or how generate is called if needed

            # Example: Extract system prompt if present in model_params
            provider_specific_params = model_params.copy() # Avoid modifying original dict
            # system_prompt = provider_specific_params.pop("system_prompt", None) # Example

            response = await self.provider.generate(core_prompt_content, **provider_specific_params)
            logging.info(f"Successfully generated response using {self.provider.provider_name}.")
            return response

        except (ValueError, ConnectionError) as e:
            logging.error(f"Failed to generate LLM response: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error during LLM response generation: {e}", exc_info=True)
            raise ConnectionError(f"An unexpected error occurred: {e}") from e

# --- Example Usage (Illustrative) ---

async def main(provider_type: str = "ollama"):
    # Choose provider based on argument
    provider: Optional[LLMProvider] = None
    if provider_type == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            print("\n--- Skipping OpenAI Example ---")
            print("Set the OPENAI_API_KEY environment variable to run.")
            print("----------------------------\n")
            return
        try:
            provider = OpenAIProvider(model="gpt-4o")
        except (ImportError, ValueError, ConnectionError) as e:
            logging.error(f"Failed to initialize OpenAI provider: {e}")
            return
    elif provider_type == "ollama":
        try:
            # Assumes default host and qwq model pulled
            provider = OllamaProvider(model="qwq")
            # Test connection quickly (optional)
            # await provider.client.ps() # Example check
        except (ImportError, ValueError, ConnectionError) as e:
            logging.error(f"Failed to initialize Ollama provider: {e}")
            print(f"\n--- Skipping Ollama Example ---")
            print(f"Failed to initialize Ollama provider: {e}")
            print("Ensure Ollama is installed (`pip install ollama`), running, and the model 'wqw' is pulled.")
            print("----------------------------\n")
            return
        except Exception as e: # Catch other potential init errors
             logging.error(f"Unexpected error initializing Ollama: {e}", exc_info=True)
             print(f"Unexpected error initializing Ollama: {e}")
             return

    else:
        print(f"Unknown provider type: {provider_type}")
        return

    integrator = LLMIntegrator(provider=provider)

    # Create a dummy conversation state
    state = ConversationState(conversation_id=f"test-conv-{provider_type}")
    state.add_user_message("Hello, chatbot!")
    state.add_assistant_message("Hello! How can I help you today?")
    state.add_user_message("Why is the sky blue?") # Use a factual question

    # Define a simple prompt template - adjust if needed for specific models
    # This template assumes the core user input is what's needed in the 'user_input' placeholder
    template = """
Conversation History:
{history}

Based on the history, answer the last user message: '{user_input}'
"""
    # Define model parameters (can vary by provider)
    params = {"temperature": 0.7, "system_prompt": "You are a helpful assistant explaining science concepts simply."}
    # For Ollama, if needed: params["options"] = {"temperature": 0.7}

    try:
        print(f"\n--- Generating Response ({provider.provider_name}) ---")
        response = await integrator.generate_response(state, template, params)
        print("\n--- Generated Response ---")
        print(response)
        print("-------------------------\n")
        state.add_assistant_message(response)
    except Exception as e:
        print(f"\n--- Error ({provider.provider_name}) ---")
        print(f"An error occurred: {e}")
        print("-------------\n")


if __name__ == "__main__":
    # Example: Choose provider via environment variable or default to openai
    provider_choice = os.getenv("LLM_PROVIDER", "openai").lower()
    asyncio.run(main(provider_type=provider_choice))
