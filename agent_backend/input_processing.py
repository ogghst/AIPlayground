import logging
from typing import Dict, Any

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def normalize_text(text: str) -> str:
    """Applies basic text normalization."""
    if not isinstance(text, str):
        logging.warning(f"normalize_text expected string, got {type(text)}. Returning empty string.")
        return ""
    # Simple normalization: lowercase and strip whitespace
    normalized = text.lower().strip()
    logging.debug(f"Normalized text: '{text}' -> '{normalized}'")
    return normalized

def process_user_input(raw_input: Any) -> Dict[str, Any]:
    """
    Processes raw user input into a structured format for the graph.

    Args:
        raw_input: The raw input received (expected to be text, but checks type).

    Returns:
        A dictionary containing the processed input, e.g.,
        {'normalized_text': 'hello there', 'original_text': ' Hello There! '}
        Can be extended to include language detection, entities, etc.
    """
    logging.debug(f"Processing user input: {raw_input}")
    if not isinstance(raw_input, str):
        logging.warning(f"Received non-string input: {type(raw_input)}. Attempting to convert.")
        raw_input = str(raw_input) # Attempt conversion

    normalized = normalize_text(raw_input)

    processed = {
        "original_text": raw_input,
        "normalized_text": normalized,
        # --- Placeholder for future enhancements ---
        # "detected_language": None,
        # "extracted_entities": [],
        # "detected_intent": None,
        # --- End Placeholder ---
    }
    logging.info(f"Processed input: {processed}")
    return processed

# Example Usage
if __name__ == "__main__":
    test_inputs = ["  Hello World!  ", "What's the weather?", 123, None]
    for inp in test_inputs:
        processed = process_user_input(inp)
        print(f"Input: {inp} => Processed: {processed}")
