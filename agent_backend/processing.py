# processing.py

import logging
from typing import Dict, Any, Optional, Literal

# Assuming state_management and llm_integration modules are available
from state_management import ConversationState, Message
# We might not directly call LLM here, but import for type hints if needed
# from llm_integration import LLMIntegrator

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Core Processing Functions (Potential LangGraph Nodes) ---

def route_message(state: ConversationState) -> Literal["greet", "generate_response", "process_tool_request", "end_conversation", "clarify", "error"]:
    """
    Determines the next step in the conversation based on the current state.
    This function typically acts as a conditional router in LangGraph.

    Args:
        state: The current ConversationState.

    Returns:
        A string indicating the next node or action branch to take.
    """
    logging.debug(f"Routing message for conversation: {state.conversation_id}")
    last_message: Optional[Message] = state.history.messages[-1] if state.history.messages else None

    if not last_message:
        logging.warning("Routing called with empty history.")
        return "error" # Or perhaps "greet" if it's the very start

    # Simple routing examples - replace with more sophisticated logic (intent detection, etc.)
    sender = last_message.sender
    content = last_message.content.lower().strip()

    if sender == "user":
        if len(state.history.messages) <= 2 and ("hello" in content or "hi" in content or "hey" in content):
             logging.info("Routing to greeting.")
             return "greet"
        elif "bye" in content or "goodbye" in content or "exit" in content:
             logging.info("Routing to end conversation.")
             return "end_conversation"
        # Basic check for potential tool use (replace with actual tool detection logic)
        elif content.startswith("search for:") or content.startswith("calculate:"):
             logging.info("Routing to process tool request.")
             # We might set something in the state here to indicate which tool
             state.set_intermediate_result("detected_tool_request", content)
             return "process_tool_request"
        else:
            # Default to generating a response
            logging.info("Routing to generate response.")
            return "generate_response"
    elif sender == "assistant":
        # After assistant speaks, usually wait for user or tool
        logging.debug("Routing after assistant message - typically waits for next input.")
        # In LangGraph, this might implicitly transition back to waiting for input
        # or specific nodes might follow assistant responses. For routing logic,
        # we might assume the next step is dictated elsewhere or we need user input.
        # Returning a specific state like 'wait_for_user' could be an option.
        # For simplicity here, we'll assume the graph handles the post-assistant flow.
        # Let's signal that we expect a user response next or maybe just 'continue'.
        return "wait_for_user" # Or define a relevant next step
    elif sender == "tool":
        # After a tool runs, usually generate a response summarizing or using the result
        logging.info("Routing to generate response after tool execution.")
        return "generate_response"
    elif sender == "system":
         logging.debug("Routing after system message - likely generate response or wait.")
         # Similar to assistant, the flow depends on the graph structure
         return "generate_response" # Or "wait_for_user"
    else:
        logging.error(f"Unknown message sender type: {sender}")
        return "error"


def handle_greeting(state: ConversationState) -> Dict[str, Any]:
    """
    Handles the initial greeting phase of the conversation.
    This function would be a node in the LangGraph graph.

    Args:
        state: The current ConversationState.

    Returns:
        A dictionary containing updates to be merged into the state.
        For example, the assistant's greeting message.
    """
    logging.info(f"Handling greeting for conversation: {state.conversation_id}")
    # Example: Add a standard greeting message from the assistant
    greeting_message = "Hello there! How can I assist you today?"
    state.add_assistant_message(greeting_message)

    # Return state updates if LangGraph requires explicit state modification return
    # In many LangGraph patterns, modifying the state object directly is sufficient if mutable.
    # If using immutable state patterns, return the required updates.
    # Example explicit update (adjust based on LangGraph setup):
    return {"history": state.history} # Indicate history has been updated


def apply_response_filters(state: ConversationState, response: str) -> str:
    """
    Applies filtering, moderation, or enhancements to the generated LLM response.
    Can be used as a utility function or potentially a separate node.

    Args:
        state: The current ConversationState (for context, user prefs, etc.).
        response: The raw response string from the LLM.

    Returns:
        The processed and filtered response string.
    """
    logging.debug(f"Applying filters to response for conversation: {state.conversation_id}")
    processed_response = response # Start with the original response

    # --- Placeholder for actual filtering/moderation logic ---
    # 1. Moderation: Check against content policies (e.g., using external APIs or rules)
    # if is_inappropriate(processed_response):
    #     logging.warning("Detected potentially inappropriate content.")
    #     return "I cannot provide a response to that topic."

    # 2. Filtering: Remove sensitive info, PII (if applicable, use regex or NLP tools)
    # processed_response = filter_pii(processed_response)

    # 3. Enhancement: Add canned phrases, formatting, emojis based on context/prefs
    # if "thank you" in response.lower():
    #     processed_response += " 😊"

    # --- End Placeholder ---

    logging.debug("Response filters applied.")
    return processed_response


def prepare_final_response(state: ConversationState) -> Dict[str, Any]:
    """
    Prepares the final response to be sent to the user, potentially after filtering.
    This function could be a node that takes an intermediate LLM result.

    Args:
        state: The current ConversationState, expected to have the raw LLM response
               in intermediate_results (e.g., under 'raw_llm_response').

    Returns:
        A dictionary containing updates to the state, including the final assistant message.
    """
    logging.info(f"Preparing final response for conversation: {state.conversation_id}")
    raw_response = state.get_intermediate_result("raw_llm_response", "")

    if not raw_response:
        logging.warning("No raw LLM response found in state to prepare.")
        # Handle error case - maybe add a generic failure message
        state.add_assistant_message("Sorry, I encountered an issue generating a response.")
        state.error_info = {"step": "prepare_final_response", "error": "Missing raw LLM response"}
        return {"history": state.history, "error_info": state.error_info}

    # Apply filters
    final_response = apply_response_filters(state, raw_response)

    # Add the final, filtered response to the history
    state.add_assistant_message(final_response)

    logging.debug("Final response prepared and added to history.")
    # Return state updates (modify based on LangGraph state handling)
    return {"history": state.history}


def handle_error(state: ConversationState) -> Dict[str, Any]:
    """
    Handles error states in the conversation flow.
    This function would be a node in the LangGraph graph.

    Args:
        state: The current ConversationState.

    Returns:
        A dictionary containing updates to the state, e.g., an error message.
    """
    logging.error(f"Handling error state for conversation: {state.conversation_id}. Error info: {state.error_info}")
    error_message = "Sorry, something went wrong. Please try again later."
    if state.error_info and isinstance(state.error_info.get("error"), str):
        # Optionally provide a bit more detail if safe
        error_message = f"Sorry, an error occurred: {state.error_info['error']}. Please try again."

    state.add_assistant_message(error_message)

    return {"history": state.history}

# --- Example Usage (Illustrative - Not for direct execution in production) ---
if __name__ == "__main__":
    # Demonstrate calling a processing function (requires a State object)
    test_state = ConversationState(conversation_id="proc-test-1")
    test_state.add_user_message("Hello there")

    # Test routing
    next_step = route_message(test_state)
    print(f"Routing decision for 'Hello there': {next_step}") # Expected: greet

    # Test greeting handler (modifies state directly in this example)
    handle_greeting(test_state)
    print("\nState after greeting:")
    for msg in test_state.get_messages():
        print(f"- {msg.sender}: {msg.content}")

    # Test filtering (using a dummy response)
    raw_resp = "This is a test response. Thank you!"
    filtered_resp = apply_response_filters(test_state, raw_resp)
    print(f"\nRaw response: '{raw_resp}'")
    print(f"Filtered response: '{filtered_resp}'") # Expected: Same as raw in this basic impl.

    test_state.add_user_message("ok bye")
    next_step = route_message(test_state)
    print(f"\nRouting decision for 'ok bye': {next_step}") # Expected: end_conversation
