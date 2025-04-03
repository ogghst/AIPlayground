# graph_definition.py

import logging
import os
from typing import Dict, Any, Literal, Optional

# LangGraph imports
try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver # For persisting state during runs
except ImportError:
    logging.error("LangGraph library not found. Please install it: pip install langgraph")
    raise

# Core module imports
from state_management import ConversationState, UserInfo, StateManager
from llm_integration import LLMIntegrator, OpenAIProvider, OllamaProvider
from processing import (
    route_message,
    handle_greeting,
    prepare_final_response,
    handle_error
)
from tool_integration import execute_tool_node, ToolExecutor, SimpleCalculatorTool, WebSearchTool # Import tools/executor if needed directly

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Graph Configuration ---

# Initialize components (should ideally be managed/injected)
# Check for API Key before initializing provider
if not os.getenv("OPENAI_API_KEY"):
    #logging.warning("OPENAI_API_KEY not set. LLM functionality will fail.")
    try:
        llm_provider = OpenAIProvider(model="gpt-4o")
        llm_integrator = LLMIntegrator(provider=llm_provider)
    except Exception as e:
         logging.error(f"Failed to initialize Ollama provider: {e}", exc_info=True)
         llm_provider = None
         llm_integrator = None

else:
    try:
        llm_provider = OllamaProvider(model="qwq")
        llm_integrator = LLMIntegrator(provider=llm_provider)
    except Exception as e:
         logging.error(f"Failed to initialize OpenAI provider: {e}", exc_info=True)
         llm_provider = None
         llm_integrator = None


# Simple prompt template (customize as needed)
DEFAULT_PROMPT_TEMPLATE = """
You are a helpful assistant. Continue the conversation based on the history.

Conversation History:
{history}

Current User Input/Tool Result: {user_input}

Your Response:"""

# --- Node Functions ---
# These functions adapt our module functions to the input/output expected by LangGraph nodes
# They take the state dict and return a dict of updates.

async def greeting_node(state: ConversationState) -> Dict[str, Any]:
    """Node wrapper for handling greetings."""
    logging.debug("Executing greeting_node")
    # handle_greeting modifies state directly in our current implementation
    handle_greeting(state)
    # Return the modified part of the state (or potentially the whole state if needed)
    return {"history": state.history}

async def call_llm_node(state: ConversationState) -> Dict[str, Any]:
    """Node wrapper for calling the LLM."""
    logging.debug("Executing call_llm_node")
    if not llm_integrator:
        logging.error("LLM Integrator not available.")
        state.error_info = {"step": "call_llm_node", "error": "LLM Integrator not initialized (check API key?)"}
        state.add_message("system", "Error: Cannot connect to the language model.")
        return {"error_info": state.error_info, "history": state.history}

    # Determine the input for the prompt template (could be last user msg or tool result)
    last_message = state.history.messages[-1] if state.history.messages else None
    prompt_input = last_message.content if last_message else ""
    template = DEFAULT_PROMPT_TEMPLATE.replace("{user_input}", prompt_input) # Simple replacement

    try:
        response = await llm_integrator.generate_response(
            state=state,
            prompt_template=template,
            model_params={"temperature": 0.7} # Example params
        )
        # Store raw response for potential filtering/processing
        return {"intermediate_results": {**state.intermediate_results, "raw_llm_response": response}}
    except Exception as e:
        logging.error(f"Error in call_llm_node: {e}", exc_info=True)
        state.error_info = {"step": "call_llm_node", "error": f"LLM generation failed: {e}"}
        state.add_message("system", f"Error generating response: {e}")
        return {"error_info": state.error_info, "history": state.history}

async def prepare_response_node(state: ConversationState) -> Dict[str, Any]:
    """Node wrapper for preparing the final response."""
    logging.debug("Executing prepare_response_node")
    # prepare_final_response modifies state and returns updates in our current impl
    updates = prepare_final_response(state)
    return updates

async def error_node(state: ConversationState) -> Dict[str, Any]:
    """Node wrapper for handling errors."""
    logging.debug("Executing error_node")
    # handle_error modifies state directly
    handle_error(state)
    return {"history": state.history, "error_info": state.error_info}


# --- Router Functions (Moved to Module Level) ---

def entry_router(state: ConversationState) -> Literal["greet", "call_llm", "execute_tool", "end_conversation", "error"]:
    """Determines the initial routing based on the current state."""
    # If there's an error flag set, immediately go to error handling
    if state.error_info:
        logging.warning("Error flag set, routing to handle_error.")
        return "error"

    # Use the processing module's router
    decision = route_message(state)

    # Map router decisions to graph node names or special outcomes
    # Note: END is handled by the conditional edge mapping, return the decision string
    route_map = {
        "greet": "greet",
        "generate_response": "call_llm",
        "process_tool_request": "execute_tool",
        "end_conversation": "end_conversation", # Return the decision string
        "clarify": "call_llm",
        "wait_for_user": "wait_for_user",     # Return the decision string
        "error": "handle_error"
    }
    graph_destination = route_map.get(decision, "handle_error") # Default to error
    logging.info(f"Entry router decision: '{decision}' -> Routing to graph node/state: '{graph_destination}'")
    return graph_destination

def agent_router_node(state: ConversationState) -> Literal["greet", "call_llm", "execute_tool", "end_conversation", "wait_for_user", "error"]:
    """This node's *only* job is to route. It calls entry_router.
       It directly returns the decision for conditional edges.
    """
    # The return type hint needs to include all possible string outputs from entry_router
    return entry_router(state)


# --- Graph Definition ---

def create_chatbot_graph() -> StateGraph:
    """Creates and configures the LangGraph StateGraph."""

    # Define the state schema using our ConversationState Pydantic model
    # Note: LangGraph often uses TypedDicts, but Pydantic/Dataclasses can work.
    # Adjust if specific LangGraph features require TypedDict.
    graph = StateGraph(ConversationState)

    # --- Add Nodes ---
    graph.add_node("greet", greeting_node)
    graph.add_node("call_llm", call_llm_node)
    # Use the execute_tool_node directly from tool_integration module
    graph.add_node("execute_tool", execute_tool_node)
    graph.add_node("prepare_response", prepare_response_node)
    graph.add_node("handle_error", error_node)
    # Use the module-level router node function
    graph.add_node("agent", agent_router_node)

    # --- Define Edges ---

    # Set entry point to the agent router
    graph.set_entry_point("agent")

    # Conditional Edges from the agent router node based on its return value
    # The path function now directly uses the output of agent_router_node
    graph.add_conditional_edges(
        source="agent",
        path=lambda state: entry_router(state), # Call the router directly here
        path_map={
            "greet": "greet",
            "call_llm": "call_llm",
            "execute_tool": "execute_tool",
            "handle_error": "handle_error",
            "end_conversation": END, # Map the specific decision string to END
            "wait_for_user": END      # Map the specific decision string to END
        }
    )

    # --- Other Edges ---
    # After greeting, usually let the LLM respond to the user's initial query that triggered greet
    graph.add_edge("greet", "call_llm")

    # After calling LLM, prepare the response
    graph.add_edge("call_llm", "prepare_response")

    # After executing a tool, call LLM to interpret results
    graph.add_edge("execute_tool", "call_llm")

    # After preparing the response, the current turn ends. Wait for next user input.
    graph.add_edge("prepare_response", END)

    # If error handling occurs, end the current turn.
    graph.add_edge("handle_error", END)


    # Compile the graph
    # Add memory for checkpointing
    checkpointer = MemorySaver()
    runnable_graph = graph.compile(checkpointer=checkpointer)
    logging.info("Chatbot graph created and compiled.")
    return runnable_graph


# --- Example Usage ---
async def run_example():
    print("\n--- Running Graph Example ---")

    if not llm_integrator:
        print("Skipping graph example: LLM Integrator not available (check OPENAI_API_KEY).")
        return

    graph_app = create_chatbot_graph()
    state_manager = StateManager() # Manages persistence across runs

    # --- Conversation Turn 1 ---
    print("\n--- Turn 1: Greeting ---")
    user_input_1 = "Hello"
    conversation_id = "graph-example-1"

    # Check if state exists, otherwise create
    current_state_obj = state_manager.get_state(conversation_id)
    if not current_state_obj:
        user_info = UserInfo(user_id="example-user-123", name="Graph Tester")
        initial_state = state_manager.create_new_state(conversation_id, user_info)
    else:
        initial_state = current_state_obj # Already a ConversationState object

    # Add user input to state before running graph
    initial_state.add_user_message(user_input_1)
    state_manager.save_state(initial_state) # Save updated state before run

    # Configuration for the graph run
    config = {"configurable": {"thread_id": conversation_id}}

    # Invoke the graph
    async for event in graph_app.astream(initial_state.to_dict(), config=config):
        # Print events for debugging/visibility
        event_type = list(event.keys())[0]
        event_data = event[event_type]
        print(f"Event: {event_type}")
        if event_type == END: # Capture the final state when END is reached
            break # Stop processing events for this turn

    # Load the final state using the state manager which handles persistence via checkpointer
    final_state = state_manager.get_state(conversation_id)

    print("\n--- Conversation History (Turn 1) ---")
    if final_state:
        for msg in final_state.get_messages():
            print(f"- {msg.sender}: {msg.content}")
    else:
        print("Could not retrieve final state.")


    # --- Conversation Turn 2 ---
    print("\n--- Turn 2: Tool Use ---")
    user_input_2 = "calculate: 5 + 3 * 2"

    # Get the latest state
    turn_2_initial_state = state_manager.get_state(conversation_id)
    if not turn_2_initial_state:
        print("Error: Could not retrieve state for Turn 2.")
        return

    # Add user input
    turn_2_initial_state.add_user_message(user_input_2)

    # Simulate an NLU step or prior node identifying the tool and args
    # In a real app, this identification would be part of the graph logic (e.g., an LLM call or regex node)
    # Here, we manually set it based on the user input for demonstration
    if user_input_2.startswith("calculate:"):
        expression = user_input_2.split("calculate:", 1)[1].strip()
        turn_2_initial_state.set_intermediate_result("tool_to_run", "simple_calculator")
        turn_2_initial_state.set_intermediate_result("tool_args", {"expression": expression})
    # Add similar logic for other tools if needed

    state_manager.save_state(turn_2_initial_state) # Save before run


    # Invoke the graph again with the updated state
    async for event in graph_app.astream(turn_2_initial_state.to_dict(), config=config):
         event_type = list(event.keys())[0]
         print(f"Event: {event_type}")
         if event_type == END:
              break # Stop after reaching the end of the turn

    final_state_turn_2 = state_manager.get_state(conversation_id)

    print("\n--- Conversation History (Turn 2) ---")
    if final_state_turn_2:
        for msg in final_state_turn_2.get_messages():
            print(f"- {msg.sender}: {msg.content}")
    else:
        print("Could not retrieve final state for Turn 2.")


if __name__ == "__main__":
    import asyncio
    try:
        asyncio.run(run_example())
    except ImportError as e:
        print(f"Example failed due to missing dependency: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during the example run: {e}", exc_info=True)
