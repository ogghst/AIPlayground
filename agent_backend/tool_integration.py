# tool_integration.py

import logging
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Type

# Assuming state_management is available
from state_management import ConversationState

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Tool Definition ---

class ToolError(Exception):
    """Custom exception for errors occurring during tool execution."""
    pass

class BaseTool(ABC):
    """Abstract base class for all tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """A unique name for the tool (used for registration and invocation)."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """A clear description of what the tool does, used for selection/understanding."""
        pass

    @abstractmethod
    async def execute(self, state: ConversationState, **kwargs) -> Any:
        """
        Executes the tool's logic.

        Args:
            state: The current ConversationState, providing context.
            **kwargs: Arguments required by the specific tool, often extracted
                      from the user query or state.

        Returns:
            The result of the tool's execution. This could be a string, dict,
            or any serializable type.

        Raises:
            ToolError: If the tool execution fails for a known reason.
            Exception: For unexpected errors during execution.
        """
        pass

# --- Example Tool Implementations ---

class SimpleCalculatorTool(BaseTool):
    """A basic tool that evaluates simple arithmetic expressions."""

    @property
    def name(self) -> str:
        return "simple_calculator"

    @property
    def description(self) -> str:
        return "Evaluates simple arithmetic expressions like '2 + 2' or '10 * (4 / 2)'. Use for basic math calculations."

    async def execute(self, state: ConversationState, expression: str) -> str:
        """
        Executes the calculator tool.

        Args:
            state: The current ConversationState (unused in this simple example).
            expression: The arithmetic expression string to evaluate.

        Returns:
            The result of the calculation as a string, or an error message.
        """
        logging.info(f"Executing {self.name} with expression: {expression}")
        try:
            # WARNING: eval() is unsafe with untrusted input.
            # In a real application, use a safer math expression parser (e.g., asteval, numexpr).
            # For this example, we proceed with caution assuming controlled input.
            allowed_chars = "0123456789+-*/(). "
            if not all(c in allowed_chars for c in expression):
                 raise ValueError("Expression contains invalid characters.")

            result = eval(expression, {"__builtins__": {}}, {}) # Basic safety attempt
            logging.debug(f"Calculation result: {result}")
            return f"The result of '{expression}' is {result}"
        except Exception as e:
            logging.error(f"Error executing {self.name} with expression '{expression}': {e}", exc_info=True)
            raise ToolError(f"Failed to calculate '{expression}': {e}")

class WebSearchTool(BaseTool):
    """A dummy tool simulating a web search."""
    # In a real scenario, this would use libraries like 'requests', 'beautifulsoup4',
    # or APIs like SerpAPI, Google Search API, etc.

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Performs a web search for a given query and returns a summary of results. Use for finding current information, facts, or general knowledge."

    async def execute(self, state: ConversationState, query: str) -> str:
        """
        Simulates executing a web search.

        Args:
            state: The current ConversationState.
            query: The search query string.

        Returns:
            A simulated summary of search results.
        """
        logging.info(f"Executing {self.name} with query: {query}")
        # Simulate API call delay
        await asyncio.sleep(0.5)

        # --- Placeholder for actual web search logic ---
        # Example:
        # try:
        #   search_results = await some_search_api_client.search(query)
        #   summary = self._summarize_results(search_results)
        #   return summary
        # except Exception as e:
        #   logging.error(f"Web search failed for query '{query}': {e}")
        #   raise ToolError(f"Web search failed: {e}")
        # --- End Placeholder ---

        # Dummy response
        return f"Simulated search results for '{query}': Found several relevant articles discussing the topic. The top result mentions..."


# --- Tool Registry and Executor ---

class ToolExecutor:
    """
    Manages registered tools and handles their execution.
    Acts as the central point for using tools within the application/graph.
    """

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        logging.info("ToolExecutor initialized.")

    def register_tool(self, tool_instance: BaseTool):
        """
        Registers a tool instance.

        Args:
            tool_instance: An instance of a class derived from BaseTool.

        Raises:
            ValueError: If a tool with the same name is already registered.
        """
        if not isinstance(tool_instance, BaseTool):
            raise TypeError("Can only register instances derived from BaseTool.")

        if tool_instance.name in self._tools:
            logging.warning(f"Attempted to register tool with duplicate name: {tool_instance.name}")
            raise ValueError(f"Tool name '{tool_instance.name}' already registered.")

        self._tools[tool_instance.name] = tool_instance
        logging.info(f"Registered tool: {tool_instance.name} - {tool_instance.description[:50]}...")

    def register_tools(self, tool_classes: List[Type[BaseTool]]):
        """
        Instantiates and registers multiple tools from their classes.

        Args:
            tool_classes: A list of classes derived from BaseTool.
        """
        for tool_cls in tool_classes:
            try:
                instance = tool_cls()
                self.register_tool(instance)
            except Exception as e:
                logging.error(f"Failed to instantiate or register tool {tool_cls.__name__}: {e}", exc_info=True)


    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Retrieves a registered tool by its name."""
        return self._tools.get(name)

    def list_tools(self) -> List[Dict[str, str]]:
        """Returns a list of registered tools with their names and descriptions."""
        return [{"name": tool.name, "description": tool.description} for tool in self._tools.values()]

    async def run_tool(self, tool_name: str, state: ConversationState, **kwargs) -> Any:
        """
        Finds and executes a registered tool by name.

        Args:
            tool_name: The name of the tool to execute.
            state: The current ConversationState.
            **kwargs: Arguments to pass to the tool's execute method.

        Returns:
            The result from the tool's execute method.

        Raises:
            ValueError: If the tool is not found.
            ToolError: If the tool execution fails with a known tool error.
            Exception: For other unexpected errors during execution.
        """
        logging.info(f"Attempting to run tool '{tool_name}' for conversation {state.conversation_id}")
        tool = self.get_tool(tool_name)
        if not tool:
            logging.error(f"Tool '{tool_name}' not found.")
            raise ValueError(f"Tool '{tool_name}' is not registered.")

        try:
            result = await tool.execute(state, **kwargs)
            logging.info(f"Tool '{tool_name}' executed successfully.")
            return result
        except ToolError as te:
            # Tool-specific, potentially recoverable error
            logging.error(f"Tool '{tool_name}' failed with ToolError: {te}", exc_info=True)
            state.error_info = {"step": f"tool_{tool_name}", "error": str(te)}
            raise # Re-raise ToolError so the calling node can potentially handle it
        except Exception as e:
            # Unexpected error during tool execution
            logging.error(f"Unexpected error during execution of tool '{tool_name}': {e}", exc_info=True)
            state.error_info = {"step": f"tool_{tool_name}", "error": f"Unexpected error: {e}"}
            # Wrap unexpected errors in a ToolError or let them propagate
            raise ToolError(f"An unexpected error occurred in tool '{tool_name}': {e}") from e


# --- Node Function for LangGraph ---

async def execute_tool_node(state: ConversationState) -> Dict[str, Any]:
    """
    A potential LangGraph node function that executes a tool based on state.

    Assumes the state contains information about which tool to run and its arguments,
    perhaps placed there by a routing or planning node. For example:
    state.intermediate_results['tool_to_run'] = 'web_search'
    state.intermediate_results['tool_args'] = {'query': 'latest AI news'}

    Args:
        state: The current ConversationState.

    Returns:
        A dictionary containing updates to the state, including the tool result
        or error information.
    """
    logging.debug(f"Entering execute_tool_node for conversation: {state.conversation_id}")
    tool_name = state.get_intermediate_result("tool_to_run")
    tool_args = state.get_intermediate_result("tool_args", {})

    if not tool_name or not isinstance(tool_name, str):
        logging.warning("No valid tool name found in state for execute_tool_node.")
        state.error_info = {"step": "execute_tool_node", "error": "Tool name not specified in state"}
        state.add_message("system", "Error: Could not determine which tool to run.") # System message about internal state
        return {"error_info": state.error_info, "history": state.history}

    # Assume a ToolExecutor instance is available (e.g., globally or passed via context)
    # In a real app, you'd need to manage the lifecycle/scope of tool_executor
    # For simplicity here, we create one ad-hoc with example tools
    tool_executor = ToolExecutor()
    tool_executor.register_tools([SimpleCalculatorTool, WebSearchTool]) # Register examples

    try:
        result = await tool_executor.run_tool(tool_name, state, **tool_args)
        logging.info(f"Tool '{tool_name}' node executed successfully, result: {str(result)[:100]}...")

        # Add the tool result to the conversation history (important for the LLM context)
        # Use a specific 'tool' sender type
        state.add_tool_message(tool_name=tool_name, content=str(result)) # Convert result to string for history

        # Optionally store the raw result elsewhere if needed (e.g., if not a simple string)
        state.set_intermediate_result(f"{tool_name}_result", result)

        # Clear the tool request fields if desired
        state.intermediate_results.pop("tool_to_run", None)
        state.intermediate_results.pop("tool_args", None)

        return {"history": state.history, "intermediate_results": state.intermediate_results, "error_info": None}

    except (ValueError, ToolError) as e:
        # Handle errors reported by run_tool (tool not found or execution failed)
        logging.error(f"Error in execute_tool_node running tool '{tool_name}': {e}")
        # Error info should already be set in state by run_tool
        # Add a system message indicating failure
        state.add_message("system", f"Error executing tool '{tool_name}': {e}")
        return {"history": state.history, "error_info": state.error_info}


# --- Example Usage (Illustrative) ---
async def main():
    # Setup
    tool_executor = ToolExecutor()
    tool_executor.register_tools([SimpleCalculatorTool, WebSearchTool])
    print("Registered Tools:", tool_executor.list_tools())

    # --- Test Calculator ---
    print("\n--- Testing Calculator ---")
    calc_state = ConversationState(conversation_id="tool-test-calc")
    try:
        result = await tool_executor.run_tool("simple_calculator", calc_state, expression="5 * (10 + 2)")
        print(f"Calculator Result: {result}")
    except Exception as e:
        print(f"Calculator Error: {e}")

    try:
        result = await tool_executor.run_tool("simple_calculator", calc_state, expression="5 / 0") # Expect error
        print(f"Calculator Result (Div by Zero): {result}")
    except Exception as e:
        print(f"Calculator Error (Div by Zero): {e}")

    # --- Test Search ---
    print("\n--- Testing Search ---")
    search_state = ConversationState(conversation_id="tool-test-search")
    try:
        result = await tool_executor.run_tool("web_search", search_state, query="What is LangGraph?")
        print(f"Search Result: {result}")
    except Exception as e:
        print(f"Search Error: {e}")

    # --- Test Node Function ---
    print("\n--- Testing Node Function ---")
    node_state = ConversationState(conversation_id="tool-test-node")
    node_state.set_intermediate_result("tool_to_run", "web_search")
    node_state.set_intermediate_result("tool_args", {"query": "Python asyncio best practices"})

    print(f"Initial state intermediate results: {node_state.intermediate_results}")
    updated_state_dict = await execute_tool_node(node_state) # Modifies node_state directly
    print("\nState after node execution:")
    for msg in node_state.get_messages(): # Access modified state directly
         print(f"- {msg.sender}: {msg.content}")
    print(f"Final state intermediate results: {node_state.intermediate_results}") # Should be cleaned up
    print(f"Error info: {node_state.error_info}")


if __name__ == "__main__":
    asyncio.run(main())
