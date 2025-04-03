import unittest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock

# Ensure agent_backend is discoverable (adjust path if needed)
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the components to be tested
from graph_definition import (
    greeting_node,
    call_llm_node,
    prepare_response_node,
    error_node,
    agent_router_node, # Use the router node directly
    DEFAULT_PROMPT_TEMPLATE
)
from state_management import ConversationState, Message, ConversationHistory, UserInfo

# Define a basic ConversationState for testing
def create_test_state(messages=None, error_info=None, intermediate_results=None):
    state = ConversationState(
        conversation_id="test_conv_id",
        user_info=UserInfo(user_id="test_user", name="Test User"),
        history=ConversationHistory(messages=messages or []),
        error_info=error_info or None,
        intermediate_results=intermediate_results or {}
    )
    return state

class TestGraphDefinitionNodes(unittest.TestCase):

    def test_greeting_node_calls_handler(self):
        """Verify greeting_node calls handle_greeting and returns history."""
        test_state = create_test_state()
        original_messages = test_state.history.messages.copy()  # Copy the messages list directly


        # Mock the function that modifies state
        with patch('graph_definition.handle_greeting') as mock_handle_greeting:
            # Simulate handle_greeting adding a message
            def side_effect(state):
                state.add_message("assistant", "Hello there!")
            mock_handle_greeting.side_effect = side_effect

            # Run the async node function
            result = asyncio.run(greeting_node(test_state))

            # Assertions
            mock_handle_greeting.assert_called_once_with(test_state)
            self.assertIn("history", result)
            self.assertEqual(len(result["history"].messages), 1) # Check if the message was added by side_effect
            self.assertEqual(result["history"].messages[0].content, "Hello there!")

    @patch('graph_definition.llm_integrator', new_callable=AsyncMock) # Mock the global instance
    def test_call_llm_node_success(self, mock_llm_integrator):
        """Test call_llm_node successful LLM call."""
        test_state = create_test_state(messages=[Message(sender="user", content="Tell me a joke")])
        mock_llm_integrator.generate_response = AsyncMock(return_value="Why did the scarecrow win an award? Because he was outstanding in his field!")

        # Run the node
        result = asyncio.run(call_llm_node(test_state))

        # Assertions
        mock_llm_integrator.generate_response.assert_awaited_once()
        # Check arguments passed to generate_response (simplified check)
        call_args, call_kwargs = mock_llm_integrator.generate_response.call_args
        self.assertEqual(call_kwargs['state'], test_state)
        self.assertTrue("Tell me a joke" in call_kwargs['prompt_template'])

        self.assertIn("intermediate_results", result)
        self.assertEqual(result["intermediate_results"]["raw_llm_response"], "Why did the scarecrow win an award? Because he was outstanding in his field!")
        self.assertIsNone(test_state.error_info) # No error info should be set

    @patch('graph_definition.llm_integrator', new_callable=AsyncMock)
    def test_call_llm_node_failure(self, mock_llm_integrator):
        """Test call_llm_node when LLM call fails."""
        test_state = create_test_state(messages=[Message(sender="user", content="Causes error")])
        error_message = "API rate limit exceeded"
        mock_llm_integrator.generate_response.side_effect = Exception(error_message)

        # Run the node
        result = asyncio.run(call_llm_node(test_state))

        # Assertions
        mock_llm_integrator.generate_response.assert_awaited_once()
        self.assertIsNotNone(result.get("error_info"))
        self.assertEqual(result["error_info"]["step"], "call_llm_node")
        self.assertTrue(error_message in result["error_info"]["error"])
        # Check if system error message was added
        self.assertEqual(len(result["history"].messages), 2) # User + System Error
        self.assertEqual(result["history"].messages[-1].sender, "system")
        self.assertTrue("Error generating response" in result["history"].messages[-1].content)

    @patch('graph_definition.llm_integrator', None) # Simulate integrator not initialized
    def test_call_llm_node_no_integrator(self):
        """Test call_llm_node when llm_integrator is None."""
        test_state = create_test_state(messages=[Message(sender="user", content="Hi")])

        # Run the node
        result = asyncio.run(call_llm_node(test_state))

        # Assertions
        self.assertIsNotNone(result.get("error_info"))
        self.assertEqual(result["error_info"]["step"], "call_llm_node")
        self.assertIn("LLM Integrator not initialized", result["error_info"]["error"])
        # Check if system error message was added
        self.assertEqual(len(result["history"].messages), 2) # User + System Error
        self.assertEqual(result["history"].messages[-1].sender, "system")
        self.assertTrue("Error: Cannot connect" in result["history"].messages[-1].content)


    def test_prepare_response_node_calls_handler(self):
        """Verify prepare_response_node calls prepare_final_response."""
        test_state = create_test_state()
        expected_updates = {"history": test_state.history, "intermediate_results": {}} # Example update

        with patch('graph_definition.prepare_final_response', return_value=expected_updates) as mock_prepare:
            result = asyncio.run(prepare_response_node(test_state))

            # Assertions
            mock_prepare.assert_called_once_with(test_state)
            self.assertEqual(result, expected_updates)

    def test_error_node_calls_handler(self):
        """Verify error_node calls handle_error and returns updates."""
        test_state = create_test_state(error_info={"step": "previous_step", "error": "Something failed"})

        with patch('graph_definition.handle_error') as mock_handle_error:
             # Simulate handle_error adding a message
            def side_effect(state):
                state.add_message("system", "An error occurred.")
                state.error_info = None # Simulate error being handled (cleared)
            mock_handle_error.side_effect = side_effect

            result = asyncio.run(error_node(test_state))

            # Assertions
            mock_handle_error.assert_called_once_with(test_state)
            self.assertIn("history", result)
            self.assertEqual(len(result["history"].messages), 1) # Check message added by side_effect
            self.assertEqual(result["history"].messages[0].content, "An error occurred.")
            self.assertIn("error_info", result)
            self.assertIsNone(result["error_info"]) # Error info should be cleared by side_effect

    @patch('graph_definition.route_message')
    def test_agent_router_node(self, mock_route_message):
        """Test the agent_router_node routes based on route_message output."""
        test_cases = [
            ("greet", "greet"),
            ("generate_response", "call_llm"),
            ("process_tool_request", "execute_tool"),
            ("end_conversation", "END"), # LangGraph's END sentinel
            ("clarify", "call_llm"),
            ("wait_for_user", "wait_for_user"),
            ("error", "handle_error"),
            ("unexpected_route", "handle_error"), # Default fallback
        ]

        for route_decision, expected_node in test_cases:
            with self.subTest(route_decision=route_decision, expected_node=expected_node):
                mock_route_message.return_value = route_decision
                test_state = create_test_state()

                # The router node itself is synchronous
                result_node = agent_router_node(test_state)

                # Assertions
                mock_route_message.assert_called_with(test_state) # Ensure route_message was called
                # Handle END comparison explicitly if needed, otherwise string compare
                if expected_node == "END":
                    # LangGraph uses a specific sentinel object for END
                    # Since we can't easily import it here, we rely on the string mapping logic
                    # in the graph definition's entry_router. The node returns the string.
                    self.assertEqual(result_node, "end_conversation") # Check if it returned the route name for END
                else:
                    self.assertEqual(result_node, expected_node)

    @patch('graph_definition.route_message')
    def test_agent_router_node_with_error_state(self, mock_route_message):
        """Test agent_router_node routes directly to handle_error if error_info is set."""
        test_state = create_test_state(error_info={"step": "some_step", "error": "pre-existing error"})

        # Run the router node
        result_node = agent_router_node(test_state)

        # Assertions
        self.assertEqual(result_node, "handle_error")
        mock_route_message.assert_not_called() # route_message shouldn't be called if error is present

if __name__ == '__main__':
    # Required for Windows compatibility with asyncio tests
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    unittest.main() 