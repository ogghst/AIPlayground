# evaluation_monitoring.py

import logging
import time
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import datetime
import statistics

# Assuming state_management is available
from state_management import ConversationState, Message

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define a standard structure for logged events
@dataclass
class LogEvent:

    conversation_id: str
    event_type: str # e.g., 'user_input', 'llm_request', 'llm_response', 'tool_call', 'final_response', 'error'
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.utcnow)
    data: Dict[str, Any] = field(default_factory=dict) # Contextual data for the event
    duration_ms: Optional[float] = None # Optional duration for timed events

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "conversation_id": self.conversation_id,
            "event_type": self.event_type,
            "data": self.data,
            "duration_ms": self.duration_ms,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class EvaluationMonitor:
    """
    Handles logging conversation events and calculating basic performance metrics.
    """
    def __init__(self, log_file_path: Optional[str] = "conversation_log.jsonl"):
        self.log_file_path = log_file_path
        # In-memory storage for metrics (could be replaced by external monitoring system)
        self._response_times: List[float] = []
        self._error_counts: Dict[str, int] = {} # Count errors by type/step
        self._feedback_scores: List[int] = [] # e.g., 1 for thumbs up, -1 for thumbs down
        logging.info(f"EvaluationMonitor initialized. Logging to: {self.log_file_path or 'stdout'}")

    def _log_to_destination(self, event: LogEvent):
        """Logs the event to the configured destination (file or stdout)."""
        log_json = event.to_json()
        if self.log_file_path:
            try:
                with open(self.log_file_path, 'a', encoding='utf-8') as f:
                    f.write(log_json + '\n')
            except IOError as e:
                logging.error(f"Failed to write to log file {self.log_file_path}: {e}. Logging to stdout instead.")
                print(log_json) # Fallback to stdout
        else:
            print(log_json) # Log to stdout if no file path

    def log_event(
        self,
        conversation_id: str,
        event_type: str,
        data: Optional[Dict[str, Any]] = None,
        duration_ms: Optional[float] = None
    ):
        """Creates and logs a standard event."""
        event = LogEvent(
            conversation_id=conversation_id,
            event_type=event_type,
            data=data or {},
            duration_ms=duration_ms
        )
        logging.debug(f"Logging event: {event_type} for conversation {conversation_id}")
        self._log_to_destination(event)

        # --- Update internal metrics based on event ---
        if duration_ms is not None and event_type in ['llm_response', 'tool_call', 'final_response']:
            self._response_times.append(duration_ms)

        if event_type == 'error':
            error_step = data.get('step', 'unknown') if data else 'unknown'
            self._error_counts[error_step] = self._error_counts.get(error_step, 0) + 1

        if event_type == 'user_feedback':
            score = data.get('score') if data else None
            if isinstance(score, int):
                 self._feedback_scores.append(score)
        # --- End metrics update ---


    def log_turn(self, state: ConversationState, start_time: float, end_time: float):
        """Logs key information about a completed conversation turn."""
        conversation_id = state.conversation_id
        duration_ms = (end_time - start_time) * 1000
        logging.info(f"Logging turn for conversation {conversation_id}. Duration: {duration_ms:.2f} ms")

        # Log user input (assuming last input is relevant to this turn)
        user_messages = [msg for msg in state.history.messages if msg.sender != 'assistant' and msg.sender != 'tool' and msg.sender != 'system']
        last_user_message = user_messages[-1].content if user_messages else "N/A"
        self.log_event(conversation_id, "user_input_processed", {"input": last_user_message})

        # Log final assistant response(s) for this turn
        assistant_messages = [msg for msg in state.history.messages if msg.sender == 'assistant']
        last_assistant_message = assistant_messages[-1].content if assistant_messages else "N/A"
        self.log_event(
            conversation_id,
            "final_response_delivered",
            {"response": last_assistant_message},
            duration_ms=duration_ms # Log overall turn duration here
        )

        # Log error if one occurred during the turn
        if state.error_info:
            self.log_event(conversation_id, "error", state.error_info)


    def get_metrics(self) -> Dict[str, Any]:
        """Calculates and returns current performance metrics."""
        metrics = {}
        if self._response_times:
            metrics["average_response_time_ms"] = statistics.mean(self._response_times)
            metrics["median_response_time_ms"] = statistics.median(self._response_times)
            metrics["p95_response_time_ms"] = statistics.quantiles(self._response_times, n=100)[94] # Index 94 for 95th percentile
            metrics["response_time_count"] = len(self._response_times)
        else:
             metrics["average_response_time_ms"] = 0

        metrics["error_counts_by_step"] = self._error_counts
        metrics["total_errors"] = sum(self._error_counts.values())

        if self._feedback_scores:
             metrics["average_feedback_score"] = statistics.mean(self._feedback_scores)
             metrics["feedback_count"] = len(self._feedback_scores)
             # Example: Calculate satisfaction rate (e.g., % positive scores)
             positive_feedback = sum(1 for score in self._feedback_scores if score > 0)
             metrics["satisfaction_rate"] = (positive_feedback / len(self._feedback_scores)) * 100 if self._feedback_scores else 0
        else:
             metrics["average_feedback_score"] = None
             metrics["feedback_count"] = 0
             metrics["satisfaction_rate"] = None


        logging.info(f"Calculated metrics: {metrics}")
        return metrics

# Example Usage (Illustrative)
if __name__ == "__main__":
    monitor = EvaluationMonitor(log_file_path=None) # Log to stdout for example

    # Simulate some events
    conv_id = "eval-test-1"
    monitor.log_event(conv_id, "llm_request", {"model": "gpt-4o", "prompt_length": 150})
    time.sleep(0.1) # Simulate delay
    monitor.log_event(conv_id, "llm_response", {"response_length": 80}, duration_ms=110.5)

    monitor.log_event(conv_id, "tool_call", {"tool_name": "calculator"}, duration_ms=25.0)
    monitor.log_event(conv_id, "error", {"step": "tool_calculator", "error": "Division by zero"})

    monitor.log_event(conv_id, "user_feedback", {"score": 1}) # Thumbs up
    monitor.log_event(conv_id, "user_feedback", {"score": -1}) # Thumbs down

    # Simulate logging a turn
    sim_state = ConversationState(conversation_id=conv_id)
    sim_state.add_user_message("What is 2+2?")
    sim_state.add_assistant_message("2 + 2 equals 4.")
    monitor.log_turn(sim_state, start_time=time.time() - 0.5, end_time=time.time())

    # Get metrics
    print("\n--- Calculated Metrics ---")
    metrics = monitor.get_metrics()
    print(json.dumps(metrics, indent=2))