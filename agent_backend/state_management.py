from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import datetime
import json
import os
import logging
from pathlib import Path

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Data Structures for State ---

@dataclass
class Message:
    """Represents a single message in the conversation."""
    sender: str  # e.g., "user", "assistant", "system", "tool"
    content: str
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict) # For tool calls, IDs, etc.

    def to_dict(self) -> Dict[str, Any]:
        """Serializes Message to a dictionary."""
        return {
            "sender": self.sender,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Deserializes Message from a dictionary."""
        return cls(
            sender=data["sender"],
            content=data["content"],
            timestamp=datetime.datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
        )

@dataclass
class ConversationHistory:
    """Manages the sequence of messages in a conversation."""
    messages: List[Message] = field(default_factory=list)

    def add_message(self, message: Message):
        """Adds a message to the history."""
        self.messages.append(message)

    def get_last_n_messages(self, n: int) -> List[Message]:
        """Retrieves the last N messages."""
        return self.messages[-n:]

    def __len__(self) -> int:
        """Returns the number of messages in the history."""
        return len(self.messages)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes ConversationHistory to a dictionary."""
        return {"messages": [msg.to_dict() for msg in self.messages]}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationHistory':
        """Deserializes ConversationHistory from a dictionary."""
        return cls(messages=[Message.from_dict(msg_data) for msg_data in data.get("messages", [])])

@dataclass
class UserInfo:
    """Stores information about the user."""
    user_id: str
    name: Optional[str] = None
    preferences: Dict[str, Any] = field(default_factory=dict)
    # Add other relevant user details like email, location, etc. if needed

    def to_dict(self) -> Dict[str, Any]:
        """Serializes UserInfo to a dictionary."""
        return self.__dict__

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserInfo':
        """Deserializes UserInfo from a dictionary."""
        return cls(**data)

@dataclass
class Memory:
    """Represents the chatbot's memory for the conversation."""
    short_term: Dict[str, Any] = field(default_factory=dict) # Key-value store
    # Reference or identifier for accessing long-term vector storage / RAG context
    vector_store_context_id: Optional[str] = None
    # Could include conversation summaries, extracted entities, user profile hints
    extracted_entities: List[Dict[str, Any]] = field(default_factory=list)
    conversation_summary: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serializes Memory to a dictionary."""
        return self.__dict__

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Memory':
        """Deserializes Memory from a dictionary."""
        return cls(**data)

# --- Core State Class ---

@dataclass
class ConversationState:
    """
    Defines the complete state for a single conversation graph instance.
    This acts as the central data carrier between nodes in LangGraph.
    """
    conversation_id: str
    history: ConversationHistory = field(default_factory=ConversationHistory)
    user_info: Optional[UserInfo] = None # May not always have specific user info
    memory: Memory = field(default_factory=Memory)

    # LangGraph specific state elements for flow control and data passing
    current_node: Optional[str] = None # Tracks the current node executing in the graph
    intermediate_results: Dict[str, Any] = field(default_factory=dict) # Stores outputs from graph nodes
    error_info: Optional[Dict[str, Any]] = None # To store error details if a node fails
    requires_human_intervention: bool = False # Flag for human-in-the-loop scenarios

    def get_messages(self) -> List[Message]:
        """Returns all messages from the conversation history."""
        return self.history.messages

    def add_message(self, sender: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Adds a generic message to the history."""
        self.history.add_message(Message(sender=sender, content=content, metadata=metadata or {}))

    def add_user_message(self, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Adds a message from the user."""
        user_id = self.user_info.user_id if self.user_info else "user"
        self.add_message(sender=user_id, content=content, metadata=metadata)

    def add_assistant_message(self, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Adds a message from the assistant."""
        self.add_message(sender="assistant", content=content, metadata=metadata)

    def add_tool_message(self, tool_name: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Adds a message resulting from a tool execution."""
        meta = metadata or {}
        meta['tool_name'] = tool_name
        self.add_message(sender="tool", content=content, metadata=meta)

    def update_memory(self, key: str, value: Any):
        """Updates a value in the short-term memory."""
        self.memory.short_term[key] = value

    def get_from_memory(self, key: str, default: Any = None) -> Any:
        """Retrieves a value from the short-term memory."""
        return self.memory.short_term.get(key, default)

    def set_intermediate_result(self, node_name: str, result: Any):
        """Stores the output of a graph node."""
        self.intermediate_results[node_name] = result

    def get_intermediate_result(self, node_name: str, default: Any = None) -> Any:
        """Retrieves the output of a previously run graph node."""
        return self.intermediate_results.get(node_name, default)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the entire ConversationState to a dictionary."""
        return {
            "conversation_id": self.conversation_id,
            "history": self.history.to_dict(),
            "user_info": self.user_info.to_dict() if self.user_info else None,
            "memory": self.memory.to_dict(),
            "current_node": self.current_node,
            "intermediate_results": self.intermediate_results, # Assuming results are serializable
            "error_info": self.error_info,
            "requires_human_intervention": self.requires_human_intervention,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationState':
        """Deserializes ConversationState from a dictionary."""
        return cls(
            conversation_id=data["conversation_id"],
            history=ConversationHistory.from_dict(data.get("history", {})),
            user_info=UserInfo.from_dict(data["user_info"]) if data.get("user_info") else None,
            memory=Memory.from_dict(data.get("memory", {})),
            current_node=data.get("current_node"),
            intermediate_results=data.get("intermediate_results", {}),
            error_info=data.get("error_info"),
            requires_human_intervention=data.get("requires_human_intervention", False),
        )

# --- State Management ---

class StateManager:
    """
    Manages the lifecycle and persistence of ConversationState objects.
    Provides an interface to load, save, and retrieve conversation states.
    """

    def __init__(self, persistence_dir: str = "./conversation_states"):
        """
        Initializes the StateManager.

        Args:
            persistence_dir: The directory to store serialized conversation state files.
        """
        self._states: Dict[str, ConversationState] = {}
        self.persistence_path = Path(persistence_dir)
        self.persistence_path.mkdir(parents=True, exist_ok=True)
        self._load_all_states()
        logging.info(f"StateManager initialized. Loaded {len(self._states)} states from {self.persistence_path}")

    def _get_persistence_filepath(self, conversation_id: str) -> Path:
        """Gets the file path for persisting a specific conversation state."""
        # Sanitize conversation_id to prevent path traversal issues if necessary
        safe_filename = "".join(c for c in conversation_id if c.isalnum() or c in ('-', '_'))
        return self.persistence_path / f"{safe_filename}.json"

    def save_state(self, state: ConversationState):
        """
        Saves a conversation state to the in-memory store and persists it to a file.

        Args:
            state: The ConversationState object to save.
        """
        if not isinstance(state, ConversationState):
            raise TypeError("Can only save objects of type ConversationState")

        self._states[state.conversation_id] = state
        filepath = self._get_persistence_filepath(state.conversation_id)
        try:
            state_dict = state.to_dict()
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(state_dict, f, indent=2)
            logging.debug(f"Saved state for conversation {state.conversation_id} to {filepath}")
        except Exception as e:
            logging.error(f"Error saving state for conversation {state.conversation_id} to {filepath}: {e}", exc_info=True)
            # Depending on requirements, might re-raise or handle differently

    def get_state(self, conversation_id: str) -> Optional[ConversationState]:
        """
        Retrieves a conversation state. Checks in-memory cache first, then tries persistence.

        Args:
            conversation_id: The unique identifier for the conversation.

        Returns:
            The ConversationState object if found, otherwise None.
        """
        if conversation_id in self._states:
            logging.debug(f"Retrieved state for conversation {conversation_id} from memory.")
            return self._states[conversation_id]

        # If not in memory, try loading from persistence
        state = self._load_state_from_file(conversation_id)
        if state:
            self._states[conversation_id] = state # Cache it
            logging.debug(f"Loaded state for conversation {conversation_id} from file.")
        else:
             logging.warning(f"State for conversation {conversation_id} not found in memory or persistence.")
        return state

    def _load_state_from_file(self, conversation_id: str) -> Optional[ConversationState]:
        """Loads a single conversation state from its persistence file."""
        filepath = self._get_persistence_filepath(conversation_id)
        if filepath.exists():
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return ConversationState.from_dict(data)
            except json.JSONDecodeError as e:
                logging.error(f"Error decoding JSON for conversation {conversation_id} from {filepath}: {e}", exc_info=True)
            except Exception as e:
                logging.error(f"Error loading state for conversation {conversation_id} from {filepath}: {e}", exc_info=True)
                # Consider moving/archiving corrupted files
        return None

    def _load_all_states(self):
        """Loads all persisted states from the directory into the in-memory store on initialization."""
        loaded_count = 0
        for filepath in self.persistence_path.glob("*.json"):
            conversation_id_from_filename = filepath.stem # Assumes filename is conversation_id
            if conversation_id_from_filename not in self._states: # Avoid overwriting if already present
                state = self._load_state_from_file(conversation_id_from_filename)
                if state:
                    self._states[state.conversation_id] = state # Use ID from state data
                    loaded_count += 1
        logging.info(f"Attempted to load all states from {self.persistence_path}. Successfully loaded: {loaded_count}")


    def create_new_state(self, conversation_id: str, user_info: Optional[UserInfo] = None) -> ConversationState:
        """
        Creates a new, empty conversation state and persists it immediately.

        Args:
            conversation_id: The unique ID for the new conversation.
            user_info: Optional UserInfo object for the conversation.

        Returns:
            The newly created ConversationState object.

        Raises:
            ValueError: If a conversation state with the given ID already exists.
        """
        if self.get_state(conversation_id) is not None:
            logging.warning(f"Attempted to create state with existing ID: {conversation_id}")
            raise ValueError(f"Conversation state with ID '{conversation_id}' already exists.")

        new_state = ConversationState(conversation_id=conversation_id, user_info=user_info)
        self.save_state(new_state) # Persist immediately
        logging.info(f"Created and saved new state for conversation {conversation_id}")
        return new_state

    def delete_state(self, conversation_id: str):
        """
        Deletes a conversation state from the in-memory store and its persistence file.

        Args:
            conversation_id: The ID of the conversation state to delete.
        """
        if conversation_id in self._states:
            del self._states[conversation_id]
            logging.debug(f"Deleted state for conversation {conversation_id} from memory.")

        filepath = self._get_persistence_filepath(conversation_id)
        if filepath.exists():
            try:
                filepath.unlink()
                logging.info(f"Deleted persisted state file {filepath}")
            except Exception as e:
                logging.error(f"Error deleting persisted state file {filepath}: {e}", exc_info=True)
        else:
             logging.warning(f"Attempted to delete non-existent state file for conversation {conversation_id}")

    def list_conversation_ids(self) -> List[str]:
        """Returns a list of all conversation IDs currently managed (in memory or persisted)."""
        # Combine keys from memory and filenames from disk
        memory_ids = set(self._states.keys())
        persisted_ids = set(p.stem for p in self.persistence_path.glob("*.json"))
        return sorted(list(memory_ids.union(persisted_ids))) 