# websocket_handlers.py

import logging
from flask import request
from flask_socketio import SocketIO, emit, join_room, leave_room, disconnect
from flask_jwt_extended import decode_token # To verify token on connect

# Import necessary components from other modules
from app import socketio # Import the initialized SocketIO instance
# Import Chatbot Integration Layer (we'll create this next)
from chatbot_integration import ChatbotBridge
# Import State Manager to manage conversation states
from state_management import StateManager, ConversationState, UserInfo
# Import Config Manager for JWT secret key
from config_manager import ConfigManager

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Initialize dependencies ---
# These should ideally be managed via dependency injection or app context,
# but for simplicity, we initialize them here.
state_manager = StateManager() # Manages conversation persistence
chatbot_bridge = ChatbotBridge() # Handles interaction with LangGraph app
config = ConfigManager()
JWT_SECRET_KEY = config.get('JWT_SECRET_KEY', 'dev-jwt-secret-key-replace-me')


# --- SocketIO Event Handlers ---

@socketio.on('connect')
def handle_connect(auth):
    """Handles new client connections."""
    # Authentication: Verify JWT token provided during connection
    token = auth.get('token') if isinstance(auth, dict) else None
    user_id = None
    if not token:
        logging.warning(f"Connection attempt without token from sid: {request.sid}. Disconnecting.")
        # disconnect(request.sid) # Enforce authentication - uncomment if needed
        # return False # Indicate connection failure
        logging.warning(f"Accepting unauthenticated connection for sid: {request.sid}. User ID set to None.")

    if token:
        try:
            # Decode the token to verify its validity and get the identity (user_id)
            decoded_token = decode_token(token, secret=JWT_SECRET_KEY, verify=True)
            user_id = decoded_token['sub'] # Assumes 'sub' contains the user_id
            logging.info(f"Client connected: sid={request.sid}, user_id={user_id}")
        except Exception as e:
            logging.error(f"Invalid token provided by sid {request.sid}: {e}. Disconnecting.")
            disconnect(request.sid)
            return False # Indicate connection failure

    # Store user_id in session for this connection (optional but useful)
    # Note: Flask-SocketIO session is different from Flask session
    # Use flask.session if needed across HTTP requests
    # from flask import session
    # session['user_id'] = user_id
    # Or use connection-specific storage if Flask-SocketIO provides it

    # Example: Join a room based on user ID or session ID
    # This allows targeting messages specifically later if needed
    join_room(request.sid) # Each client gets their own room by default using sid
    if user_id:
         join_room(f"user_{user_id}") # Optional: User-specific room

    emit('connection_ack', {'message': 'Connected successfully!', 'sid': request.sid})


@socketio.on('disconnect')
def handle_disconnect():
    """Handles client disconnections."""
    # Clean up resources associated with the session ID if necessary
    # E.g., leave rooms (though SocketIO might handle this automatically)
    # leave_room(request.sid)
    logging.info(f"Client disconnected: sid={request.sid}")


@socketio.on('send_message')
def handle_send_message(data):
    """
    Handles incoming messages from a client.
    Processes the message using the Chatbot Bridge and streams responses back.
    """
    sid = request.sid
    if not isinstance(data, dict):
        logging.warning(f"Received invalid message data format from {sid}: {data}")
        emit('error', {'message': 'Invalid message format.'}, room=sid)
        return

    user_message = data.get('message')
    conversation_id = data.get('conversation_id') # Client should track and send this

    if not user_message or not conversation_id:
        logging.warning(f"Missing 'message' or 'conversation_id' from {sid}: {data}")
        emit('error', {'message': 'Missing message or conversation ID.'}, room=sid)
        return

    logging.info(f"Received message from {sid} for conversation {conversation_id}: '{user_message[:50]}...'")

    # --- Interaction with Chatbot Bridge ---
    try:
        # Start processing the message via the bridge
        # The bridge should handle state loading/saving and invoking the graph
        # It should yield partial responses for streaming
        logging.debug(f"Passing message to ChatbotBridge for conversation {conversation_id}")

        # Use a generator or async generator returned by the bridge
        # to stream responses back to the *specific* client (using sid)
        # Note: socketio.start_background_task is needed if process_message is async
        # and you want to avoid blocking the main SocketIO event loop.
        def background_task(sid, conversation_id, user_message):
             try:
                 # Process and yield responses
                 for partial_response in chatbot_bridge.process_message_stream(conversation_id, user_message, sid):
                     logging.debug(f"Streaming partial response to {sid}: {partial_response}")
                     socketio.emit('partial_response', {'message': partial_response}, room=sid)

                 # Signal completion after stream finishes
                 socketio.emit('response_complete', {'message': 'Processing complete.'}, room=sid)
                 logging.info(f"Finished streaming response to {sid} for conversation {conversation_id}")

             except Exception as e:
                  logging.error(f"Error during background message processing for {sid}: {e}", exc_info=True)
                  # Notify client of the error
                  socketio.emit('error', {'message': f"An error occurred: {e}"}, room=sid)

        socketio.start_background_task(background_task, sid, conversation_id, user_message)

    except Exception as e:
        logging.error(f"Failed to initiate message processing for {sid}: {e}", exc_info=True)
        emit('error', {'message': f'Failed to process message: {e}'}, room=sid)


def register_socketio_events(socket_instance: SocketIO):
    """
    Placeholder function to confirm registration.
    The handlers are registered implicitly by the @socketio.on decorators
    when this module is imported, provided the `socketio` instance is the same
    as the one used in `app.py`.
    """
    logging.info("WebSocket event handlers registered implicitly.") 