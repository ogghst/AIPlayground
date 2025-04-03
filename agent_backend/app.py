# app.py

import logging
import os
from flask import Flask, jsonify # Added jsonify
from flask_socketio import SocketIO # Import SocketIO
from flask_caching import Cache # Import Cache

# Import configuration and module initializers
from config_manager import ConfigManager
from database import init_db, db # Import db object for convenience if needed elsewhere
from auth import init_auth, jwt # Import jwt for user lookup

# Import services (needed for user lookup)
from services import user_service

# Import Blueprints
from controllers.auth_controller import auth_bp
# from controllers.conversation_controller import conversation_bp # Example for later
# from websocket_handlers import register_socketio_events # Example for later

# *** Import WebSocket handlers ***
import websocket_handlers # Import the module to ensure decorators run

# Configure basic logging (can be refined using ConfigManager later)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize extensions that don't need app context immediately
# SocketIO needs to be initialized here or within create_app
# Use eventlet for async mode, suitable for SocketIO and background tasks
socketio = SocketIO(async_mode='eventlet', cors_allowed_origins="*") # Allow CORS for SocketIO
cache = Cache()

def create_app():
    """Factory function to create and configure the Flask application."""
    app = Flask(__name__)
    logging.info("Creating Flask app...")

    # --- Load Configuration ---
    config = ConfigManager()
    app.config['SECRET_KEY'] = config.get('SECRET_KEY', 'dev-secret-key-replace-me') # Essential for sessions, JWT, etc.
    app.config['SQLALCHEMY_DATABASE_URI'] = config.get('DATABASE_URL', 'sqlite:///app.db')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['JWT_SECRET_KEY'] = config.get('JWT_SECRET_KEY', 'dev-jwt-secret-key-replace-me')
    # Add other JWT configs if needed (e.g., token expiry)
    # from datetime import timedelta
    # app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=1)

    # Cache configuration (example: simple in-memory cache)
    app.config['CACHE_TYPE'] = config.get('CACHE_TYPE', 'SimpleCache') # e.g., 'SimpleCache', 'RedisCache'
    app.config['CACHE_DEFAULT_TIMEOUT'] = config.get_int('CACHE_DEFAULT_TIMEOUT', 300) # Default timeout 5 mins
    if app.config['CACHE_TYPE'].lower() == 'rediscache':
         app.config['CACHE_REDIS_URL'] = config.get('CACHE_REDIS_URL', 'redis://localhost:6379/0')

    # Set logging level from config
    log_level_name = config.log_level
    log_level = getattr(logging, log_level_name, logging.INFO)
    logging.getLogger().setLevel(log_level) # Set root logger level
    logging.info(f"Flask app log level set to: {log_level_name}")


    # --- Initialize Extensions ---
    init_db(app)
    init_auth(app)
    socketio.init_app(app) # Initialize SocketIO with the app
    cache.init_app(app) # Initialize Cache with the app
    logging.info("Flask extensions initialized (DB, Auth, SocketIO, Cache).")

    # --- Configure JWT User Loading ---
    # Register a callback function that takes whatever object is passed in as the
    # identity when creating JWTs and converts it to a JSON serializable format.
    @jwt.user_identity_loader
    def user_identity_lookup(user_id):
        # The identity is already the user ID we stored
        return user_id

    # Register a callback function that loads a user from your database whenever
    # a protected route is accessed. This should return an object based
    # on the identity stored in the JWT, or None if the user needs to log in.
    @jwt.user_lookup_loader
    def user_lookup_callback(_jwt_header, jwt_data):
        identity = jwt_data["sub"] # "sub" is where the identity is stored
        user = user_service.find_user_by_id(identity) # Need to add find_user_by_id to user_service
        if user is None:
            logging.warning(f"User lookup failed for identity: {identity}")
        return user

    # --- Error Handlers for JWT --- (Optional but Recommended)
    @jwt.expired_token_loader
    def expired_token_callback(jwt_header, jwt_payload):
        return jsonify(message="Token has expired"), 401

    @jwt.invalid_token_loader
    def invalid_token_callback(error_string):
        return jsonify(message=f"Invalid token: {error_string}"), 401

    @jwt.unauthorized_loader
    def missing_token_callback(error_string):
        return jsonify(message=f"Authorization required: {error_string}"), 401

    # --- Register Blueprints/Routes ---
    # Example: Create a simple root route here for testing
    @app.route('/')
    def index():
        logging.debug("Accessed root route /")
        return "Chatbot Backend is running!"

    # Register the authentication blueprint
    app.register_blueprint(auth_bp)
    # app.register_blueprint(conversation_bp) # Register others later

    # register_socketio_events(socketio) # Register SocketIO events later

    # *** Acknowledge SocketIO handler registration ***
    # No explicit call needed here as decorators handle it,
    # but logging confirms the module was imported.
    logging.info("WebSocket handlers registered via import.")

    logging.info("Flask app created successfully with blueprints.")
    return app

# --- Main Execution ---
if __name__ == '__main__':
    app = create_app()
    # Use SocketIO's run method which includes a WSGI server (like eventlet)
    port = int(os.environ.get('PORT', 5001)) # Use port 5001 for backend commonly
    logging.info(f"Starting SocketIO server on port {port}...")
    # Setting host='0.0.0.0' makes it accessible externally, use '127.0.0.1' for local only
    socketio.run(app, host='0.0.0.0', port=port, debug=True) # debug=True for development 