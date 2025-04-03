# controllers/auth_controller.py

import logging
from flask import Blueprint, request, jsonify
from flask_restful import Api, Resource
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity

# Import user service functions
from services import user_service

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create Blueprint
auth_bp = Blueprint('auth_api', __name__, url_prefix='/api/auth')
api = Api(auth_bp)

class Register(Resource):
    """Handles user registration."""
    def post(self):
        data = request.get_json()
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')

        if not username or not email or not password:
            logging.warning("Registration attempt with missing fields.")
            return {"message": "Username, email, and password are required"}, 400

        # Basic validation (add more robust validation as needed)
        if len(password) < 6:
             return {"message": "Password must be at least 6 characters long"}, 400

        logging.info(f"Attempting registration for username: {username}")
        user = user_service.create_user(username, email, password)

        if user:
            logging.info(f"User '{username}' registered successfully.")
            # Optionally, log the user in immediately by creating a token
            # access_token = create_access_token(identity=user.id)
            # return {"message": "User registered successfully", "access_token": access_token}, 201
            return {"message": "User registered successfully. Please login."}, 201
        else:
            logging.warning(f"Registration failed for username: {username} (likely already exists).")
            return {"message": "Registration failed. Username or email may already exist."}, 409 # Conflict

class Login(Resource):
    """Handles user login and token generation."""
    def post(self):
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')

        if not username or not password:
            logging.warning("Login attempt with missing fields.")
            return {"message": "Username and password are required"}, 400

        logging.info(f"Attempting login for username: {username}")
        user = user_service.authenticate_user(username, password)

        if user:
            # Create JWT access token
            # The identity can be the user ID or any other unique identifier
            access_token = create_access_token(identity=user.id)
            logging.info(f"Login successful for user: {username} (ID: {user.id}). Token created.")
            return {"access_token": access_token}, 200
        else:
            # Keep error message generic to avoid revealing which field was wrong
            logging.warning(f"Login failed for username: {username}")
            return {"message": "Invalid credentials"}, 401 # Unauthorized

class ProtectedResource(Resource):
    """Example of a protected resource requiring JWT."""
    @jwt_required()
    def get(self):
        current_user_id = get_jwt_identity() # Get user ID from token
        # You can fetch user details from DB if needed using current_user_id
        logging.debug(f"Accessed protected resource by user ID: {current_user_id}")
        return {"message": f"Hello user {current_user_id}! This is a protected resource."}, 200


# Add resources to the API
api.add_resource(Register, '/register')
api.add_resource(Login, '/login')
api.add_resource(ProtectedResource, '/protected') # Example protected route 