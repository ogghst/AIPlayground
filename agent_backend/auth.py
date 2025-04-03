# auth.py

import logging
from flask_jwt_extended import JWTManager
from passlib.context import CryptContext

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize JWT Manager
jwt = JWTManager()

# Initialize Password Hashing context
# Use a strong hashing algorithm like bcrypt
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def init_auth(app):
    """Initialize authentication extensions with the Flask app."""
    jwt.init_app(app)
    logging.info("JWTManager initialized.")
    # You might add user loader functions here for JWT
    # @jwt.user_identity_loader
    # def user_identity_lookup(user): ...
    # @jwt.user_lookup_loader
    # def user_lookup_callback(_jwt_header, jwt_data): ...


def hash_password(password: str) -> str:
    """Hashes a plain text password."""
    return pwd_context.hash(password)

def check_password(password_hash: str, plain_password: str) -> bool:
    """Verifies a plain text password against a stored hash."""
    return pwd_context.verify(plain_password, password_hash)

# Add JWT token generation/verification helpers if needed,
# although Flask-JWT-Extended provides decorators for most common use cases. 