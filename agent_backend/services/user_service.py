# services/user_service.py

import logging
from typing import Optional

# Import database models and session
from database import db, User
# Import password hashing utilities
from auth import hash_password, check_password

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def create_user(username: str, email: str, password: str) -> Optional[User]:
    """
    Creates a new user in the database after hashing the password.

    Args:
        username: The desired username.
        email: The user's email address.
        password: The plain text password.

    Returns:
        The created User object if successful, None otherwise (e.g., if username/email exists).
    """
    # Check if username or email already exists
    if User.query.filter((User.username == username) | (User.email == email)).first():
        logging.warning(f"Attempted to register existing username '{username}' or email '{email}'")
        return None # Indicate failure due to existing user

    hashed_pw = hash_password(password)
    new_user = User(username=username, email=email, password_hash=hashed_pw)

    try:
        db.session.add(new_user)
        db.session.commit()
        logging.info(f"Successfully created user: {username}")
        return new_user
    except Exception as e:
        db.session.rollback()
        logging.error(f"Error creating user '{username}': {e}", exc_info=True)
        return None

def find_user_by_username(username: str) -> Optional[User]:
    """
    Finds a user by their username.

    Args:
        username: The username to search for.

    Returns:
        The User object if found, None otherwise.
    """
    logging.debug(f"Searching for user by username: {username}")
    return User.query.filter_by(username=username).first()

def find_user_by_id(user_id: int) -> Optional[User]:
    """
    Finds a user by their primary key ID.

    Args:
        user_id: The integer ID of the user.

    Returns:
        The User object if found, None otherwise.
    """
    logging.debug(f"Searching for user by ID: {user_id}")
    return User.query.get(user_id)

def authenticate_user(username: str, password: str) -> Optional[User]:
    """
    Authenticates a user by username and password.

    Args:
        username: The username.
        password: The plain text password.

    Returns:
        The User object if authentication is successful, None otherwise.
    """
    user = find_user_by_username(username)
    if user and check_password(user.password_hash, password):
        logging.info(f"Successfully authenticated user: {username}")
        return user
    else:
        logging.warning(f"Failed authentication attempt for username: {username}")
        return None 