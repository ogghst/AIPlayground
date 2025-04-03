# database.py

import logging
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize extensions without app context initially
db = SQLAlchemy()
migrate = Migrate()

def init_db(app):
    """Initialize database extensions with the Flask app."""
    db.init_app(app)
    migrate.init_app(app, db)
    logging.info("Database extensions initialized.")

# --- Models ---
# Define models here or import them from a dedicated models module

class User(db.Model):
    """Basic User model for authentication."""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    created_at = db.Column(db.DateTime, server_default=db.func.now())

    def __repr__(self):
        return f'<User {self.username}>'

# Add other models as needed (e.g., Conversation, MessageLog, etc.)
# class Conversation(db.Model):
#    id = db.Column(db.String, primary_key=True) # Use the conversation_id
#    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True) # Link to user if logged in
#    created_at = db.Column(db.DateTime, server_default=db.func.now())
#    last_updated = db.Column(db.DateTime, onupdate=db.func.now())
#    # Potentially store the full state JSON here, or link to separate message logs
#    # full_state_json = db.Column(db.Text, nullable=True) 