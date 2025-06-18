import sqlite3
from datetime import datetime
from logger import db_logger, error_logger, PerformanceTimer
import os
import hashlib
import secrets

DB_NAME = "rag_app.db"
UPLOAD_DIR = "./uploaded_files"

# Initialize database
db_logger.info(f"Database path: {DB_NAME}")
if not os.path.exists(DB_NAME):
    db_logger.info("Creating new database")
else:
    db_logger.info(f"Using existing database: {DB_NAME}")


def get_db_connection():
    try:
        conn = sqlite3.connect(DB_NAME)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as e:
        error_msg = f"Failed to connect to database: {str(e)}"
        db_logger.error(error_msg)
        error_logger.error(error_msg, exc_info=True)
        raise


def check_column_exists(table_name, column_name):
    """Check if a column exists in a table."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        conn.close()

        for column in columns:
            if column['name'] == column_name:
                return True
        return False
    except Exception as e:
        error_msg = f"Failed to check column existence: {str(e)}"
        db_logger.error(error_msg)
        error_logger.error(error_msg, exc_info=True)
        return False


def add_column_if_not_exists(table_name, column_name, column_type):
    """Add a column to a table if it doesn't exist."""
    try:
        if not check_column_exists(table_name, column_name):
            conn = get_db_connection()
            conn.execute(
                f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
            conn.commit()
            conn.close()
            db_logger.info(f"Added column {column_name} to table {table_name}")
            return True
        return False
    except Exception as e:
        error_msg = f"Failed to add column {column_name} to table {table_name}: {str(e)}"
        db_logger.error(error_msg)
        error_logger.error(error_msg, exc_info=True)
        return False


def create_application_logs():
    with PerformanceTimer(db_logger, "create_application_logs"):
        try:
            conn = get_db_connection()
            conn.execute('''CREATE TABLE IF NOT EXISTS application_logs
                            (id INTEGER PRIMARY KEY AUTOINCREMENT,
                             session_id TEXT,
                             user_query TEXT,
                             gpt_response TEXT,
                             model TEXT,
                             created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
            conn.close()
            db_logger.info("Application logs table created or verified")

            # Add processing_time column if it doesn't exist
            add_column_if_not_exists(
                "application_logs", "processing_time", "REAL")

        except Exception as e:
            error_msg = f"Failed to create application logs table: {str(e)}"
            db_logger.error(error_msg)
            error_logger.error(error_msg, exc_info=True)
            raise


def insert_application_logs(session_id, question, answer, model, processing_time=0.0):
    with PerformanceTimer(db_logger, f"insert_logs:{session_id}"):
        try:
            # Ensure the processing_time column exists
            add_column_if_not_exists(
                "application_logs", "processing_time", "REAL")

            conn = get_db_connection()
            conn.execute('''INSERT INTO application_logs 
                            (session_id, user_query, gpt_response, model, processing_time) 
                            VALUES (?, ?, ?, ?, ?)''',
                         (session_id, question, answer, model, processing_time))
            conn.commit()
            conn.close()
            db_logger.info(f"Inserted log for session: {session_id}")
        except Exception as e:
            error_msg = f"Failed to insert application log: {str(e)}"
            db_logger.error(error_msg)
            error_logger.error(error_msg, exc_info=True)
            raise


def get_chat_history(session_id):
    with PerformanceTimer(db_logger, f"get_chat_history:{session_id}"):
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                'SELECT user_query, gpt_response FROM application_logs WHERE session_id = ? ORDER BY created_at', (session_id,))
            history = []
            for row in cursor.fetchall():
                history.append({
                    "question": row['user_query'],
                    "answer": row['gpt_response']
                })
            conn.close()
            db_logger.info(
                f"Retrieved {len(history)} messages for session {session_id}")
            return history
        except Exception as e:
            error_msg = f"Failed to retrieve chat history for session {session_id}: {str(e)}"
            db_logger.error(error_msg)
            error_logger.error(error_msg, exc_info=True)
            return []


def create_document_store():
    with PerformanceTimer(db_logger, "create_document_store"):
        try:
            conn = get_db_connection()
            conn.execute('''CREATE TABLE IF NOT EXISTS document_store
                            (id INTEGER PRIMARY KEY AUTOINCREMENT,
                             filename TEXT,
                             upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
            conn.close()
            db_logger.info("Document store table created or verified")
        except Exception as e:
            error_msg = f"Failed to create document store table: {str(e)}"
            db_logger.error(error_msg)
            error_logger.error(error_msg, exc_info=True)
            raise


def insert_document_record(filename):
    with PerformanceTimer(db_logger, f"insert_document:{filename}"):
        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            # Check if file already exists
            cursor.execute(
                'SELECT id FROM document_store WHERE filename = ?', (filename,))
            existing = cursor.fetchone()

            if existing:
                file_id = existing['id']
                # Update the timestamp to current time
                cursor.execute(
                    'UPDATE document_store SET upload_timestamp = CURRENT_TIMESTAMP WHERE id = ?', (file_id,))
                conn.commit()
                db_logger.info(
                    f"Document {filename} already exists with ID {file_id}, updated timestamp")
                conn.close()
                return file_id

            # If not exists, insert new record
            cursor.execute(
                'INSERT INTO document_store (filename) VALUES (?)', (filename,))
            file_id = cursor.lastrowid
            conn.commit()
            conn.close()
            db_logger.info(
                f"Inserted new document record: {filename} with ID {file_id}")
            return file_id
        except Exception as e:
            error_msg = f"Failed to insert document record for {filename}: {str(e)}"
            db_logger.error(error_msg)
            error_logger.error(error_msg, exc_info=True)
            raise


def delete_document_record(file_id):
    with PerformanceTimer(db_logger, f"delete_document:{file_id}"):
        try:
            conn = get_db_connection()
            conn.execute('DELETE FROM document_store WHERE id = ?', (file_id,))
            conn.commit()
            conn.close()
            db_logger.info(f"Deleted document record with ID {file_id}")
            return True
        except Exception as e:
            error_msg = f"Failed to delete document record {file_id}: {str(e)}"
            db_logger.error(error_msg)
            error_logger.error(error_msg, exc_info=True)
            return False


def get_all_documents():
    """Get all documents from the database."""
    with PerformanceTimer(db_logger, "get_all_documents"):
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, filename, upload_timestamp FROM document_store ORDER BY upload_timestamp DESC")
            documents = cursor.fetchall()
            conn.close()

            result = []
            for doc in documents:
                result.append({
                    "id": doc["id"],
                    "filename": doc["filename"],
                    "upload_timestamp": doc["upload_timestamp"]
                })
            return result
        except Exception as e:
            error_msg = f"Failed to get documents: {str(e)}"
            db_logger.error(error_msg)
            error_logger.error(error_msg, exc_info=True)
            return []


def get_document_path(file_id):
    """Get the path of a document by its ID."""
    with PerformanceTimer(db_logger, f"get_document_path:{file_id}"):
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT filename FROM document_store WHERE id = ?", (file_id,))
            document = cursor.fetchone()
            conn.close()

            if document:
                # First check if the file exists in the upload directory
                filename = document["filename"]
                upload_path = os.path.join(
                    UPLOAD_DIR, f"doc-{file_id}-{filename}")

                if os.path.exists(upload_path):
                    db_logger.info(
                        f"Found document in upload directory: {upload_path}")
                    return upload_path

                # Fall back to the old location if not found in upload directory
                old_path = os.path.join(
                    "./faiss_db/document_collection", filename)
                if os.path.exists(old_path):
                    db_logger.info(
                        f"Found document in old location: {old_path}")
                    return old_path

                db_logger.error(f"Document file not found for ID {file_id}")
                return None
            return None
        except Exception as e:
            error_msg = f"Failed to get document path: {str(e)}"
            db_logger.error(error_msg)
            error_logger.error(error_msg, exc_info=True)
            return None


def create_users_table():
    """Create users table if it doesn't exist."""
    with PerformanceTimer(db_logger, "create_users_table"):
        try:
            conn = get_db_connection()
            conn.execute('''CREATE TABLE IF NOT EXISTS users
                            (id INTEGER PRIMARY KEY AUTOINCREMENT,
                             username TEXT UNIQUE,
                             password_hash TEXT,
                             role TEXT,
                             created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
            conn.commit()
            conn.close()
            db_logger.info("Users table created or verified")

            # Create default admin and user accounts if they don't exist
            if not get_user_by_username("admin"):
                create_user("admin", "admin", "admin")
                db_logger.info("Created default admin user")

            if not get_user_by_username("user"):
                create_user("user", "user", "user")
                db_logger.info("Created default regular user")

        except Exception as e:
            error_msg = f"Failed to create users table: {str(e)}"
            db_logger.error(error_msg)
            error_logger.error(error_msg, exc_info=True)
            raise


def hash_password(password):
    """Hash a password for storing."""
    # In a real application, use a proper password hashing library like bcrypt
    # This is a simplified version for the CTF
    salt = secrets.token_hex(8)
    pwdhash = hashlib.sha256(salt.encode() + password.encode()).hexdigest()
    return f"{salt}${pwdhash}"


def verify_password(stored_password, provided_password):
    """Verify a stored password against provided password."""
    salt, stored_hash = stored_password.split('$')
    pwdhash = hashlib.sha256(
        salt.encode() + provided_password.encode()).hexdigest()
    return pwdhash == stored_hash


def create_user(username, password, role):
    """Create a new user in the database."""
    with PerformanceTimer(db_logger, f"create_user:{username}"):
        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            # Check if username already exists
            cursor.execute(
                'SELECT id FROM users WHERE username = ?', (username,))
            if cursor.fetchone():
                conn.close()
                return None, "Username already exists"

            # Hash the password
            password_hash = hash_password(password)

            # Insert the new user
            cursor.execute(
                'INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)',
                (username, password_hash, role)
            )
            user_id = cursor.lastrowid
            conn.commit()
            conn.close()

            db_logger.info(f"Created new user: {username} with role {role}")
            return user_id, None
        except Exception as e:
            error_msg = f"Failed to create user {username}: {str(e)}"
            db_logger.error(error_msg)
            error_logger.error(error_msg, exc_info=True)
            return None, str(e)


def authenticate_user(username, password):
    """Authenticate a user by username and password."""
    with PerformanceTimer(db_logger, f"authenticate_user:{username}"):
        try:
            user = get_user_by_username(username)
            if not user:
                return None

            if verify_password(user["password_hash"], password):
                return user

            return None
        except Exception as e:
            error_msg = f"Failed to authenticate user {username}: {str(e)}"
            db_logger.error(error_msg)
            error_logger.error(error_msg, exc_info=True)
            return None


def get_user_by_username(username):
    """Get a user by username."""
    with PerformanceTimer(db_logger, f"get_user_by_username:{username}"):
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                'SELECT id, username, password_hash, role FROM users WHERE username = ?',
                (username,)
            )
            user = cursor.fetchone()
            conn.close()

            if user:
                return dict(user)
            return None
        except Exception as e:
            error_msg = f"Failed to get user {username}: {str(e)}"
            db_logger.error(error_msg)
            error_logger.error(error_msg, exc_info=True)
            return None


def get_user_by_id(user_id):
    """Get a user by ID."""
    with PerformanceTimer(db_logger, f"get_user_by_id:{user_id}"):
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                'SELECT id, username, role FROM users WHERE id = ?',
                (user_id,)
            )
            user = cursor.fetchone()
            conn.close()

            if user:
                return dict(user)
            return None
        except Exception as e:
            error_msg = f"Failed to get user by ID {user_id}: {str(e)}"
            db_logger.error(error_msg)
            error_logger.error(error_msg, exc_info=True)
            return None


def modify_username(user_id, new_username):
    """Modify a user's username."""
    with PerformanceTimer(db_logger, f"modify_username:{user_id}"):
        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            # Check if new username already exists
            cursor.execute('SELECT id FROM users WHERE username = ? AND id != ?',
                           (new_username, user_id))
            if cursor.fetchone():
                conn.close()
                return False, "Username already exists"

            # Update the username
            cursor.execute(
                'UPDATE users SET username = ? WHERE id = ?',
                (new_username, user_id)
            )
            conn.commit()
            conn.close()

            if cursor.rowcount > 0:
                db_logger.info(
                    f"Modified username for user ID {user_id} to {new_username}")
                return True, None
            else:
                return False, "User not found"
        except Exception as e:
            error_msg = f"Failed to modify username for user ID {user_id}: {str(e)}"
            db_logger.error(error_msg)
            error_logger.error(error_msg, exc_info=True)
            return False, str(e)


def delete_user(user_id):
    """Delete a user by ID."""
    with PerformanceTimer(db_logger, f"delete_user:{user_id}"):
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute('DELETE FROM users WHERE id = ?', (user_id,))
            conn.commit()
            conn.close()

            if cursor.rowcount > 0:
                db_logger.info(f"Deleted user with ID {user_id}")
                return True, None
            else:
                return False, "User not found"
        except Exception as e:
            error_msg = f"Failed to delete user with ID {user_id}: {str(e)}"
            db_logger.error(error_msg)
            error_logger.error(error_msg, exc_info=True)
            return False, str(e)


def get_all_users():
    """
    Retrieve all users from the database.

    Returns:
        list: A list of dictionaries containing user information (id, username, role)
    """
    with PerformanceTimer(db_logger, "get_all_users"):
        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            # For security, we don't return passwords
            cursor.execute("SELECT id, username, role FROM users")
            users = [{"id": row[0], "username": row[1], "role": row[2]}
                     for row in cursor.fetchall()]

            return users
        except Exception as e:
            error_logger.error(
                f"Error retrieving all users: {str(e)}", exc_info=True)
            return []


# Initialize the database tables
try:
    create_application_logs()
    create_document_store()
    create_users_table()
    db_logger.info("Database tables initialized successfully")
except Exception as e:
    error_msg = f"Failed to initialize database tables: {str(e)}"
    db_logger.error(error_msg)
    error_logger.error(error_msg, exc_info=True)
