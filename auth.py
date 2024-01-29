# auth.py
from passlib.hash import pbkdf2_sha256
from db import create_connection

def create_users_table(conn):
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)
    conn.commit()
    cursor.close()

def insert_user(conn, email, password):
    cursor = conn.cursor()
    hashed_password = pbkdf2_sha256.hash(password)
    cursor.execute("INSERT INTO users (email, password) VALUES (%s, %s)", (email, hashed_password))
    conn.commit()
    cursor.close()

def get_user_by_email(conn, email):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
    user = cursor.fetchone()
    cursor.close()
    return user
