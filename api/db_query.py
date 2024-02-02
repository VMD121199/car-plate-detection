# auth.py
from passlib.hash import pbkdf2_sha256
from db import create_connection
from psycopg2 import sql
import pandas as pd


def create_users_table(conn, table_name="users"):
    cursor = conn.cursor()
    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        id SERIAL PRIMARY KEY,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    )
"""
    cursor.execute(create_table_query)
    conn.commit()
    cursor.close()


def insert_user(conn, email, password):
    cursor = conn.cursor()
    insert_query = f"""
    INSERT INTO users (email, password)
    VALUES (%s, %s)
"""
    cursor.execute(insert_query, (email, password))
    conn.commit()
    cursor.close()


def get_user_by_email(conn, email):
    cursor = conn.cursor()
    get_user = f"""
    SELECT * FROM users
    WHERE email = %s
"""
    cursor.execute(get_user, (email,))
    user = cursor.fetchone()
    cursor.close()
    return user


def create_plate_detection_table(conn, table_name="plate_detection"):
    cursor = conn.cursor()
    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        id SERIAL PRIMARY KEY,
        x_min DOUBLE PRECISION NOT NULL,
        y_min DOUBLE PRECISION NOT NULL,
        x_max DOUBLE PRECISION NOT NULL,
        y_max DOUBLE PRECISION NOT NULL,
        license_text TEXT NOT NULL,
        bbox_score DOUBLE PRECISION NOT NULL,
        text_score DOUBLE PRECISION NOT NULL,
        user_detect TEXT
    )
"""
    cursor.execute(create_table_query)
    conn.commit()
    cursor.close()


def insert_detection(conn, detection):
    cursor = conn.cursor()
    xmin, ymin, xmax, ymax, license_text, bbox_score, text_score, region = (
        detection
    )
    insert_query = f"""
    INSERT INTO plate_detection (x_min, y_min, x_max, y_max, license_text, bbox_score, text_score, region)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
"""
    cursor.execute(
        insert_query,
        (xmin, ymin, xmax, ymax, license_text, bbox_score, text_score, region),
    )
    conn.commit()
    cursor.close()


def get_data(conn):
    cursor = conn.cursor()
    get_data = f"""
    SELECT * FROM plate_detection
"""
    cursor.execute(get_data)
    df = pd.read_sql(get_data, conn)
    return df
