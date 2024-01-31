# db.py
import psycopg2

def create_connection():
    conn = psycopg2.connect(
        dbname="car_plate_detection",
        user="postgres",
        password="121199",
        host="localhost",
        port="5432"
    )
    return conn
