# db.py
import psycopg2

def create_connection():
    conn = psycopg2.connect(
        dbname="car_plate_detection",
        user="postgres",
        password="backspace1",
        host="localhost",
        port="5433"
    )
    return conn
