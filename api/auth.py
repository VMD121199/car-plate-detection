# auth.py
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
