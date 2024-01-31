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
    xmin, ymin, xmax, ymax, license_text, bbox_score, text_score = detection
    insert_query = f"""
    INSERT INTO plate_detection (x_min, y_min, x_max, y_max, license_text, bbox_score, text_score)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
"""
    cursor.execute(
        insert_query,
        (xmin, ymin, xmax, ymax, license_text, bbox_score, text_score),
    )
    conn.commit()
    cursor.close()
