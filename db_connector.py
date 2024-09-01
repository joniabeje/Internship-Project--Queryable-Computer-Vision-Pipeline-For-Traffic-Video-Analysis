import mysql.connector
from mysql.connector import Error

# MySQL database connection
def create_connection(host_name, user_name, user_password, db_name):
    connection = None
    try:
        connection = mysql.connector.connect(
            host=host_name,
            user=user_name,
            password=user_password,
            database=db_name
        )
        if connection.is_connected():
            print("MySQL Database connection successful")
            return connection
    except Error as e:
        print(f"The error '{e}' occurred")
    return None

# Drop the tables if they exist
def drop_tables(connection):
    drop_tables_query = """
    DROP TABLE IF EXISTS vehicle_counts;
    DROP TABLE IF EXISTS all_frames;
    """
    cursor = connection.cursor()
    try:
        for query in drop_tables_query.split(';'):
            if query.strip():
                cursor.execute(query)
        connection.commit()
        print("Tables dropped successfully")
    except Error as e:
        print(f"The error '{e}' occurred")

# Create the tables
def create_tables(connection):
    create_vehicle_counts_table_query = """
    CREATE TABLE IF NOT EXISTS vehicle_counts (
        frame_id INT,
        tracker_id VARCHAR(50),
        class_name VARCHAR(50),
        confidence FLOAT,
        traffic_signal VARCHAR(10),
        jaywalking VARCHAR(10),
        red_light_dashing VARCHAR(10),
        speed FLOAT,
        time VARCHAR(12),
        PRIMARY KEY (frame_id, tracker_id)
    );
    """

    create_all_frames_table_query = """
    CREATE TABLE IF NOT EXISTS all_frames (
        frame_id INT PRIMARY KEY,
        time VARCHAR(12),
        traffic_signal VARCHAR(10)
    );
    """

    cursor = connection.cursor()
    try:
        cursor.execute(create_vehicle_counts_table_query)
        cursor.execute(create_all_frames_table_query)
        connection.commit()
        print("Tables created successfully")
    except Error as e:
        print(f"The error '{e}' occurred")
    finally:
        cursor.close()

# Insert data into the vehicle_counts table
def insert_car_count(connection, frame_id, tracker_id, class_name, confidence, traffic_signal, jaywalking,
                     red_light_dashing, speed, time):
    insert_query = """
    INSERT INTO vehicle_counts 
    (frame_id, tracker_id, class_name, confidence, traffic_signal, jaywalking, red_light_dashing, speed, time)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    cursor = connection.cursor()
    try:
        # If tracker_id is None or 'N/A', replace it with a placeholder value
        if tracker_id is None or tracker_id == 'N/A':
            tracker_id = 'UNKNOWN'

        cursor.execute(insert_query, (
        frame_id, tracker_id, class_name, confidence, traffic_signal, jaywalking, red_light_dashing, speed, time))
        connection.commit()
        print("Data inserted successfully")
    except Error as e:
        print(f"The error '{e}' occurred")
    finally:
        cursor.close()

# Insert or update frame in all_frames table
def insert_or_update_frame(connection, frame_id, time, traffic_signal):
    insert_query = """
    INSERT INTO all_frames (frame_id, time, traffic_signal)
    VALUES (%s, %s, %s)
    ON DUPLICATE KEY UPDATE time = VALUES(time), traffic_signal = VALUES(traffic_signal)
    """
    cursor = connection.cursor()
    try:
        cursor.execute(insert_query, (frame_id, time, traffic_signal))
        connection.commit()
    except Error as e:
        print(f"The error '{e}' occurred")
    finally:
        cursor.close()

# Function to ensure all frames are recorded
def ensure_all_frames_recorded(connection, max_frame_id):
    cursor = connection.cursor()
    try:
        for frame_id in range(1, max_frame_id + 1):
            check_query = "SELECT * FROM all_frames WHERE frame_id = %s"
            insert_query = """
            INSERT INTO all_frames 
            (frame_id, time, traffic_signal)
            VALUES (%s, %s, %s)
            """

            cursor.execute(check_query, (frame_id,))
            result = cursor.fetchone()
            if result is None:
                cursor.execute(insert_query, (frame_id, 'N/A', 'N/A'))
                connection.commit()
    except mysql.connector.Error as err:
        print(f"Database error: {err}")
    finally:
        while connection.unread_result:
            connection.get_rows()
        cursor.close()

