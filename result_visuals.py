import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

# Load the CSV file
csv_path = '/Users/joni/Downloads/YOLOv8x_output1.csv'
data = pd.read_csv(csv_path)

# Database connection details
db_connection_str = 'mysql+pymysql://root:12jo34ni@127.0.0.1/traffic_db1'
db_connection = create_engine(db_connection_str)

# Table name
table_name = 'vehicle_counts'

# Import the data into SQL
data.to_sql(name=table_name, con=db_connection, if_exists='replace', index=False)

print(f"Data imported successfully into the table '{table_name}'")

# Define your queries
queries = {
    'Jaywalker Count': """
        SELECT COUNT(DISTINCT tracker_id) AS jaywalker_count
        FROM vehicle_counts
        WHERE jaywalking = 'yes'
        AND tracker_id IN (
            SELECT tracker_id
            FROM vehicle_counts
            GROUP BY tracker_id
            HAVING COUNT(*) >= 5
        );
    """,
    'Red Light Dasher Count': """
        SELECT COUNT(DISTINCT tracker_id) AS dasher_count
        FROM vehicle_counts
        WHERE red_light_dashing = 'Yes'
        AND tracker_id IN (
            SELECT tracker_id
            FROM vehicle_counts
            GROUP BY tracker_id
            HAVING COUNT(*) >= 10
        );
    """,
    'Speeding Vehicle Count': """
        SELECT COUNT(DISTINCT tracker_id) AS speeding_vehicle_count
        FROM vehicle_counts
        WHERE speed > 40
        AND tracker_id IN (
            SELECT tracker_id
            FROM vehicle_counts
            GROUP BY tracker_id
            HAVING COUNT(*) >= 10
        );
    """,
    'Total Vehicle Count': """
        SELECT COUNT(DISTINCT tracker_id) AS total_vehicle_count
        FROM vehicle_counts
        WHERE class_name IN ('car', 'truck', 'bus', 'bicycle')
        AND tracker_id IN (
            SELECT tracker_id
            FROM vehicle_counts
            WHERE class_name IN ('car', 'truck', 'bus', 'bicycle')
            GROUP BY tracker_id
            HAVING COUNT(*) >= 10
        );
    """,
    'Total People Count': """
        SELECT COUNT(DISTINCT tracker_id) AS total_people_count
        FROM vehicle_counts
        WHERE class_name = 'person'
        AND tracker_id IN (
            SELECT tracker_id
            FROM vehicle_counts
            WHERE class_name = 'person'
            GROUP BY tracker_id
            HAVING COUNT(*) >= 10
        );
    """
}

# Execute queries and store results
results = {}
for key, query in queries.items():
    results[key] = pd.read_sql(query, db_connection).iloc[0, 0]

# Plotting the results
fig, ax = plt.subplots()
ax.bar(results.keys(), results.values())
ax.set_ylabel('Count')
ax.set_title('Traffic Violation Statistics')

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
