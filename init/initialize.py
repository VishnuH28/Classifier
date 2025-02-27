from database.postgres import postgres, check_connection
from data.table_names import TableNames

def create_tables():
    """Creates necessary database tables for classification."""
    queries = {
        TableNames.DETECTION_REQUEST.value: """
        CREATE TABLE IF NOT EXISTS detection_request (
            req_id VARCHAR(50) PRIMARY KEY,
            r_id VARCHAR(50),
            category VARCHAR(20),
            status VARCHAR(20) DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """,
        TableNames.DETECTED_OBJECTS.value: """
        CREATE TABLE IF NOT EXISTS detected_objects (
            id SERIAL PRIMARY KEY,
            req_id VARCHAR(50) REFERENCES detection_request(req_id),
            image_path TEXT NOT NULL,
            object_label VARCHAR(50),
            confidence FLOAT,
            detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    }

    global postgres
    postgres = check_connection(postgres)
    try:
        with postgres.cursor() as cur:
            for table, query in queries.items():
                print(f"Creating table {table}...")
                cur.execute(query)
            postgres.commit()
        print("All tables created successfully.")
    except Exception as e:
        print(f"Error creating tables: {e}")

def initialize():
    """Runs all initialization functions."""
    print("Initializing database and system setup for classification...")
    create_tables()
    print("Initialization complete.")

if __name__ == "__main__":
    initialize()