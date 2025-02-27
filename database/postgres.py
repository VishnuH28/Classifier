import psycopg2
from psycopg2 import OperationalError
import data.env as env
import time
import threading

def create_connection():
    """Create a database connection to the PostgreSQL database."""
    connection = None
    try:
        connection = psycopg2.connect(
            database=env.POSTGRES_DB,
            user=env.POSTGRES_USER,
            password=env.POSTGRES_PASSWORD,
            host=env.POSTGRES_HOST,
            port=env.POSTGRES_PORT
        )
        print("Postgres DB connected")
    except OperationalError as e:
        print(f"Connection error: {e}")
    return connection

def check_connection(connection):
    """Check if the connection is alive, if not, reconnect."""
    try:
        if connection is None or connection.closed != 0:
            print("Postgres DB connection lost. Reconnecting...")
            connection = create_connection()
    except Exception as e:
        print(f"Check connection error: {e}")
    return connection

postgres = create_connection()
if not postgres:
    raise Exception("Initial PostgreSQL connection failed. Check .env settings.")

def check_connection_periodically():
    global postgres
    while True:
        postgres = check_connection(postgres)
        time.sleep(60)

thread = threading.Thread(target=check_connection_periodically, daemon=True)
thread.start()