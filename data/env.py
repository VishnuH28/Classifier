import os
from dotenv import load_dotenv

load_dotenv()

REQUIRED_ENV_VARS = ["POSTGRES_DB", "POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_HOST", "POSTGRES_PORT"]

class Env:
    @staticmethod
    def check_missing_vars():
        missing_vars = [var for var in REQUIRED_ENV_VARS if os.getenv(var) is None]
        if missing_vars:
            raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

    @staticmethod
    def get_env(var_name, default=None):
        return os.getenv(var_name, default)

Env.check_missing_vars()

POSTGRES_DB = Env.get_env("POSTGRES_DB")
POSTGRES_USER = Env.get_env("POSTGRES_USER")
POSTGRES_PASSWORD = Env.get_env("POSTGRES_PASSWORD")
POSTGRES_HOST = Env.get_env("POSTGRES_HOST")
POSTGRES_PORT = Env.get_env("POSTGRES_PORT")