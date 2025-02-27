from flask import Flask
from routes.api_routes import apiRoutes
from data.env import POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_HOST, POSTGRES_PORT
import database.postgres
from init.initialize import initialize

app = Flask(__name__)
app.register_blueprint(apiRoutes, url_prefix="/api/v1")

@app.route('/')
def home():
    return "Classifier Server is running!"

initialize()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)