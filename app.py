import logging
from decouple import config
from controller import app

from waitress import serve
from flask_swagger_ui import get_swaggerui_blueprint

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')

SWAGGER_URL = "/docs"
UI_URL = '/static/swagger.json'

# load blueprint from json file
blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    UI_URL,
    config={
        "app_name": "Technical test"
    }
)

app.register_blueprint(blueprint)

ADDRESS = config("ADDRESS")
PORT = config("PORT")

serve(app, port=PORT, host=ADDRESS)
