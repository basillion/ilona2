from flask import Flask, request
from predict import predict
from loguru import logger
import configparser

config = configparser.ConfigParser()
config.read("/app/config.ini")


logger.remove()
log_path = config['path']['log_path']
logger.add(log_path, format="{time:YYY-MM-DD HH:mm:ss} {module} {level} {message}", level="DEBUG", rotation="500kb", compression="zip")


# create the Flask app
app = Flask(__name__)


@app.route('/check')
def index():
    return "Ilona is here"

@app.route('/query')
def query():
    logger.debug("Got request")

    request_id = request.args.get('request_id')
    description = request.args['description']

    logger.debug(f"Request info query: {request_id}: {description} ")

    if request_id and description:
        predict(description, request_id)

    return f'''
            <h1>request_id is: {request_id}</h1>
            <h1>description is: {description}</h1>
            '''


@app.route('/json', methods=['POST'])
def json():

    logger.debug("Got request")

    request_data = request.get_json()
    request_id = request_data['request_id']
    description = request_data['description']

    logger.debug(f"Request info json: {request_id}: {description} ")

    if request_id and description:
        predict(description, request_id)

    return f'''
              <h1>request_id is: {request_id}</h1>
              <h1>description is: {description}</h1>
            '''


if __name__ == '__main__':
    app.run(debug=False, port=5000, host='0.0.0.0')

