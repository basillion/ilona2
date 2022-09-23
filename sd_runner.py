import requests
import json
from bs4 import BeautifulSoup
import re
import sys
from loguru import logger

#logger.remove()
sd_addr = '10.32.0.211'

#log_path = 'D:\\Neuro\\ilona\\debug.log'
#run_log = logger.add(log_path, format="{time:YYY-MM-DD HH:mm:ss} {module} {level} {message}", level="DEBUG", rotation="100kb", compression="zip")

logger.debug("Reading json path")
file_Path = sys.argv[1]
logger.debug("Path readed successful!")
with open(file_Path, encoding='utf8', errors='ignore') as data_file:
    data = json.load(data_file)
logger.debug("Parsing id and description")
request_id = str(data['request']['id'])
description = str(data['request']['description'])

def clear_html(text):
    soup = BeautifulSoup(text, features="html.parser")
    clear_text = soup.get_text()
    clear_text = re.sub('(\r\n){2,}|(\n){2,}', '\n', clear_text)

    return clear_text

if request_id and description:
    logger.debug("Request info: {request_id}: {description} ")

r = requests.post(f"http://{sd_addr}:5000/json", json={'request_id': request_id, 'description': clear_html(description)})
logger.debug(r.request)
logger.debug(r.status_code)
#logger.remove(run_log)
