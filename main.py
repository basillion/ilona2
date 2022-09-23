from predict import predict
from train import train
from test import test
import sys
import json
from loguru import logger
from bs4 import BeautifulSoup
import re


logger.add("debug.log", format="{time} {level} {message}", level="DEBUG", rotation="500kb", compression="zip")

'''
logger.debug("\nReading path")
file_Path = sys.argv[1]
logger.debug("Path readed successful!")
with open(file_Path, encoding='utf8', errors='ignore') as data_file:
    data = json.load(data_file)
logger.debug("Parsing id and description")
request_id = str(data['request']['id'])
description = str(data['request']['description'])

soup = BeautifulSoup(description, features="html.parser")
clear_description = soup.get_text()
clear_description = re.sub('(\r\n){2,}|(\n){2,}', '\n', clear_description)

logger.debug("Parsed!")
logger.debug(f"request_id: {request_id}\n description: {clear_description}")

logger.debug("Starting prediction")
@logger.catch
def main():
    predict(clear_description, request_id)
if description:
    main()
else:
    logger.debug("No description")
'''
train()
    
