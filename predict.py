import requests
from keras.utils import pad_sequences
import pickle
from keras.models import Sequential
from keras.layers import Dense, GRU, Embedding
from loguru import logger
import sys
import json
import configparser
import os

config = configparser.ConfigParser()
config.read('/app/config.ini')
with open(config['path']['teams'], encoding='utf8', errors='ignore') as data_file:
    teams = json.load(data_file)


server = os.environ['SD_SERV']
api_key = os.environ['API_KEY']
# Строим сеть GRU
logger.debug('Building a model...')
model_gru = Sequential()
model_gru.add(Embedding(int(config['model']['num_words']),\
                        int(config['model']['n_emb']), input_length=\
                        int(config['model']['max_desc_len'])))
model_gru.add(GRU(int(config['model']['n_gru'])))
model_gru.add(Dense(int(config['model']['n_out']), activation='softmax'))

# Функция очистки текста и морфологического приведения
def clean_text(text):
    text = str(text).replace("\\", " ").replace(u"╚", " ").replace(u"╩", " ")
    text = text.lower()
    text = re.sub('\-\s\r\n\s{1,}|\-\s\r\n|\r\n', '', text)  # deleting newlines and line-breaks
    text = re.sub('[.,:;_%©?*,!@#$%^&()\d]|[+=]|[[]|[]]|[/]|"|\s{2,}|-', ' ', text)  # deleting symbols
    #text = " ".join(ma.parse(word)[0].normal_form for word in text.split())
    text = ' '.join(word for word in text.split() if len(word) >= 2)
    # text = text.encode("utf-8")

    return text

@logger.catch
def predict(req, request_id):
    logger.debug("Loading a model")
    try:
    	model_gru.load_weights(config['path']['model'])
    except:
    	logger.error(sys.stderr)
    logger.debug("Cleaning the text")
    try:
    	req = clean_text(req)
    	logger.debug("Cleaned")
    except:
    	logger.error(sys.stderr)
    logger.debug("Loading a tokenizer")
    # Загружаем токенайзер из файла
    with open(config['path']['token'], 'rb') as handle:
        tokenizer = pickle.load(handle)

    data = tokenizer.texts_to_sequences([req])
    data = pad_sequences(data, maxlen=int(config['model']['max_desc_len']))

    # Вызываем функцию предсказания
    logger.debug("Start a new prediction")
    result = model_gru.predict(data)[0].tolist()
    # Выбираем ID максимально близкий к 1
    tmp = max(result)
    result_team = result.index(tmp) + 1

    # Возвращаем ID команды
    logger.debug("Sending a request to SD")
    
    url = f"https://{server}/api/v3/requests/{request_id}/notes"
    headers = {"authtoken": api_key}

    input_data = '''{{
    "note": {{
        "description": "ИИ рекомендует назначить специалиста из группы {group} ",
        "show_to_requester": false,
        "mark_first_response": false,
        "add_to_linked_requests": false
    }}
    }}'''.format(group=list(teams.keys())[list(teams.values()).index(result_team)])

    data = {'input_data': input_data}
    # params = {'input_data': input_data}
    req = requests.post(url, data=data, headers=headers, verify=False)
    logger.debug("The request sent")
    return result_team
