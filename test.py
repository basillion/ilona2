import pandas as pd
from keras import utils
from keras.utils import pad_sequences
import pickle
import json
import configparser
from keras.models import Sequential
from keras.layers import Dense, GRU, Embedding
import re

config = configparser.ConfigParser()
config.read("./config.ini")

# Строим сеть GRU
model_gru = Sequential()
model_gru.add(Embedding(int(config['model']['num_words']),\
                        int(config['model']['n_emb']), input_length=\
                        int(config['model']['max_desc_len'])))
model_gru.add(GRU(int(config['model']['n_gru'])))
#model_gru.add(Dense(256, activation='elu'))
model_gru.add(Dense(128, activation='relu'))
model_gru.add(Dense(int(config['model']['n_out']), activation='softmax'))

# Функция очистки текста и морфологического приведения
def clean_text(text):
    text = str(text).replace("\\", " ").replace(u"╚", " ").replace(u"╩", " ")
    text = text.lower()
    text = re.sub('\-\s\r\n\s{1,}|\-\s\r\n|\r\n', '', text)  # deleting newlines and lin>
    text = re.sub('[.,:;_%©?*,!@#$%^&()\d]|[+=]|[[]|[]]|[/]|"|\s{2,}|-', ' ', text)  # d>
    #text = " ".join(ma.parse(word)[0].normal_form for word in text.split())
    text = ' '.join(word for word in text.split() if len(word) >= 2)
    # text = text.encode("utf-8")

    return text


def test():
    # Загружаем токенайзер из файла
    with open(config['path']['token'], 'rb') as handle:
        tokenizer = pickle.load(handle)

    # Готовим данные для теста
    with open(config['path']['teams'], encoding='utf8', errors='ignore') as data_file:
        teams = json.load(data_file)


    df_test = pd.read_excel(config['path']['testset'])
    #df_test['text'] = df_test['author'] + ' ' + df_test['text'] + ' ' + df_test['place']
    df_test['Description'] = df_test.apply(lambda x: clean_text(x[u'text']), axis=1)
    test_sequences = tokenizer.texts_to_sequences(df_test['Description'])
    x_test = pad_sequences(test_sequences, maxlen=int(config['model']['max_desc_len']))
    df_test['team_code'] = df_test[u'team'].map(teams)
    y_test = utils.to_categorical(df_test['team_code'] - 1, 15)

    model_gru.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
    # Загружаем сохраненную модель из файла
    model_gru.load_weights(config['path']['model'])

    # Выводим результат работы на тестовом датасете
    model_gru.evaluate(x_test, y_test, verbose=1)


test()
