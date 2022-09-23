import pandas as pd
from keras import utils
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import pickle
import matplotlib.pyplot as plt
import pymorphy2
import re
from keras.models import Sequential
from keras.layers import Dense, GRU, Embedding
from loguru import logger
import json
import configparser
# Объявляем переменные
ma = pymorphy2.MorphAnalyzer()

config = configparser.ConfigParser()
config.read('config.ini')

# Строим сеть GRU
logger.debug('Building a model...')
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
    text = re.sub('\-\s\r\n\s{1,}|\-\s\r\n|\r\n', '', text)  # deleting newlines and line-breaks
    text = re.sub('[.,:;_%©?*,!@#$%^&()\d]|[+=]|[[]|[]]|[/]|"|\s{2,}|-', ' ', text)  # deleting symbols
    #text = " ".join(ma.parse(word)[0].normal_form for word in text.split())
    text = ' '.join(word for word in text.split() if len(word) >= 2)
    # text = text.encode("utf-8")

    return text


def train():
    df = pd.read_excel(config['path']['dataset'])
    #df['text'] = df['author'] + ' ' + df['text'] + ' ' + df['place']
    #print(df)
    logger.debug('Cleaning descriptions...')
    df['description'] = df.apply(lambda x: clean_text(x[u'text']), axis=1)
    print(df)

    # создадим массив, содержащий уникальные категории из нашего DataFrame
    logger.debug('Making a list of the teams...')
    teams = {}
    for key, value in enumerate(df[u'team'].unique()):
        teams[value] = key + 1
    logger.debug(teams)
    with open ('teams.txt', 'w') as file:
        file.write(json.dumps(teams))

    # Запишем в новую колонку числовое обозначение категории
    df['team_code'] = df[u'team'].map(teams)

    total_teams = len(df[u'team'].unique())
    logger.debug('Всего команд: {}'.format(total_teams))

    df = df.sample(frac=1).reset_index(drop=True)

    descriptions = df['description']
    teams = df[u'team_code']
    '''
    # Посчитаем максимальную длинну текста описания в словах
    max_words = 0
    for desc in descriptions:
        words = len(desc.split())
        if words > max_words:
            max_words = words
    print('Максимальная длина описания: {} слов'.format(max_words))
    maxSequenceLength = max_words
    '''
    # Выделяем правильные ответы
    y_train = utils.to_categorical(df['team_code'] - 1, total_teams)
    #print(y_train)

    tokenizer = Tokenizer(num_words=int(config['model']['num_words']))

    # Обучаем токенизатор на описаниях
    tokenizer.fit_on_texts(descriptions)
    #print(tokenizer.word_index)

    # Преобразуем описания в числовое представление
    sequences = tokenizer.texts_to_sequences(descriptions)

    # Просматриваем описания в числовом представлении
    #index = 1
    #print(descriptions[index])
    #print(sequences[index])

    # Сохраняем токенайзер в файл
    with open(config['path']['token'], 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Ограничиваем длину описания
    x_train = pad_sequences(sequences, maxlen=int(config['model']['max_desc_len']))
    #print(x_train[:5])

    model_gru.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    logger.debug(model_gru.summary())

    # Создаем callback для сохранения нейронной сети на каждой эпохе,
    # если качество работы на проверочном наборе данных улучшилось. Сеть сохраняется в файл best_model_gru.h5
    model_gru_save_path = config['path']['model'] 
    checkpoint_callback_gru = ModelCheckpoint(model_gru_save_path,
                                              monitor='val_accuracy',
                                              save_best_only=True,
                                              verbose=1)

    history_gru = model_gru.fit(x_train,
                                y_train,
                                epochs=3,
                                batch_size=128,
                                validation_split=0.1,
                                callbacks=[checkpoint_callback_gru])

    plt.plot(history_gru.history['accuracy'],
             label='Доля верных ответов на обучающем наборе')
    plt.plot(history_gru.history['val_accuracy'],
             label='Доля верных ответов на проверочном наборе')
    plt.xlabel('Эпоха обучения')
    plt.ylabel('Доля верных ответов')
    plt.legend()
    plt.savefig(config['path']['plt_fig'])


train()
