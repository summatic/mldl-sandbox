# __Author__: Hanseok Jo
import numpy as np
import pandas as pd
import glob
from gensim.models import Phrases
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.preprocessing import text
from datetime import datetime
import string
from keras.layers.core import Dense, Activation


def trans():
    f = string.ascii_uppercase
    return f


def read_sentences(n):
    files = glob.glob('/Users/Hanseok/PycharmProjects/ml/test/*/*')
    files = [file for file in files if file.split('/')[-1] != 'rule']
    sentence_list, words_list = [], []
    for file in files[:n]:
        try:
            df = pd.read_table(file)
            sentence = ['<S> %s <E>' % row['textPattern'] for index, row in df.iterrows()]
            sentence_list.append(sentence)
            words = [row['textPattern'] for index, row in df.iterrows()]
            words_list.extend(words)
        except Exception as e:
            continue
    nb_words = len(list(set(words_list)))
    return sentence_list, nb_words


def n_gram(n, sentence_stream):
    """
    :param n:
    :param sentence_stream:
    :return:
    """
    for i in range(n):
        phrases = Phrases(sentence_stream)
        sentence_stream = [phrases[s] for s in sentence_stream]
    return sentence_stream


def window(sentence_stream, return_form, window_siz):
    """
    :param sentence_stream:
    :param return_form:
    :param window_siz:
    :return:
    """
    word_list = []
    for sentence in sentence_stream:
        s = '<S> %s <E>' % ' '.join(sentence)
        word_list.extend(s.split())

    return_list = []
    for i in range(len(word_list) - window_siz):
        words = word_list[i:i + window_siz]
        words = words if return_form == 'seq' else ' '.join(words)
        return_list.append(words)
    return return_list


sentences, nb_words = read_sentences(10)
phrased = n_gram(3, sentences)
window_size = 5
windowed = window(phrased, 'text', window_size)

X, y = [], []
for i in windowed:
    one_hot = text.one_hot(i, n=nb_words, filters=trans())
    X.append(one_hot)
    y.append(one_hot[int(window_size / 2) - 1])
X, y = np.array(X), np.array(y)
print(X.shape, y.shape)

model = Sequential()
model.add(Embedding(input_dim=nb_words, output_dim=1, input_length=window_size))

input_array = np.array(X)

model.compile('rmsprop', 'mse')

model.fit(X, y)
