# __Author__: Hanseok Jo
import numpy as np
import pandas as pd
import glob
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.preprocessing import text
import string
from keras.layers.core import Dense, Activation


def trans():
    f = string.ascii_uppercase
    return f


files = glob.glob('/Users/Hanseok/PycharmProjects/ml/test/*/*')
files = [file for file in files if file.split('/')[-1] != 'rule']

for file in files:
    try:
        df = pd.read_table(file)
        textPattern = [row['textPattern'] for index, row in df.iterrows()]
        docs = ' <S> '.join(textPattern)
    except Exception as e:
        continue
    print(docs)
    docs = text.one_hot(docs, n=100, filters=trans())
    print(docs)
    break

model = Sequential()
model.add(Embedding(1000, 64, input_length=10000))
# the model will take as input an integer matrix of size (batch, input_length).
# the largest integer (i.e. word index) in the input should be no larger than 1000 (vocabulary size).
# now model.output_shape == (None, 10, 64), where None is the batch dimension.

input_array = np.random.randint(1000, size=(32, 10000))

model.compile('rmsprop', 'mse')
output_array = model.predict(input_array)
print(output_array.shape)
