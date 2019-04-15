import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split
import numpy as np
import tokenize
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

label={0:'real',1:'fake' }

#input taken from training dataset for real and fake articles
data=[]
path1="C:/Users/Mainaki Saraf/Desktop/SMDK/training/fakeNewsDataset/legit"
path2="C:/Users/Mainaki Saraf/Desktop/SMDK/training/fakeNewsDataset/fake"
for file in os.listdir(path1):
   # print(file)
    with open(os.path.join(path1, file), "r") as f:
        text = f.read()
        data.append([text,0])
        
for file in os.listdir(path2):
  #  print(file)
    with open(os.path.join(path2, file), "r") as f:
        text = f.read()
        data.append([text,1])        
		
#datafram created from text files
data_t=pd.DataFrame(data,columns=['text', 'label'])

import matplotlib.pyplot as plt

import re

from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization, LSTM, Embedding, Reshape
from keras.models import load_model, model_from_json

from sklearn.model_selection import train_test_split

import os
import urllib

from urllib.request import urlretrieve

from os import mkdir, makedirs, remove, listdir

from collections import Counter

with open('glove.6B.50d.txt','rb') as f:
    lines = f.readlines()
	
glove_weights = np.zeros((len(lines), 50))
words = []
for i, line in enumerate(lines):
    word_weights = line.split()
    words.append(word_weights[0])
    weight = word_weights[1:]
    glove_weights[i] = np.array([float(w) for w in weight])
word_vocab = [w.decode("utf-8") for w in words]

word2glove = dict(zip(word_vocab, glove_weights))

from keras.engine.topology import Layer
import keras.backend as K
from keras import initializers
import numpy as np

class Embedding2(Layer):

    def __init__(self, input_dim, output_dim, fixed_weights, embeddings_initializer='uniform', 
                 input_length=None, **kwargs):
        kwargs['dtype'] = 'int32'
        if 'input_shape' not in kwargs:
            if input_length:
                kwargs['input_shape'] = (input_length,)
            else:
                kwargs['input_shape'] = (None,)
        super(Embedding2, self).__init__(**kwargs)
    
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embeddings_initializer = embeddings_initializer
        self.fixed_weights = fixed_weights
        self.num_trainable = input_dim - len(fixed_weights)
        self.input_length = input_length
        
        w_mean = fixed_weights.mean(axis=0)
        w_std = fixed_weights.std(axis=0)
        self.variable_weights = w_mean + w_std*np.random.randn(self.num_trainable, output_dim)

    def build(self, input_shape, name='embeddings'):        
        fixed_weight = K.variable(self.fixed_weights, name=name+'_fixed')
        variable_weight = K.variable(self.variable_weights, name=name+'_var')
        
        self._trainable_weights.append(variable_weight)
        self._non_trainable_weights.append(fixed_weight)
        
        self.embeddings = K.concatenate([fixed_weight, variable_weight], axis=0)
        
        self.built = True

    def call(self, inputs):
        if K.dtype(inputs) != 'int32':
            inputs = K.cast(inputs, 'int32')
        out = K.gather(self.embeddings, inputs)
        return out

    def compute_output_shape(self, input_shape):
        if not self.input_length:
            input_length = input_shape[1]
        else:
            input_length = self.input_length
        return (input_shape[0], input_length, self.output_dim)
		
data_t.text = data_t.text.str.lower()

data_t.text = data_t.text.str.replace(r'http[\w:/\.]+','<URL>') # remove urls
data_t.text = data_t.text.str.replace(r'[^\.\w\s]','') #remove everything but characters and punctuation
data_t.text = data_t.text.str.replace(r'\.\.+','.') #replace multple periods with a single one
data_t.text = data_t.text.str.replace(r'\.',' . ') #replace multple periods with a single one
data_t.text = data_t.text.str.replace(r'\s\s+',' ') #replace multple white space with a single one
data_t.text = data_t.text.str.strip() 
print(data_t.shape)
data_t.head()

all_text = ' '.join(data_t.text.values)
words = all_text.split()
u_words = Counter(words).most_common()
u_words_counter = u_words
u_words_frequent = [word[0] for word in u_words if word[1]>5] # we will only consider words that have been used more than 5 times

u_words_total = [k for k,v in u_words_counter]
word_vocab = dict(zip(word_vocab, range(len(word_vocab))))
word_in_glove = np.array([w in word_vocab for w in u_words_total])

words_in_glove = [w for w,is_true in zip(u_words_total,word_in_glove) if is_true]
words_not_in_glove = [w for w,is_true in zip(u_words_total,word_in_glove) if not is_true]

print('Fraction of unique words in glove vectors: ', sum(word_in_glove)/len(word_in_glove))

# # create the dictionary
word2num = dict(zip(words_in_glove,range(len(words_in_glove))))
len_glove_words = len(word2num)
freq_words_not_glove = [w for w in words_not_in_glove if w in u_words_frequent]
b = dict(zip(freq_words_not_glove,range(len(word2num), len(word2num)+len(freq_words_not_glove))))
word2num = dict(**word2num, **b)
word2num['<Other>'] = len(word2num)
num2word = dict(zip(word2num.values(), word2num.keys()))

int_text = [[word2num[word] if word in word2num else word2num['<Other>'] 
             for word in content.split()] for content in data_t.text.values]

print('The number of unique words are: ', len(u_words))
print('The first review looks like this: ')
print(int_text[0][:20])
print('And once this is converted back to words, it looks like: ')
print(' '.join([num2word[i] for i in int_text[0][:20]]))

print('The number of articles greater than 150 in length is: ', np.sum(np.array([len(t)>150 for t in int_text])))
print('The number of articles less than 100 in length is: ', np.sum(np.array([len(t)<100 for t in int_text])))

num2word[len(word2num)] = '<PAD>'
word2num['<PAD>'] = len(word2num)

for i, t in enumerate(int_text):
    if len(t)<100:
        int_text[i] = [word2num['<PAD>']]*(100-len(t)) + t
    elif len(t)>100:
        int_text[i] = t[:100]
    else:
        continue

x = np.array(int_text)
y = np.array(data_t.label)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

model = Sequential()
model.add(Embedding(len(word2num), 50)) # , batch_size=batch_size
#model.add(Embedding2(len(word2num), 50, fixed_weights=np.array([word2glove[w] for w in words_in_glove])))
model.add(LSTM(16))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.summary()

batch_size = 64
epochs = 5
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test))

model.save("model.hdf5")

