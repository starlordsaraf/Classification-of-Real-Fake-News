#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.models import load_model
model = load_model('model.hdf5')


# In[13]:


import sys
from collections import Counter
import numpy as np


# In[24]:


file=str(sys.argv[1])
f=open(file,'r')
sentence=f.read()


# In[14]:

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


# In[15]:



all_text = ' '.join(sentence)
words = all_text.split()
u_words = Counter(words).most_common()
u_words_counter = u_words
u_words_frequent = [word[0] for word in u_words if word[1]>5] # we will only consider words that have been used more than 5 times

u_words_total = [k for k,v in u_words_counter]
word_vocab = dict(zip(word_vocab, range(len(word_vocab))))
word_in_glove = np.array([w in word_vocab for w in u_words_total])

words_in_glove = [w for w,is_true in zip(u_words_total,word_in_glove) if is_true]
words_not_in_glove = [w for w,is_true in zip(u_words_total,word_in_glove) if not is_true]


# In[17]:


word2num = dict(zip(words_in_glove,range(len(words_in_glove))))
len_glove_words = len(word2num)
freq_words_not_glove = [w for w in words_not_in_glove if w in u_words_frequent]
b = dict(zip(freq_words_not_glove,range(len(word2num), len(word2num)+len(freq_words_not_glove))))
word2num = dict(**word2num, **b)
word2num['<Other>'] = len(word2num)
num2word = dict(zip(word2num.values(), word2num.keys()))

int_text = [[word2num[word] if word in word2num else word2num['<Other>'] 
             for word in content.split()] for content in sentence]


# In[20]:


sentence_num = [word2num[w] if w in word2num else word2num['<Other>'] for w in sentence.split()]
#sentence_num = [word2num['<PAD>']]*(500-len(sentence_num)) + sentence_num
sentence_num = np.array(sentence_num)
answer=model.predict(sentence_num[None,:])


# In[23]:


if(answer[0]>=0.5):
    print(0)   	#real
	exit(0)
else:
    print(1)        #fake
	exit(1)

