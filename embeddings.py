#!/usr/bin/env python
# coding: utf-8

import numpy as np 
import pandas as pd 
import sklearn
import random
import io
import numbers
import sklearn.feature_extraction.text
import pickle
import nltk
from nltk.tokenize import word_tokenize
from collections import OrderedDict

# getting all videos' names
videos = []
with open('./input') as f:
    for i in f:
        names = i.split('.')[0]
        videos.append(names)

# Reading catptions
captions = pd.read_csv('video_corpus.csv')
english_captions = captions[captions['Language'] == 'English']
print("Total DPs: ", len(english_captions));

english_captions['NewID'] =  english_captions['VideoID'].str.cat(english_captions['Start'].values.astype(str), sep='_')
english_captions['NewID'] =  english_captions['NewID'].str.cat(english_captions['End'].values.astype(str), sep='_')

trained = english_captions.loc[english_captions['NewID'].isin(videos)]

def remove(my_str):
    # define punctuation
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    no_punct = ""
    if len(my_str) == 1:
        return my_str;
    for char in my_str:
       if char not in punctuations:
           no_punct = no_punct + char
    return no_punct

index_to_word = []
data = english_captions['Description']
data = data.str.lower()
data = list(data)
new = data
nltk.download('punkt');
print(len(trained));
words = [];
for item in new:
    try:
        #print(word_tokeniz)
        words += (word_tokenize(item))
    except:
        continue;

for i in range(len(words)):
    words[i] = remove(words[i]);
print(len(words))
    
index_to_word = list(OrderedDict.fromkeys(words));
index_to_word.append("<sos>");
index_to_word.append("<eos>");
index_to_word.append("<pad>");
word_to_index = {index_to_word[i]:i for i in range(len(index_to_word))}

print("Length of Word to Index and Index to Word: ", len(index_to_word), len(word_to_index));

def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    remaining_vocab = set(index_to_word)
    print(len(remaining_vocab));
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0];
        if word in remaining_vocab:
            embedding = np.array([float(val) for val in splitLine[1:]])
            model[word_to_index[word]] = embedding
            remaining_vocab.remove(word)
    print("Done.",len(model)," words loaded!")
    
    for key in word_to_index.keys():
        idx = word_to_index[key]
        if idx not in model.keys():
            model[idx] = np.random.rand(300);
        
    return model;

#model = loadGloveModel('../glove.6B.300d.txt')

#print("Model Length: ", len(model));
# print(index_to_word)
#model[word_to_index['<pad>']] = np.zeros((300));
#word_to_index['chili'];
#print(word_to_index['chili']);

# # Save the embedding to a pickle file 
#f = open("embedding.pkl","wb")
#pickle.dump(model,f)
#f.close();

f = open('vocab.pkl', 'wb');
pickle.dump(word_to_index, f);
f.close();

# new = pickle.load(open('embedding.pkl','rb'))



# print(new[word_to_index['\'']])
