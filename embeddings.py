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
from nltk.tokenize import word_tokenize

# getting all videos' names
videos = []
with open('./input') as f:
    for i in f:
        # extract names of the videos 
        names = i.split('.')[0]
        videos.append(names)

# Reading catptions
captions = pd.read_csv('video_corpus.csv')
english_captions = captions[captions['Language'] == 'English']

# print(english_captions.iloc[:100,:])
# print(len(english_captions)) ## 85511 rows 

english_captions['NewID'] =  english_captions['VideoID'].str.cat(english_captions['Start'].values.astype(str), sep='_')
english_captions['NewID'] =  english_captions['NewID'].str.cat(english_captions['End'].values.astype(str), sep='_')



# get all the training data based on the ids on videos 
trained = english_captions.loc[english_captions['NewID'].isin(videos)]

def remove(my_str):
    # define punctuation
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    # To take input from the user
    # my_str = input("Enter a string: ")
    # remove punctuation from the string
    no_punct = ""
    for char in my_str:
       if char not in punctuations:
           no_punct = no_punct + char
    return no_punct

index_to_word = []
data = english_captions['Description']
data = data.str.lower()
data = list(data)
new = data
words = []
for item in new:
    try:
        words.append(word_tokenize(item))
    except:
        continue
for item in words:
    for word in item:
        word = remove(word)
        if word not in index_to_word:
            index_to_word.append(word)


word_to_index = {index_to_word[i]:i for i in range(len(index_to_word))}

# Load Glove models

def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    remaining_vocab = set(index_to_word)
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        if word in remaining_vocab:
            embedding = np.array([float(val) for val in splitLine[1:]])
            model[word_to_index[word]] = embedding
            remaining_vocab.remove(word)
    print("Done.",len(model)," words loaded!")
    
    for key in word_to_index.keys():
        idx = word_to_index[key]
        if idx not in model.keys():
            model[idx] = np.zeros(300)
        
    return model



model = loadGloveModel('./glove.6B.300d.txt')


# Save the embedding to a pickle file 
f = open("embedding.pkl","wb")
pickle.dump(model,f)


new = pickle.load(open('embedding.pkl','rb'))



new[word_to_index['rading']]