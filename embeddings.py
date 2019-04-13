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



### Getting all words for the Description 
vectorizer = sklearn.feature_extraction.text.CountVectorizer(min_df=1,stop_words='english')
X = vectorizer.fit_transform(trained['Description'])
index_to_word = vectorizer.get_feature_names()
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
    return model



model = loadGloveModel('./glove/glove.6B.300d.txt')

idx = word_to_index['chair']
print(model[idx])

# Save the embedding to a pickle file 
f = open("embedding.pkl","wb")
pickle.dump(model,f)


new = pickle.load(open('embedding.pkl','rb'))


