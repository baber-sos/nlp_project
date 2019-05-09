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
from collections import OrderedDict

glove_path = '/common/users/bk456/glove.6B.300d.txt';

# getting all videos' names

videos = []
with open('./input') as f:
    for i in f:
        names = i.split('.')[0]
        videos.append(names)

# with open('./val_input') as f:
#     for i in f:
#         names = i.split('.')[0]
#         videos.append(names)

# Reading catptions
captions = pd.read_csv('video_corpus.csv')
english_captions = captions[captions['Language'] == 'English']

english_captions['NewID'] =  english_captions['VideoID'].str.cat(english_captions['Start'].values.astype(str), sep='_')
english_captions['NewID'] =  english_captions['NewID'].str.cat(english_captions['End'].values.astype(str), sep='_')

trained = english_captions.loc[english_captions['NewID'].isin(videos)]

# print("Total DPs: ", len(trained));

glove_file = open(glove_path);
glove_words = {};

for line in glove_file:
    cur_word = line.strip().split(' ')[0];
    glove_words[cur_word] = 1;

glove_file.close();

# print("Glove Words: ", len(glove_words))
def remove(my_str):
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    no_punct = ""
    if len(my_str) == 1:
        return my_str;
    for char in my_str:
       if char not in punctuations:
           no_punct = no_punct + char
    return no_punct

index_to_word = []
# data = english_captions['Description']
data = trained['Description']
data = data.str.lower()
data = list(data)
new = data

# print(len(data));
words = [];
for item in new:
    try:
        cur = [remove(x) if remove(x) in glove_words else "a" for x in word_tokenize(item)];
        words += (cur);
    except:
        continue;

cur_vocab = list(OrderedDict.fromkeys(words));
# print(len(cur_vocab));

nvnames = [];
count = 0;
for i in range(len(english_captions)):
    caption = english_captions['Description'].iloc[i];
    this_vocab = set([remove(word) if remove(word) in glove_words else 'a' for word in word_tokenize(caption)]);
    if this_vocab.issubset(cur_vocab):
        vname = english_captions['NewID'].iloc[i]+ '.avi';
        if vname in videos or vname in nvnames:
            continue;
        nvnames.append(vname);
        count += 1;
    if count >= 20:
        break;

# print(nvnames);
for name in nvnames:
    print(name);


# count = 0;
# noted = -1;
# for word in index_to_word:
#     if word == "":
#         noted = count;
#     count += 1;

# if noted != -1:
#     index_to_word = index_to_word[:noted] + index_to_word[noted+1:];

# index_to_word.append("<sos>");
# index_to_word.append("<eos>");
# index_to_word.append("<pad>");
# index_to_word.append("<unk>");
# word_to_index = {index_to_word[i]:i for i in range(len(index_to_word))}

# print("Length of Word to Index and Index to Word: ", len(index_to_word), len(word_to_index));

# def loadGloveModel(gloveFile):
#     print("Loading Glove Model")
#     remaining_vocab = set(index_to_word)
#     print(len(remaining_vocab));
#     f = open(gloveFile,'r');
#     model = {}
#     glove_words = [];
#     # w_emb = {};
#     for line in f:
#         splitLine = line.strip().split()
#         word = splitLine[0];
#         # glove_words.append(word);
#         # w_emb[word] = np.array([float(val) for val in splitLine[1:]]);
#         if word in remaining_vocab:
#             embedding = np.array([float(val) for val in splitLine[1:]])
#             model[word_to_index[word]] = embedding
#             # remaining_vocab.remove(word);
    
#     # for word in index_to_word:
#     #     if word in glove_words:
#     #         model[word_to_index[word]] = w_emb[word];
#     #     else:
#     #         print("Word not in glove file:", word);
            
#     print("Done.",len(model)," words loaded!")
    
#     for key in word_to_index.keys():
#         idx = word_to_index[key];
#         if idx not in model.keys():
#             model[idx] = np.random.rand(300);
        
#     return model;

# model = loadGloveModel(glove_path)

# print("Model Length: ", len(model));
# model[word_to_index['<pad>']] = np.zeros((300));
# word_to_index['chili'];
# print(word_to_index['chili']);

# # exit();

# # # Save the embedding to a pickle file 
# f = open("embedding.pkl","wb");
# pickle.dump(model,f);
# f.close();

# f = open("vocab.pkl","wb");
# pickle.dump(word_to_index, f);
# f.close();



# new = pickle.load(open('embedding.pkl','rb'))



# print(new[word_to_index['\'']])