import numpy as np 
import pandas as pd 
import sklearn
import random
import io
import numbers
import sklearn.feature_extraction.text
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import json
import nltk
from nltk.tokenize import word_tokenize
from model import attention_compute
from model import get_embeddings
from model import multi_modal_layer
import sys
# from pycocoeval

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

nltk.download('punkt');

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

english_captions['NewID'] =  english_captions['VideoID'].str.cat(english_captions['Start'].values.astype(str), sep='_')
english_captions['NewID'] =  english_captions['NewID'].str.cat(english_captions['End'].values.astype(str), sep='_')



# get all the training data based on the ids on videos 
trained = english_captions.loc[english_captions['NewID'].isin(videos)]

pad_word = '<pad>';
unk_word = '<unk>';
start_word = '<str>';
end_word = '<end>';
### Getting all words for the Description 
vectorizer = sklearn.feature_extraction.text.CountVectorizer(min_df=1);
X = vectorizer.fit_transform(trained['Description'])
index_to_word = vectorizer.get_feature_names()
# print('a' in index_to_word);
# print(index_to_word);
ix_to_word = dict();
for (ix, word) in enumerate(index_to_word):
    ix_to_word[ix] = word;
ix_to_word[len(ix_to_word)] = pad_word;
ix_to_word[len(ix_to_word)] = unk_word;
ix_to_word[len(ix_to_word)] = start_word;
ix_to_word[len(ix_to_word)] = end_word;
word_to_index = {ix_to_word[i]:i for i in range(len(ix_to_word))}
embeddings = pickle.load(open('embedding.pkl', 'rb'));


##configurable params
vocab_size = len(word_to_index);
batch_size = 5;
embedding_dim = 300;
hidden_dim = 512;
learn_rate = 10**(-4);
dropout_prob = 0.5;
percentage = 0.90;
train_set = int(len(trained) * percentage);
val_set = len(trained) - int(len(trained) * percentage);
epochs = 2;

embeddings[word_to_index[unk_word]] = [0.0 for i in range(embedding_dim)];
embeddings[word_to_index[pad_word]] = [0.0 for i in range(embedding_dim)];
embeddings[word_to_index[start_word]] = [0.0 for i in range(embedding_dim)];
embeddings[word_to_index[end_word]] = [0.0 for i in range(embedding_dim)];

# print(len(trained));
temporal_data = dict();
with open('temporal.json') as temp_data:
    temporal_data = json.load(temp_data);

motion_data = dict();
with open('motion.json') as temp_data:
    motion_data = json.load(temp_data);

def seq2ind(seq, word_to_index, embeddings, unk_word):
    indices = [];
    for word in seq:
        word = word.lower();
        if word in word_to_index and word_to_index[word] in embeddings:
            indices.append(word_to_index[word]);
        else:
            indices.append(word_to_index[unk_word]);
    return indices;

frame_attn_model = attention_compute(embedding_dim, 512, embeddings, batch_size=batch_size).to(device);
motion_attn_model = attention_compute(embedding_dim, 512, embeddings, batch_size=batch_size).to(device);
mmodel = multi_modal_layer(embedding_dim, 512, 512, embeddings, hidden_dim, vocab_size, \
    batch_size=batch_size, dropout=dropout_prob).to(device);
loss_fn = nn.NLLLoss(ignore_index=word_to_index[pad_word])
optimizer = optim.RMSprop([p for p in frame_attn_model.parameters()] + \
    [p for p in motion_attn_model.parameters()] + [p for p in mmodel.parameters()] , lr=learn_rate);

train_loss_epoch = 0.0;
val_loss_epoch = 0.0;

if len(sys.argv) > 1 and sys.argv[1] == 'load':
    frame_attn_model.load_state_dict(torch.load("temporal_attention.pt"));
    motion_attn_model.load_state_dict(torch.load("motion_attention.pt"));
    mmodel.load_state_dict(torch.load("multi_modal_attention.pt"));

print("Total Number of Entries: ", len(trained));
print("Gonna Start Training!");
for epoch in range(epochs):
    for i in range(0, len(trained), batch_size):
        if i % 10 == 0:
            print("Train Loss till Iteration %d is %f" % (i + 1, train_loss_epoch));
            print("--------------");
        frame_attn_model.zero_grad();
        motion_attn_model.zero_grad();
        mmodel.zero_grad();
        batch_frame_tensor = torch.tensor([]);
        batch_seq_ind = torch.tensor([], dtype=torch.long);
        batch_motion_frames = torch.tensor([]);

        max_frames = 0;
        max_cap_len = 0;
        max_motion_frames = 0;

        batch_num_frames = [];
        batch_caption_lens = [];
        batch_motion_lens= [];

        for b_ele in range(i, i+batch_size):
            video_name = trained['NewID'].iloc[b_ele] + '.avi';

            batch_num_frames.append(len(temporal_data[video_name]));
            max_frames = max(max_frames, len(temporal_data[video_name]));
            
            vectorizer = sklearn.feature_extraction.text.CountVectorizer(min_df=1,stop_words='english');
            caption = word_tokenize(trained['Description'].iloc[b_ele]);
            batch_caption_lens.append(len(caption) + 2);
            max_cap_len = max(max_cap_len, len(caption) + 2);

            cur_len = 0;
            for vid in motion_data:
                if vid['video'] == video_name:
                    cur_len = len(vid['clips']);
                    break;
            batch_motion_lens.append(cur_len);
            max_motion_frames = max(max_motion_frames, cur_len);

        for b_ele in range(i, i+batch_size):
            cur_tensor = torch.tensor([]);
            video_name = trained['NewID'].iloc[b_ele] + '.avi';
            caption = word_tokenize(trained['Description'].iloc[b_ele]);
            caption = ['<str>'] + caption + ['<end>'];
            caption_ind = seq2ind(caption, word_to_index, embeddings, unk_word);
            caption_ind = caption_ind + \
                [word_to_index[pad_word] for i in range(max_cap_len - batch_caption_lens[b_ele - i])];
            batch_seq_ind = torch.cat((batch_seq_ind, torch.tensor(caption_ind, dtype=torch.long)));
            # print(batch_seq_ind.shape, max_cap_len);
            this_dict = temporal_data[video_name];
            cur_dim = len(this_dict['1']);
            for frame_num in range(max_frames):
                if frame_num < len(this_dict):
                    cur_tensor = torch.cat((cur_tensor, torch.tensor(this_dict[str(frame_num + 1)], \
                        dtype=torch.float)));
                else:
                    cur_tensor = torch.cat((cur_tensor, torch.zeros(cur_dim)));
            cur_tensor = cur_tensor.view(max_frames, -1);
            batch_frame_tensor = torch.cat((batch_frame_tensor, cur_tensor));
            
            cur_mot_dict = dict();
            for vid in motion_data:
                if vid['video'] == video_name:
                    cur_mot_dict = vid;
                    break;
            cur_mot_len = batch_motion_lens[b_ele - i];
            motion_dim = len(motion_data[0]['clips'][0]['features']);
            # print("Motion Dimension: ", motion_dim);
            cur_motion_frames = [x['features'] for x in cur_mot_dict['clips']];
            cur_motion_frames = cur_motion_frames + \
                [[0.0 for num_zero in range(motion_dim)] for j in range(max_motion_frames - cur_mot_len)];
            batch_motion_frames = torch.cat((batch_motion_frames, \
                torch.tensor(cur_motion_frames)));

        batch_frame_tensor = batch_frame_tensor.view(batch_size, max_frames, -1).to(device);
        batch_seq_ind = batch_seq_ind.view(batch_size, -1);
        targets = batch_seq_ind.clone().to(device).view(-1);
        batch_motion_frames = batch_motion_frames.view(batch_size, max_motion_frames, -1).to(device);
        batch_embeddings = get_embeddings(batch_seq_ind, embeddings).to(device);
        with torch.set_grad_enabled(i < train_set):
            attn_tfeat = frame_attn_model(batch_embeddings, batch_frame_tensor, \
                batch_num_frames);
            attn_mfeat = motion_attn_model(batch_embeddings, batch_motion_frames, \
                batch_motion_lens);
            prob_dist = mmodel(batch_motion_frames, batch_frame_tensor, attn_mfeat, attn_tfeat, \
                batch_embeddings, batch_motion_lens, batch_num_frames, batch_caption_lens);
        
        prob_dist = prob_dist.view(-1, vocab_size);
        total_loss = loss_fn(prob_dist, targets);
        if i < train_set:
            total_loss.backward();
            optimizer.step();
            train_loss_epoch += total_loss;
        else:
            val_loss_epoch += total_loss;
    print("Train Loss for Epoch %d: %f" % (epoch + 1, train_loss_epoch/train_set));
    print("Validation Loss for Epoch %d: %f" % (epoch + 1, val_loss_epoch));
    print("--------------");
    torch.save(frame_attn_model.state_dict(), "temporal_attention.pt");
    torch.save(motion_attn_model.state_dict(), "motion_attention.pt");
    torch.save(mmodel.state_dict(), "multi_modal_attention.pt");
