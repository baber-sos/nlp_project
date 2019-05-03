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
from model import multi_modal_layer
import sys
from dataloader import custom_ds
# from pycocoeval

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# nltk.download('punkt');

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu");

# # getting all videos' names
# videos = []
# with open('./input') as f:
#     for i in f:
#         # extract names of the videos 
#         names = i.split('.')[0]
#         videos.append(names)

# # Reading catptions
# captions = pd.read_csv('video_corpus.csv')
# english_captions = captions[captions['Language'] == 'English']

# english_captions['NewID'] =  english_captions['VideoID'].str.cat(english_captions['Start'].values.astype(str), sep='_')
# english_captions['NewID'] =  english_captions['NewID'].str.cat(english_captions['End'].values.astype(str), sep='_')



# # get all the training data based on the ids on videos 
# trained = english_captions.loc[english_captions['NewID'].isin(videos)]


# ### Getting all words for the Description 
# vectorizer = sklearn.feature_extraction.text.CountVectorizer(min_df=1);
# X = vectorizer.fit_transform(trained['Description'])
# index_to_word = vectorizer.get_feature_names()
# # print('a' in index_to_word);
# # print(index_to_word);
# ix_to_word = dict();
# for (ix, word) in enumerate(index_to_word):
#     ix_to_word[ix] = word;
# ix_to_word[len(ix_to_word)] = pad_word;
# ix_to_word[len(ix_to_word)] = unk_word;
# ix_to_word[len(ix_to_word)] = start_word;
# ix_to_word[len(ix_to_word)] = end_word;
# word_to_index = {ix_to_word[i]:i for i in range(len(ix_to_word))}
# embeddings = pickle.load(open('embedding.pkl', 'rb'));


# embeddings[word_to_index[unk_word]] = [0.0 for i in range(embedding_dim)];
# embeddings[word_to_index[pad_word]] = [0.0 for i in range(embedding_dim)];
# embeddings[word_to_index[start_word]] = [0.0 for i in range(embedding_dim)];
# embeddings[word_to_index[end_word]] = [0.0 for i in range(embedding_dim)];

# # print(len(trained));
# temporal_data = dict();
# with open('temporal.json') as temp_data:
#     temporal_data = json.load(temp_data);

# motion_data = dict();
# with open('motion.json') as temp_data:
#     motion_data = json.load(temp_data);

# def seq2ind(seq, word_to_index, embeddings, unk_word):
#     indices = [];
#     for word in seq:
#         word = word.lower();
#         if word in word_to_index and word_to_index[word] in embeddings:
#             indices.append(word_to_index[word]);
#         else:
#             indices.append(word_to_index[unk_word]);
#     return indices;


print("Gonna Create the dataset!");
video_ds = custom_ds('input', 'video_corpus.csv', 'embedding.pkl', batch_size=8);

##configurable params
vocab_size = len(video_ds.word_to_index);
batch_size = 8;
embedding_dim = 300;
hidden_dim = 512;
learn_rate = 10**(-4);
dropout_prob = 0.5;
percentage = 0.90;
train_set = int(len(video_ds.trained));
# train_set = train_set - (train_set % batch_size);
# val_set = len(video_ds.trained) - train_set
epochs = 5;

pad_word = '<pad>';
start_word = '<str>';
end_word = '<end>';

data_loader = torch.utils.data.DataLoader(video_ds, batch_size=batch_size, \
    shuffle=True, num_workers=batch_size);

frame_attn_model = attention_compute(embedding_dim, 512, batch_size=batch_size).to(device);
motion_attn_model = attention_compute(embedding_dim, 512, batch_size=batch_size).to(device);
mmodel = multi_modal_layer(embedding_dim, 512, 512, hidden_dim, vocab_size, \
    batch_size=batch_size, dropout=dropout_prob).to(device);
loss_fn = nn.NLLLoss(reduction='sum', ignore_index=video_ds.word_to_index[pad_word])

# train_loss_epoch = 0.0;
# val_loss_epoch = 0.0;

if len(sys.argv) > 1 and sys.argv[1] == 'load':
    frame_attn_model.load_state_dict(torch.load("temporal_attention.pt"));
    motion_attn_model.load_state_dict(torch.load("motion_attention.pt"));
    mmodel.load_state_dict(torch.load("multi_modal_attention.pt"));

optimizer = optim.RMSprop([p for p in frame_attn_model.parameters()] + \
    [p for p in motion_attn_model.parameters()] + [p for p in mmodel.parameters()] , lr=learn_rate);
#out_file = open("loss_tracking.txt", "w");

count = 0;

print("Training is about to start!");
for epoch in range(epochs):
    train_loss_epoch = 0.0;
    val_loss_epoch = 0.0;
    count = 0;
    for batch in data_loader:
        frame_attn_model.zero_grad();
        motion_attn_model.zero_grad();
        mmodel.zero_grad();
        if count % 10 == 0:
            print("-------------------------------------");
            print("Train Loss till Now: ", train_loss_epoch);
            print("Val Loss till Now: ", val_loss_epoch);
            print("-------------------------------------");
        
        sen_emb, temp_feats, mot_feats, seq_lens, temp_lens, mot_lens, targets, _ = batch;
        
        max_slen = torch.max(seq_lens); 
        max_tlen = torch.max(temp_lens); 
        max_mlen = torch.max(mot_lens);

        batch_emb = sen_emb[:, :max_slen].to(device); 
        batch_temp = temp_feats[:, :max_tlen].to(device); 
        batch_mot = mot_feats[:, :max_mlen].to(device);


        targets = targets[:, :max_slen].to(device);
        print("Start of the Iteration %d" % count);
        with torch.set_grad_enabled(count < train_set):
            attn_tfeat = frame_attn_model(batch_emb, batch_temp, temp_lens.to(device));
            attn_mfeat = motion_attn_model(batch_emb, batch_mot, mot_lens.to(device));
            prob_dist = mmodel(batch_mot, batch_temp, attn_mfeat, attn_tfeat, \
                batch_emb, mot_lens.to(device), temp_lens.to(device), seq_lens.to(device));
            targets = targets.contiguous();
            targets = targets.view(-1);
            prob_dist = prob_dist.view(-1, vocab_size);
            
            batch_loss = loss_fn(prob_dist, targets);

        if count < train_set:
            train_loss_epoch += batch_loss.data;
            batch_loss /= batch_size;
            batch_loss.backward();
            optimizer.step();
        else:
            val_loss_epoch += batch_loss.data;
        del batch_loss, prob_dist;
        count += batch_size;
        print("End of Iteration!");
    torch.save(frame_attn_model.state_dict(), "temporal_attention.pt");
    torch.save(motion_attn_model.state_dict(), "motion_attention.pt");
    torch.save(mmodel.state_dict(), "multi_modal_attention.pt");
    print("Train loss for Epoch %d is %.3f" % (epoch + 1, train_loss_epoch/train_set));
    # print("Validation loss for Epoch %d is %.3f" % (epoch + 1, val_loss_epoch/val_set));
    print("*************************");
    out_file = open("loss_tracking.txt", 'a');
    out_file.write("Train Loss for Epoch %d: %f\n" % (epoch + 1, train_loss_epoch/train_set));
    # out_file.write("Validation Loss for Epoch %d: %f\n" % (epoch + 1, val_loss_epoch/val_set));
    out_file.write("--------------\n");
    out_file.close();

# print("Total Number of Entries: ", len(trained));
# print("train set: %d interations: %d", train_set, train_set/batch_size);
# print("Gonna Start Training!");
# for epoch in range(epochs):
#     for i in range(0, len(trained), batch_size):
#         if i % 10 == 0:
#             print("Train Loss till Iteration %d is %f" % (i + 1, train_loss_epoch));
#             print("--------------");
#         frame_attn_model.zero_grad();
#         motion_attn_model.zero_grad();
#         mmodel.zero_grad();
#         batch_frame_tensor = torch.tensor([]).to(device);
#         batch_seq_ind = torch.tensor([], dtype=torch.long).to(device);
#         batch_motion_frames = torch.tensor([]).to(device);

#         max_frames = 0;
#         max_cap_len = 0;
#         max_motion_frames = 0;

#         batch_num_frames = [];
#         batch_caption_lens = [];
#         batch_motion_lens= [];

#         for b_ele in range(i, i+batch_size):
#             if b_ele >= len(trained):
#                 continue;
#             video_name = trained['NewID'].iloc[b_ele] + '.avi';

#             batch_num_frames.append(len(temporal_data[video_name]));
#             max_frames = max(max_frames, len(temporal_data[video_name]));
            
#             vectorizer = sklearn.feature_extraction.text.CountVectorizer(min_df=1,stop_words='english');
#             caption = word_tokenize(trained['Description'].iloc[b_ele]);
#             batch_caption_lens.append(len(caption) + 2);
#             max_cap_len = max(max_cap_len, len(caption) + 2);

#             cur_len = 0;
#             for vid in motion_data:
#                 if vid['video'] == video_name:
#                     cur_len = len(vid['clips']);
#                     break;
#             batch_motion_lens.append(cur_len);
#             max_motion_frames = max(max_motion_frames, cur_len);

#         for b_ele in range(i, i+batch_size):
#             if b_ele >= len(trained):
#                 continue;
#             cur_tensor = torch.tensor([]).to(device);
#             video_name = trained['NewID'].iloc[b_ele] + '.avi';
#             caption = word_tokenize(trained['Description'].iloc[b_ele]);
#             caption = ['<str>'] + caption + ['<end>'];
#             caption_ind = seq2ind(caption, word_to_index, embeddings, unk_word);
#             caption_ind = caption_ind + \
#                 [word_to_index[pad_word] for i in range(max_cap_len - batch_caption_lens[b_ele - i])];
#             batch_seq_ind = torch.cat((batch_seq_ind, torch.tensor(caption_ind, dtype=torch.long).to(device)));
#             # print(batch_seq_ind.shape, max_cap_len);
#             this_dict = temporal_data[video_name];
#             cur_dim = len(this_dict['1']);
#             for frame_num in range(max_frames):
#                 if frame_num < len(this_dict):
#                     cur_tensor = torch.cat((cur_tensor, torch.tensor(this_dict[str(frame_num + 1)], \
#                         dtype=torch.float).to(device)));
#                 else:
#                     cur_tensor = torch.cat((cur_tensor, torch.zeros(cur_dim).to(device)));
#             cur_tensor = cur_tensor.view(max_frames, -1);
#             batch_frame_tensor = torch.cat((batch_frame_tensor, cur_tensor));
            
#             cur_mot_dict = dict();
#             for vid in motion_data:
#                 if vid['video'] == video_name:
#                     cur_mot_dict = vid;
#                     break;
#             cur_mot_len = batch_motion_lens[b_ele - i];
#             motion_dim = len(motion_data[0]['clips'][0]['features']);
#             # print("Motion Dimension: ", motion_dim);
#             cur_motion_frames = [x['features'] for x in cur_mot_dict['clips']];
#             cur_motion_frames = cur_motion_frames + \
#                 [[0.0 for num_zero in range(motion_dim)] for j in range(max_motion_frames - cur_mot_len)];
#             batch_motion_frames = torch.cat((batch_motion_frames, \
#                 torch.tensor(cur_motion_frames).to(device)));

#         print("Batch Dimensions: ", batch_frame_tensor.shape, batch_motion_frames.shape);
#         # print()
#         batch_frame_tensor = batch_frame_tensor.view(batch_size, max_frames, -1).to(device);
#         batch_seq_ind = batch_seq_ind.view(batch_size, -1);
#         targets = batch_seq_ind.clone().to(device).view(-1);
#         batch_motion_frames = batch_motion_frames.view(batch_size, max_motion_frames, -1).to(device);
#         batch_embeddings = get_embeddings(batch_seq_ind, embeddings).to(device);
#         with torch.set_grad_enabled(i < train_set):
#             attn_tfeat = frame_attn_model(batch_embeddings, batch_frame_tensor, \
#                 batch_num_frames);
#             attn_mfeat = motion_attn_model(batch_embeddings, batch_motion_frames, \
#                 batch_motion_lens);
#             prob_dist = mmodel(batch_motion_frames, batch_frame_tensor, attn_mfeat, attn_tfeat, \
#                 batch_embeddings, batch_motion_lens, batch_num_frames, batch_caption_lens);
        
#         prob_dist = prob_dist.view(-1, vocab_size);
#         total_loss = loss_fn(prob_dist, targets);
#         if i < train_set:
#             total_loss.backward();
#             optimizer.step();
#             train_loss_epoch += float(total_loss);
#         else:
#             val_loss_epoch += float(total_loss);
#         del total_loss, prob_dist;
#         print("On to the Next Iteration!");
#     print("Train Loss for Epoch %d: %f" % (epoch + 1, train_loss_epoch/train_set));
#     print("Validation Loss for Epoch %d: %f" % (epoch + 1, val_loss_epoch));
#     print("--------------");
#     out_file.write("Train Loss for Epoch %d: %f\n" % (epoch + 1, train_loss_epoch/train_set));
#     out_file.write("Validation Loss for Epoch %d: %f\n" % (epoch + 1, val_loss_epoch));
#     out_file.write("--------------\n");
#     torch.save(frame_attn_model.state_dict(), "temporal_attention.pt");
#     torch.save(motion_attn_model.state_dict(), "motion_attention.pt");
#     torch.save(mmodel.state_dict(), "multi_modal_attention.pt");
