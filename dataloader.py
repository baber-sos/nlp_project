import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import json
import pandas as pd
import pickle
from nltk.tokenize import word_tokenize
from collections import OrderedDict


device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu");
dev_2 = torch.device("cuda:3" if torch.cuda.is_available() else "cpu");

class custom_ds(data.Dataset):
    def __init__(self, video_file, csv_path, emb_file, vocab_file, batch_size, \
        temporal_file='temporal.json', motion_file='motion.json', trainable=False):
        super(custom_ds, self).__init__();
        self.batch_size = batch_size;
        self.temporal_file = temporal_file;
        self.motion_file = motion_file;

        self.index_to_vid = []
        with open(video_file) as f:
            for i in f:
                names = i.split('.')[0]
                # names = i.strip();
                self.index_to_vid.append(names)
        captions = pd.read_csv(csv_path);
        english_captions = captions[captions['Language'] == 'English'];
        english_captions['NewID'] =  english_captions['VideoID'].str.\
            cat(english_captions['Start'].values.astype(str), sep='_');
        english_captions['NewID'] =  english_captions['NewID'].str.\
            cat(english_captions['End'].values.astype(str), sep='_');
        self.trained = english_captions.loc[english_captions['NewID'].isin(self.index_to_vid)];
        print("Length of all dps: ", len(self.trained));
        
        for i in range(len(self.index_to_vid)):
            self.index_to_vid[i] += '.avi';
        
        self.vid_to_ix = {self.index_to_vid[i] : i for i in range(len(self.index_to_vid))};
        
        with open(vocab_file, 'rb') as vfile:
            self.word_to_index = pickle.load(vfile);
        ix_to_word = {self.word_to_index[i] : i for i in self.word_to_index.keys()};
        self.index_to_word = [];
        for i in range(len(self.word_to_index)):
            self.index_to_word.append(ix_to_word[i]);
        self.word_to_index = {self.index_to_word[i] : int(i) for i in range(len(self.index_to_word))}

        self.max_caption_len = 0;
        for item in self.trained['Description']:
            self.max_caption_len = max(self.max_caption_len, len(word_tokenize(item))+2);
        # data = english_captions['Description']
        # data = data.str.lower()
        # data = list(data)
        # new = data
        # words = [];
        # self.max_caption_len = 0;
        # for item in new:
        #     try:
        #         self.max_caption_len = max(self.max_caption_len, \
        #             len(word_tokenize(self.remove(item))) + 2);
        #         words += (word_tokenize(item));
        #     except:
        #         continue

        # for i in range(len(words)):
        #     words[i] = self.remove(words[i]);

        # print("Number of Words: ", len(words));
        # self.index_to_word = list(OrderedDict.fromkeys(words));
        # self.index_to_word.append("<sos>");
        # self.index_to_word.append("<eos>");
        # self.index_to_word.append("<pad>");
        # for item in words:
        #     for word in item:
        #         word = self.remove(word);
        #         if word not in self.index_to_word:
        #             self.index_to_word.append(word);
        
        # self.word_to_index = {self.index_to_word[i]:i for i in range(len(self.index_to_word))};

        embedding_map = pickle.load(open(emb_file, 'rb'));
        print("Index to Word Length: ", len(self.index_to_word));
        print("Word to Index Length: ", len(self.word_to_index));
        print("Embedding Map Length: ", len(embedding_map));
        for idx in range(len(self.index_to_word)):
            if idx not in embedding_map:
                print("Word not in embeddings: ", self.index_to_word[idx]);
                embedding_map[idx] = np.random.rand(len(embedding_map[0]));

        weights = np.array([embedding_map[i] for i in range(len(self.index_to_word))], ndmin=2);
        print('Weights Shape: ', weights.shape);

        self.embedding_dim = len(embedding_map[0]);
        self.embeddings = nn.Embedding(len(self.index_to_word), self.embedding_dim);
        
        self.embeddings.load_state_dict({'weight' : torch.from_numpy(weights)});
        self.embeddings.weight.requires_grad = trainable;
        # print("Embedding Test: ", self.embeddings(torch.tensor([[0]])).shape);
        with open(self.temporal_file) as temp_data:
            temporal_features = json.load(temp_data);
        
        with open(self.motion_file) as temp_data:
            motion_features = json.load(temp_data);
        
        print("Done loading features from files!");
        self.temp_ix_to_len = [];
        max_num_frames = 0;
        for video in self.index_to_vid:
            # video += '.avi';
            self.temp_ix_to_len.append(len(temporal_features[video]));
            max_num_frames = max(max_num_frames, len(temporal_features[video]));
        print("Number of videos: ", len(self.temp_ix_to_len));
        print("Number of videos confirmed: ", len(self.index_to_vid));
        print("Max number of frames: ", max_num_frames);
        self.temp_feat_matrix = torch.tensor([]);
        for video in self.index_to_vid:
            cur_num_frames = len(temporal_features[video]);
            cur_matrix = [temporal_features[video][str(i+1)] for i in range(cur_num_frames)];
            cur_matrix = cur_matrix + [[0.0 for i in range(len(temporal_features[video]['1']))] \
                for j in range(max_num_frames - cur_num_frames)];
            self.temp_feat_matrix = torch.cat((self.temp_feat_matrix, torch.tensor(cur_matrix, \
                dtype=torch.float)));
        self.temp_feat_matrix = self.temp_feat_matrix.view(len(self.index_to_vid), max_num_frames, -1);
        print("Temporal Feature Matrix Shape: ", self.temp_feat_matrix.shape);

        self.motion_ix_to_len = [];
        max_motion_frames = 0;
        for video in motion_features:
            self.motion_ix_to_len.append(len(video['clips']));
            max_motion_frames = max(max_motion_frames, len(video['clips']));

        m_weights = dict();
        for video in motion_features:
            ix = self.vid_to_ix[video['video']];
            cur_weights = [x['features'] for x in video['clips']];
            cur_weights = cur_weights + [[0.0 for i in range(len(video['clips'][0]['features']))] \
                for j in range(max_motion_frames - len(video['clips']))];
            m_weights[ix] = torch.tensor(cur_weights, dtype=torch.float);
        print("Length and shape of motion weights: ", len(m_weights), m_weights[0].shape);
        print("Max Motion Frames: ", max_motion_frames);
        self.motion_feat_matrix = torch.tensor([]);
        for ix in range(len(self.index_to_vid)):
            self.motion_feat_matrix = torch.cat((self.motion_feat_matrix, m_weights[ix]));
        self.motion_feat_matrix = self.motion_feat_matrix.view(len(self.index_to_vid), \
            max_motion_frames, -1);
        print("Motion Features Shape: ", self.motion_feat_matrix.shape);

    def remove(self, my_str):
        punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
        no_punct = ""
        if len(my_str) == 1:
            return my_str;
        for char in my_str:
            if char not in punctuations:
                no_punct = no_punct + char
        return no_punct

    def __len__(self):
        return len(self.trained);
    
    def __getitem__(self, idx):
        # print("len of all dps: ", len(self.trained));
        # print("Index: ", idx);
        video_name = self.trained['NewID'].iloc[idx] + '.avi';
        # print("Video Name: ", video_name);
        caption = word_tokenize(self.trained['Description'].iloc[idx]);

        for i in range(len(caption)):
            this_word = self.remove(caption[i].lower());
            if this_word in self.index_to_word:
                caption[i] = this_word;
            else:
                caption[i] = '<unk>'
        caption = ['<sos>'] + caption + ['<eos>'];
        caption = [x.lower() for x in caption];
        
        # print(caption);
        seq_ind = torch.tensor([self.word_to_index[x] for x in caption]);
        difference = self.max_caption_len - len(seq_ind);
        vid_ind = self.vid_to_ix[video_name];
        emb_zeros = torch.tensor([]);
        target = seq_ind.clone()
        if self.batch_size > 1:
            target = torch.cat((target, \
                torch.tensor([int(self.word_to_index['<pad>']) for i in range(difference)], 
                dtype=torch.long)));
            emb_zeros = torch.zeros(difference, self.embedding_dim);

        ix_embeddings = torch.cat(((self.embeddings(seq_ind), emb_zeros)));
        # del emb_zeros, seq_ind;
        # print(caption);
        return ix_embeddings, self.temp_feat_matrix[vid_ind],\
            self.motion_feat_matrix[vid_ind], len(caption), self.temp_ix_to_len[vid_ind],\
                self.motion_ix_to_len[vid_ind], target, video_name;

# if __name__ == '__main__':
#     my_ds = custom_ds('input', 'video_corpus.csv', 'embedding.pkl');
#     my_dl = torch.utils.data.DataLoader(my_ds, batch_size=2, num_workers=1, shuffle=True)
#     count = 0;
#     for i in my_dl:
#         print( i[3] );
#     print("Total Count: ", count)