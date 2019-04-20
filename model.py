import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu");

def get_embeddings(sentence_ind, embedding_map):
    sentence_shape = sentence_ind.shape;
    sentence_ind = sentence_ind.view(-1);
    embedding_matrix = torch.tensor([]).to(device);
    for ind in sentence_ind:
        embedding_matrix = torch.cat((embedding_matrix, \
            torch.tensor(embedding_map[int(ind)], dtype=torch.float).to(device)));
    return embedding_matrix.view(sentence_shape[0], sentence_shape[1], -1);

class attention_compute(nn.Module):
    def __init__(self, embedding_dim, feature_dim, embeddings, batch_size=1):
        super(attention_compute, self).__init__();
        self.emb_dim = embedding_dim;
        self.feature_dim = feature_dim;

        self.scores = nn.Linear(embedding_dim + feature_dim, 1);
        self.embedding_map =  embeddings;
        self.softmax = nn.Softmax();

    def forward(self, sen_emb, features, feature_lens):
        # sen_emb = self.get_embeddings(sentence);
        # sen_emb = sen_emb.view(-1, self.emb_dim);
        # print("Sentence Shape: ", sen_emb.shape);
        att_ftr = torch.tensor([], dtype=torch.float).to(device);
        for i in range(sen_emb.shape[0]):
            for j in range(sen_emb.shape[1]):
               att_ftr = torch.cat((att_ftr, self.compute_one(sen_emb[i][j], features[i][:feature_lens[i]]))); 
        att_ftr = att_ftr.view(sen_emb.shape[0], sen_emb.shape[1], -1);
        return att_ftr;

    def get_embeddings(self, sentence):
        sentence_shape = sentence.shape;
        sentence = sentence.view(-1);
        embedding_matrix = torch.tensor([]).to(device);
        for ind in sentence:
            embedding_matrix = torch.cat((embedding_matrix, \
                torch.tensor(self.embedding_map[int(ind)], dtype=torch.float).to(device)));
        return embedding_matrix.view(sentence_shape[0], sentence_shape[1], -1);
    
    def compute_one(self, emb_vector, video):
        # print("Video Shape: ", video.shape);
        score_vec = torch.tensor([]).to(device);
        for feature_vec in video:
            score = self.scores(torch.cat((emb_vector, feature_vec)))
            score_vec = torch.cat((score_vec, score));
        attn_weights = nn.functional.softmax(score_vec.view(-1), dim=0).view(-1, 1);
        result = (attn_weights * video.view(1, video.shape[0], -1)).view(video.shape[0], -1).sum(dim=0);
        return result;
    
class multi_modal_layer(nn.Module):
    def __init__(self, embedding_dim, mfeature_dim, tfeature_dim, embeddings, hidden_dim, \
        vocab_size, batch_size=1, dropout=0.5, kernel=10):
        super(multi_modal_layer, self).__init__();
        self.emb_map = embeddings;
        self.emb_dim = embedding_dim;
        self.mfeat_dim = mfeature_dim;
        self.tfeat_dim = tfeature_dim;
        self.hdim = hidden_dim;
        self.kernel_size = 10;
        self.batch_size = batch_size;
        self.dropout_prob = dropout;

        self.drop_layer = nn.Dropout(p=self.dropout_prob);
        self.inter_motion = nn.Linear(mfeature_dim, mfeature_dim);
        self.inter_temporal = nn.Linear(tfeature_dim, tfeature_dim);
        self.multi_modal = nn.Linear(embedding_dim + mfeature_dim + tfeature_dim, self.hdim);
        self.lstm_init = nn.Linear( int(tfeature_dim/self.kernel_size) \
            + int(mfeature_dim/self.kernel_size), hidden_dim);
        self.lang_layer = nn.LSTM(hidden_dim, hidden_dim, batch_first=True);
        self.lstm_activ = nn.Tanh();
        self.mh_attention = attention_compute(self.hdim, self.mfeat_dim, \
            self.emb_map, batch_size=self.batch_size);
        self.th_attention = attention_compute(self.hdim, self.tfeat_dim, \
            self.emb_map, batch_size=self.batch_size);
        self.multi_modal_2 = nn.Linear(self.hdim + self.tfeat_dim + self.mfeat_dim, vocab_size);

    def forward(self, mfeat, tfeat, att_motion, att_tempo, embeddings, \
        motion_batch_lens, temp_batch_lens, seq_batch_lens):
        temp_mout = self.inter_motion(att_motion);
        temp_tout = self.inter_temporal(att_tempo);
        mout_weights = nn.functional.softmax(temp_mout, dim=len(temp_mout.shape) - 1);
        tout_weights = nn.functional.softmax(temp_mout, dim=len(temp_tout.shape) - 1);
        # print(mout_weights.shape, tout_weights.shape);
        temp_mout = temp_mout * mout_weights;
        temp_tout = temp_tout * tout_weights;

        multi_modal_inp = torch.cat((embeddings, temp_tout, temp_mout), dim=2);
        multi_modal_inp = self.drop_layer(multi_modal_inp);
        # print("Mutimodal Input Shape: ", multi_modal_inp.shape);
        multi_modal_out = self.multi_modal(multi_modal_inp);
        # print("Mutimodal Output Shape: ", multi_modal_out.shape);
        pooling_layer = nn.AvgPool2d(self.kernel_size);
        mean_feat = torch.cat(( pooling_layer(att_tempo), pooling_layer(att_motion)), dim=2);
        # print("Average Pooling Shape: ", mean_feat.shape);
        m_0 = self.lstm_init(mean_feat);
        # print("Initial LSTM input shape: ", m_0.shape);
        _, init_hidden = self.lang_layer(m_0, \
            (torch.zeros(1, att_motion.shape[0], self.hdim).to(device), \
            torch.zeros(1, att_motion.shape[0], self.hdim).to(device)));
        # print("Hidden Size: ", init_hidden[0].shape);
        out, _ = self.lang_layer(multi_modal_out, init_hidden);
        # print("Output before activation shape: ", out.shape);
        act_out = self.lstm_activ(out);
        # print("Output After activation shape: ", out.shape);
        att_motion_2 = self.mh_attention(act_out, mfeat, motion_batch_lens);
        att_tempo_2 = self.th_attention(act_out, tfeat, temp_batch_lens);
        # print("Second Attention Shape: ", att_motion_2.shape, att_tempo_2.shape);
        final_out = self.multi_modal_2(self.drop_layer(torch.cat((act_out, att_tempo_2, att_motion_2), \
            dim=2)));
        # print("Multi Modal Output: ", final_out.shape);
        prob_dist = nn.functional.log_softmax(final_out, dim=2);
        # print("Probability Distribution: ", prob_dist);
        return prob_dist;

