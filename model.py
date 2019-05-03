import torch
import torch.nn as nn
# from torch.nn.utils.rnn import pack_padded_sequence
# from torch.nn.utils.rnn import pad_packed_sequence
# from torch.nn.utils.rnn import PackedSequence

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu");

class attention_compute(nn.Module):
    def __init__(self, embedding_dim, feature_dim, batch_size=1):
        super(attention_compute, self).__init__();
        self.emb_dim = embedding_dim;
        self.feature_dim = feature_dim;

        self.scores = nn.Linear(embedding_dim + feature_dim, 1);
        self.softmax = nn.Softmax();

    def forward(self, sen_emb, features, feature_lens):
        att_ftr = torch.tensor([], dtype=torch.float).to(device);
        for i in range(sen_emb.shape[0]):
            for j in range(sen_emb.shape[1]):
               att_ftr = torch.cat((att_ftr, self.compute_one(sen_emb[i][j], features[i][:feature_lens[i]]))); 
        att_ftr = att_ftr.view(sen_emb.shape[0], sen_emb.shape[1], -1);
        return att_ftr;

    def compute_one(self, emb_vector, video):
        # print("Video Shape: ", video.shape);
        score_vec = torch.tensor([]).to(device);
        for feature_vec in video:
            score = self.scores(torch.cat((emb_vector, feature_vec)))
            score_vec = torch.cat((score_vec, score));
        attn_weights = nn.functional.softmax(score_vec.view(-1), dim=0).view(-1, 1);
        result = (attn_weights * video.view(1, video.shape[0], -1)).view(video.shape[0], -1).sum(dim=0);
        # del attn_weights, score_vec;
        return result;
    
class multi_modal_layer(nn.Module):
    def __init__(self, embedding_dim, mfeature_dim, tfeature_dim, hidden_dim, \
        vocab_size, batch_size=1, dropout=0.5, kernel=10):
        super(multi_modal_layer, self).__init__();
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
            batch_size=self.batch_size);
        self.th_attention = attention_compute(self.hdim, self.tfeat_dim, \
            batch_size=self.batch_size);
        self.multi_modal_2 = nn.Linear(self.hdim + self.tfeat_dim + self.mfeat_dim, vocab_size);

    def forward(self, mfeat, tfeat, att_motion, att_tempo, embeddings, \
        motion_batch_lens, temp_batch_lens, seq_batch_lens):


        temp_mout = self.inter_motion(att_motion);
        temp_tout = self.inter_temporal(att_tempo);
        mout_weights = nn.functional.softmax(temp_mout, dim=len(temp_mout.shape) - 1);
        tout_weights = nn.functional.softmax(temp_tout, dim=len(temp_tout.shape) - 1);
        temp_mout = temp_mout * mout_weights;
        temp_tout = temp_tout * tout_weights;

        multi_modal_inp = torch.cat((embeddings, temp_tout, temp_mout), dim=2);
        multi_modal_inp = self.drop_layer(multi_modal_inp);
        multi_modal_out = self.multi_modal(multi_modal_inp);
        pooling_layer = nn.AvgPool2d(self.kernel_size);
        mean_feat = torch.cat(( pooling_layer(att_tempo), pooling_layer(att_motion)), dim=2);

        m_0 = self.lstm_init(mean_feat);
        _, init_hidden = self.lang_layer(m_0, \
            (torch.zeros(1, att_motion.shape[0], self.hdim).to(device), \
            torch.zeros(1, att_motion.shape[0], self.hdim).to(device)));
        out, _ = self.lang_layer(multi_modal_out, init_hidden);
        act_out = self.lstm_activ(out);
        
        att_motion_2 = self.mh_attention(act_out, mfeat, motion_batch_lens);
        att_tempo_2 = self.th_attention(act_out, tfeat, temp_batch_lens);
        final_out = self.multi_modal_2(self.drop_layer(torch.cat((act_out, att_tempo_2, att_motion_2), \
            dim=2)));
        prob_dist = nn.functional.log_softmax(final_out, dim=2);

        # del temp_mout, mout_weights, tout_weights, 
        return prob_dist;

