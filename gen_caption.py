from model import attention_compute
from model import multi_modal_layer
from dataloader import custom_ds
import torch
import torch.nn as nn
import heapq
import copy

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu");

prediction_set = custom_ds('val_input', 'video_corpus.csv', 'embedding.pkl', batch_size=1, \
    temporal_file='val_temporal.json', motion_file='val_motion.json');

embedding_dim = 300;
hidden_dim = 512;
vocab_size = len(prediction_set.word_to_index);
dropout_prob = 0.5;
batch_size = 1;

start_token = "<sos>"
end_token = "<eos>"
pad_word = "<pad>"
str_ix = prediction_set.word_to_index[start_token];
end_ix = prediction_set.word_to_index[end_token];

data_loader = torch.utils.data.DataLoader(prediction_set, batch_size=batch_size, \
    shuffle=True, num_workers=batch_size);

frame_attn_model = attention_compute(embedding_dim, 512, batch_size=batch_size).to(device);
motion_attn_model = attention_compute(embedding_dim, 512, batch_size=batch_size).to(device);
mmodel = multi_modal_layer(embedding_dim, 512, 512, hidden_dim, vocab_size, \
    batch_size=batch_size, dropout=dropout_prob).to(device);
loss_fn = nn.NLLLoss(reduction='sum', ignore_index=prediction_set.word_to_index[pad_word]);

frame_attn_model.load_state_dict(torch.load("temporal_attention.pt"));
motion_attn_model.load_state_dict(torch.load("motion_attention.pt"));
mmodel.load_state_dict(torch.load("multi_modal_attention.pt"));

val_loss_epoch = 0.0;
val_set = len(data_loader);
count = 0;

max_seqlen = 20;

count = 0;
#cost, next state, tracker, previous
state_queue = [(0, count, str_ix, [0])]
heapq.heapify(state_queue);
visited = [];
beam_size = 5;

for batch in data_loader:
    frame_attn_model.zero_grad();
    motion_attn_model.zero_grad();
    mmodel.zero_grad();
    
    _, temp_feats, mot_feats, seq_lens, temp_lens, mot_lens, targets, vid_name = batch;
    print("Video Name: ", vid_name);
    max_slen = torch.max(seq_lens);
    max_tlen = torch.max(temp_lens);
    max_mlen = torch.max(mot_lens);

    batch_temp = temp_feats[:, :max_tlen].to(device); 
    batch_mot = mot_feats[:, :max_mlen].to(device);
    while len(state_queue) > 0:
        cur_state = heapq.heappop(state_queue);

        if cur_state[1] == prediction_set.word_to_index["<eos>"]:
            break;
        elif len(cur_state[-1]) > 20:
            break;

        visited.append(cur_state[1]);
        # print("Cur Index and Word: ", prediction_set.index_to_word[cur_state[2]]);
        seq_start = torch.tensor([cur_state[1]], dtype=torch.long);

        batch_emb = prediction_set.embeddings(seq_start).view(batch_size, 1, embedding_dim).to(device);

        print("Start of the Iteration %d" % count);
        with torch.set_grad_enabled(False):
            attn_tfeat = frame_attn_model(batch_emb, batch_temp, temp_lens.to(device));
            attn_mfeat = motion_attn_model(batch_emb, batch_mot, mot_lens.to(device));
            prob_dist = mmodel(batch_mot, batch_temp, attn_mfeat, attn_tfeat, \
                batch_emb, mot_lens.to(device), temp_lens.to(device), seq_lens.to(device));
            prob_dist = prob_dist.view(-1, vocab_size);
            costs, next_states = torch.topk(prob_dist, beam_size, dim=1);
        
        costs = costs.view(-1);
        next_states = next_states.view(-1);

        for i, ncost in enumerate(costs):
            nstate = int(next_states[i]);
            if int(nstate) in visited:
                continue;
            count += 1;
            cur_state[3].append(nstate);
            est_cost = -float(cur_state[0] + ncost);
            state_queue.append( (est_cost, count, nstate, \
                copy.deepcopy(cur_state[3]) ) );

    print(" ".join([prediction_set.index_to_word[i] for i in cur_state[3]]));
    break;
    # print("End of Iteration!");

print("Average Loss: ", val_loss_epoch/val_set);