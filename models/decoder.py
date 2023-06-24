# -*- coding: utf-8 -*-

import math
from functools import partial
import pdb
import torch
import torch.nn as nn

from captioning.models.utils import generate_length_mask, init


class BaseDecoder(nn.Module):
    """
    Take word/audio embeddings and output the next word probs
    Base decoder, cannot be called directly
    All decoders should inherit from this class
    """

    def __init__(self, emb_dim, vocab_size, fc_emb_dim, attn_emb_dim, dropout=0.2):
        super().__init__()
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.fc_emb_dim = fc_emb_dim
        self.attn_emb_dim = attn_emb_dim
        self.word_embedding = nn.Embedding(vocab_size, emb_dim)
        self.in_dropout = nn.Dropout(dropout)

    def forward(self, x):
        raise NotImplementedError


class RnnDecoder(BaseDecoder):

    def __init__(self, emb_dim, vocab_size, fc_emb_dim, attn_emb_dim, dropout, d_model, **kwargs):
        super().__init__(emb_dim, vocab_size, fc_emb_dim, attn_emb_dim, dropout)
        self.d_model = d_model
        self.num_layers = kwargs.get('num_layers', 1)
        self.bidirectional = kwargs.get('bidirectional', False)
        self.rnn_type = kwargs.get('rnn_type', "GRU")
        self.classifier = nn.Linear(
            self.d_model * (self.bidirectional + 1), vocab_size)

    def forward(self, x):
        raise NotImplementedError

    def load_word_embedding(self, weight, freeze=True):
        assert weight.shape[0] == self.vocab_size, "vocabulary size mismatch"
        assert weight.shape[1] == self.emb_dim, "embed size mismatch"
        
        # embeddings = torch.as_tensor(embeddings).float()
        # self.word_embeddings.weight = nn.Parameter(embeddings)
        # for para in self.word_embeddings.parameters():
            # para.requires_grad = tune
        self.word_embedding = nn.Embedding.from_pretrained(weight, freeze=freeze)


    def init_hidden(self, bs, device):
        num_dire = self.bidirectional + 1
        n_layer = self.num_layers
        hid_dim = self.d_model
        if self.rnn_type == "LSTM":
            return (torch.zeros(num_dire * n_layer, bs, hid_dim).to(device),
                    torch.zeros(num_dire * n_layer, bs, hid_dim).to(device))
        else:
            return torch.zeros(num_dire * n_layer, bs, hid_dim).to(device)


class RnnFcDecoder(RnnDecoder):

    def __init__(self, emb_dim, vocab_size, fc_emb_dim, attn_emb_dim, dropout, d_model, **kwargs):
        super().__init__(emb_dim, vocab_size, fc_emb_dim, attn_emb_dim, dropout, d_model, **kwargs)
        self.model = getattr(nn, self.rnn_type)(
            input_size=self.emb_dim * 2,
            hidden_size=self.d_model,
            batch_first=True,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional)
        self.fc_proj = nn.Linear(self.fc_emb_dim, self.emb_dim)
        self.apply(init)
    
    def forward(self, input_dict):
        """
        RNN-style decoder must implement `forward` like this:
            accept a word input and last time hidden state, return the word
            logits output and hidden state of this timestep
        the return dict must contain at least `logits` and `state`
        """
        word = input_dict["word"]
        state = input_dict.get("state", None)
        fc_embs = input_dict["fc_embs"]

        word = word.to(fc_embs.device)
        embed = self.in_dropout(self.word_embedding(word))
        
        p_fc_embs = self.fc_proj(fc_embs)
        # embed: [N, T, embed_size]
        embed = torch.cat((embed, p_fc_embs), dim=-1)

        out, state = self.model(embed, state)
        # out: [N, T, hs], states: [num_layers * num_dire, N, hs]
        logits = self.classifier(out)
        output = {
            "state": state,
            "embeds": out,
            "logits": logits
        }

        return output


class Seq2SeqAttention(nn.Module):

    def __init__(self, hs_enc, hs_dec, attn_size):
        """
        Args:
            hs_enc: encoder hidden size
            hs_dec: decoder hidden size
            attn_size: attention vector size
        """
        super(Seq2SeqAttention, self).__init__()
        self.h2attn = nn.Linear(hs_enc + hs_dec, attn_size)
        self.v = nn.Parameter(torch.randn(attn_size))
        self.apply(init)

    def forward(self, h_dec, h_enc, src_lens):
        """
        Args:
            h_dec: decoder hidden (query), [N, hs_dec]
            h_enc: encoder memory (key/value), [N, src_max_len, hs_enc]
            src_lens: source (encoder memory) lengths, [N, ]
        """
        N = h_enc.size(0)
        src_max_len = h_enc.size(1)
        h_dec = h_dec.unsqueeze(1).repeat(1, src_max_len, 1) # [N, src_max_len, hs_dec]

        attn_input = torch.cat((h_dec, h_enc), dim=-1)
        attn_out = torch.tanh(self.h2attn(attn_input)) # [N, src_max_len, attn_size]

        v = self.v.repeat(N, 1).unsqueeze(1) # [N, 1, attn_size]
        score = torch.bmm(v, attn_out.transpose(1, 2)).squeeze(1) # [N, src_max_len]

        idxs = torch.arange(src_max_len).repeat(N).view(N, src_max_len)
        mask = (idxs < src_lens.view(-1, 1)).to(h_dec.device)

        score = score.masked_fill(mask == 0, -1e10)
        weights = torch.softmax(score, dim=-1) # [N, src_max_len]
        ctx = torch.bmm(weights.unsqueeze(1), h_enc).squeeze(1) # [N, hs_enc]
        return ctx, weights

class PointNet(Seq2SeqAttention):

    def __init__(self, hs_enc, hs_dec, attn_size):
        """
        Args:
            hs_enc: encoder hidden size
            hs_dec: decoder hidden size
            attn_size: attention vector size
        """
        super(PointNet, self).__init__(hs_enc, hs_dec, attn_size)

class AttentionProj(nn.Module):

    def __init__(self, hs_enc, hs_dec, embed_dim, attn_size):
        self.q_proj = nn.Linear(hs_dec, embed_dim)
        self.kv_proj = nn.Linear(hs_enc, embed_dim)
        self.h2attn = nn.Linear(embed_dim * 2, attn_size)
        self.v = nn.Parameter(torch.randn(attn_size))
        self.apply(init)

    def init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, h_dec, h_enc, src_lens):
        """
        Args:
            h_dec: decoder hidden (query), [N, hs_dec]
            h_enc: encoder memory (key/value), [N, src_max_len, hs_enc]
            src_lens: source (encoder memory) lengths, [N, ]
        """
        h_enc = self.kv_proj(h_enc) # [N, src_max_len, embed_dim]
        h_dec = self.q_proj(h_dec) # [N, embed_dim]
        N = h_enc.size(0)
        src_max_len = h_enc.size(1)
        h_dec = h_dec.unsqueeze(1).repeat(1, src_max_len, 1) # [N, src_max_len, hs_dec]

        attn_input = torch.cat((h_dec, h_enc), dim=-1)
        attn_out = torch.tanh(self.h2attn(attn_input)) # [N, src_max_len, attn_size]

        v = self.v.repeat(N, 1).unsqueeze(1) # [N, 1, attn_size]
        score = torch.bmm(v, attn_out.transpose(1, 2)).squeeze(1) # [N, src_max_len]

        idxs = torch.arange(src_max_len).repeat(N).view(N, src_max_len)
        mask = (idxs < src_lens.view(-1, 1)).to(h_dec.device)

        score = score.masked_fill(mask == 0, -1e10)
        weights = torch.softmax(score, dim=-1) # [N, src_max_len]
        ctx = torch.bmm(weights.unsqueeze(1), h_enc).squeeze(1) # [N, hs_enc]

        return ctx, weights

class BahAttnDecoder(RnnDecoder):

    def __init__(self, emb_dim, vocab_size, fc_emb_dim, attn_emb_dim, dropout, d_model, **kwargs):
        """
        concatenate fc, attn, word to feed to the rnn
        """
        super().__init__(emb_dim, vocab_size, fc_emb_dim, attn_emb_dim, dropout, d_model, **kwargs)
        attn_size = kwargs.get("attn_size", self.d_model)
        self.model = getattr(nn, self.rnn_type)(
            input_size=self.emb_dim * 3,
            hidden_size=self.d_model,
            batch_first=True,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional)
        self.attn = Seq2SeqAttention(self.attn_emb_dim,
                                     self.d_model * (self.bidirectional + 1) * self.num_layers,
                                     attn_size)
        self.fc_proj = nn.Linear(self.fc_emb_dim, self.emb_dim)
        self.ctx_proj = nn.Linear(self.attn_emb_dim, self.emb_dim)
        self.apply(init)

    def forward(self, input_dict):
        word = input_dict["word"]
        state = input_dict.get("state", None) # [n_layer * n_dire, bs, d_model]
        fc_embs = input_dict["fc_embs"]
        attn_embs = input_dict["attn_embs"]
        attn_emb_lens = input_dict["attn_emb_lens"]

        word = word.to(fc_embs.device)
        embed = self.in_dropout(self.word_embedding(word))

        # embed: [N, 1, embed_size]
        if state is None:
            state = self.init_hidden(word.size(0), fc_embs.device)
        if self.rnn_type == "LSTM":
            query = state[0].transpose(0, 1).flatten(1)
        else:
            query = state.transpose(0, 1).flatten(1)
        c, attn_weight = self.attn(query, attn_embs, attn_emb_lens)

        p_fc_embs = self.fc_proj(fc_embs)
        p_ctx = self.ctx_proj(c)
        rnn_input = torch.cat((embed, p_ctx.unsqueeze(1), p_fc_embs.unsqueeze(1)), dim=-1)

        out, state = self.model(rnn_input, state)

        output = {
            "state": state,
            "embeds": out,
            "logits": self.classifier(out),
            "weights": attn_weight
        }
        return output


class BahAttnDecoder2(RnnDecoder):

    def __init__(self, emb_dim, vocab_size, fc_emb_dim, attn_emb_dim, dropout, d_model, **kwargs):
        """
        add fc, attn, word together to feed to the rnn
        """
        super().__init__(emb_dim, vocab_size, fc_emb_dim, attn_emb_dim, dropout, d_model, **kwargs)
        attn_size = kwargs.get("attn_size", self.d_model)
        self.model = getattr(nn, self.rnn_type)(
            input_size=self.emb_dim,
            hidden_size=self.d_model,
            batch_first=True,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional)
        self.attn = Seq2SeqAttention(self.emb_dim,
                                     self.d_model * (self.bidirectional + 1) * self.num_layers,
                                     attn_size)
        self.fc_proj = nn.Linear(self.fc_emb_dim, self.emb_dim)
        self.attn_proj = nn.Linear(self.attn_emb_dim, self.emb_dim)
        self.apply(partial(init, method="xavier"))

    def forward(self, input_dict):
        word = input_dict["word"]
        state = input_dict.get("state", None) # [n_layer * n_dire, bs, d_model]
        fc_embs = input_dict["fc_embs"]
        attn_embs = input_dict["attn_embs"]
        attn_emb_lens = input_dict["attn_emb_lens"]

        word = word.to(fc_embs.device)
        embed = self.in_dropout(self.word_embedding(word))
        p_attn_embs = self.attn_proj(attn_embs)

        # embed: [N, 1, embed_size]
        if state is None:
            state = self.init_hidden(word.size(0), fc_embs.device)
        if self.rnn_type == "LSTM":
            query = state[0].transpose(0, 1).flatten(1)
        else:
            query = state.transpose(0, 1).flatten(1)
        c, attn_weight = self.attn(query, p_attn_embs, attn_emb_lens)

        p_fc_embs = self.fc_proj(fc_embs)
        rnn_input = embed + c.unsqueeze(1) + p_fc_embs.unsqueeze(1)

        out, state = self.model(rnn_input, state)

        output = {
            "state": state,
            "embeds": out,
            "logits": self.classifier(out),
            "weights": attn_weight
        }
        return output


class BahAttnDecoder3(RnnDecoder):

    def __init__(self, emb_dim, vocab_size, fc_emb_dim, attn_emb_dim, dropout, d_model, **kwargs):
        """
        concatenate fc, attn, word to feed to the rnn
        """
        super().__init__(emb_dim, vocab_size, fc_emb_dim, attn_emb_dim, dropout, d_model, **kwargs)
        attn_size = kwargs.get("attn_size", self.d_model)
        self.start_by_fc = kwargs.get("start_by_fc", False)
        if self.start_by_fc:
            assert self.emb_dim == self.fc_emb_dim
        self.model = getattr(nn, self.rnn_type)(
            input_size=self.emb_dim + attn_emb_dim,
            hidden_size=self.d_model,
            batch_first=True,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional)
        self.attn = Seq2SeqAttention(self.attn_emb_dim,
                                     self.d_model * (self.bidirectional + 1) * self.num_layers,
                                     attn_size)
        self.ctx_proj = lambda x: x
        self.apply(init)

    def forward(self, input_dict):
        word = input_dict["word"]
        state = input_dict.get("state", None) # [n_layer * n_dire, bs, d_model]
        fc_embs = input_dict["fc_embs"]
        attn_embs = input_dict["attn_embs"]
        attn_emb_lens = input_dict["attn_emb_lens"]

        if word.size(-1) == self.fc_emb_dim: # fc_embs
            embed = word.unsqueeze(1)
        elif word.size(-1) == 1: # word
            word = word.to(fc_embs.device)
            embed = self.in_dropout(self.word_embedding(word))
        else:
            raise Exception(f"problem with word input size {word.size()}")

        # embed: [N, 1, embed_size]
        if state is None:
            state = self.init_hidden(word.size(0), fc_embs.device)
        if self.rnn_type == "LSTM":
            query = state[0].transpose(0, 1).flatten(1)
        else:
            query = state.transpose(0, 1).flatten(1)
        c, attn_weight = self.attn(query, attn_embs, attn_emb_lens)

        p_ctx = self.ctx_proj(c)
        rnn_input = torch.cat((embed, p_ctx.unsqueeze(1)), dim=-1)

        out, state = self.model(rnn_input, state)

        output = {
            "state": state,
            "embeds": out,
            "logits": self.classifier(out),
            "weights": attn_weight
        }
        return output

class BahAttnSkeletonDecoder(RnnDecoder):

    def __init__(self, emb_dim, vocab_size, prefix_vocab_size, suffix_vocab_size,
            fc_emb_dim, attn_emb_dim, dropout, d_model, **kwargs):
        """
        concatenate fc, attn, word to feed to the rnn
        """
        super().__init__(emb_dim, vocab_size, fc_emb_dim, attn_emb_dim, dropout, d_model, **kwargs)
        attn_size = kwargs.get("attn_size", self.d_model)
        self.model = getattr(nn, self.rnn_type)(
            input_size=self.emb_dim * 3,
            hidden_size=self.d_model,
            batch_first=True,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional)
        self.attn = Seq2SeqAttention(self.attn_emb_dim,
                                     self.d_model * (self.bidirectional + 1) * self.num_layers,
                                     attn_size)
        self.fc_proj = nn.Linear(self.fc_emb_dim, self.emb_dim)
        self.ctx_proj = nn.Linear(self.attn_emb_dim, self.emb_dim)
        self.apply(init)
        self.prefix_classifier = nn.Linear(
            self.d_model * (self.bidirectional + 1), prefix_vocab_size)
        self.suffix_classifier = nn.Linear(
            self.d_model * (self.bidirectional + 1), suffix_vocab_size)

    def forward(self, input_dict):
        word = input_dict["word"]
        state = input_dict.get("state", None) # [n_layer * n_dire, bs, d_model]
        fc_embs = input_dict["fc_embs"]
        attn_embs = input_dict["attn_embs"]
        attn_emb_lens = input_dict["attn_emb_lens"]
        if word.size(-1) == self.fc_emb_dim: # fc_embs
            embed = word.unsqueeze(1)
        elif word.size(-1) == 1: # word
            word = word.to(fc_embs.device)
            embed = self.in_dropout(self.word_embedding(word))
        else:
            raise Exception(f"problem with word input size {word.size()}")

        # embed: [N, 1, embed_size]
        if state is None:
            state = self.init_hidden(word.size(0), fc_embs.device)
        if self.rnn_type == "LSTM":
            query = state[0].transpose(0, 1).flatten(1)
        else:
            query = state.transpose(0, 1).flatten(1)
        c, attn_weight = self.attn(query, attn_embs, attn_emb_lens)

        p_ctx = self.ctx_proj(c)
        p_fc_embs = self.fc_proj(fc_embs)
        p_ctx = self.ctx_proj(c)
        rnn_input = torch.cat((embed, p_ctx.unsqueeze(1), p_fc_embs.unsqueeze(1)), dim=-1)

        out, state = self.model(rnn_input, state)

        output = {
            "state": state,
            "embeds": out,
            "logits": self.classifier(out),
            "weights": attn_weight
        }
        return output
class BahAttnTemporalDecoder(RnnDecoder):

    def __init__(self, emb_dim, vocab_size, fc_emb_dim, attn_emb_dim, dropout, d_model, **kwargs):
        """
        concatenate fc, attn, word to feed to the rnn
        """
        super().__init__(emb_dim, vocab_size, fc_emb_dim, attn_emb_dim, dropout, d_model, **kwargs)
        attn_size = kwargs.get("attn_size", self.d_model)
        self.model = getattr(nn, self.rnn_type)(
            input_size=self.emb_dim * 3,
            hidden_size=self.d_model,
            batch_first=True,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional)
        self.attn = Seq2SeqAttention(self.attn_emb_dim,
                                     self.d_model * (self.bidirectional + 1) * self.num_layers,
                                     attn_size)
        self.fc_proj = nn.Linear(self.fc_emb_dim, self.emb_dim)
        self.ctx_proj = nn.Linear(self.attn_emb_dim, self.emb_dim)
        self.temporal_embedding = nn.Embedding(4, emb_dim)
        self.apply(init)

    def forward(self, input_dict):
        word = input_dict["word"]
        state = input_dict.get("state", None) # [n_layer * n_dire, bs, d_model]
        fc_embs = input_dict["fc_embs"]
        attn_embs = input_dict["attn_embs"]
        attn_emb_lens = input_dict["attn_emb_lens"]
        temporal_label = input_dict["temporal_label"]
        if word.size(-1) == self.fc_emb_dim: # fc_embs
            embed = word.unsqueeze(1)
        elif word.size(-1) == 1: # word
            word = word.to(fc_embs.device)
            embed = self.in_dropout(self.word_embedding(word))
        else:
            raise Exception(f"problem with word input size {word.size()}")
        if input_dict["t"] == 0:
            embed = self.in_dropout(self.temporal_embedding(temporal_label)).unsqueeze(1)

        # embed: [N, 1, embed_size]
        if state is None:
            state = self.init_hidden(word.size(0), fc_embs.device)
        if self.rnn_type == "LSTM":
            query = state[0].transpose(0, 1).flatten(1)
        else:
            query = state.transpose(0, 1).flatten(1)
        c, attn_weight = self.attn(query, attn_embs, attn_emb_lens)

        p_ctx = self.ctx_proj(c)
        p_fc_embs = self.fc_proj(fc_embs)
        p_ctx = self.ctx_proj(c)
        rnn_input = torch.cat((embed, p_ctx.unsqueeze(1), p_fc_embs.unsqueeze(1)), dim=-1)

        out, state = self.model(rnn_input, state)

        output = {
            "state": state,
            "embeds": out,
            "logits": self.classifier(out),
            "weights": attn_weight
        }
        return output

class BahAttnSedprobDecoder(RnnDecoder):

    def __init__(self, emb_dim, vocab_size, fc_emb_dim, attn_emb_dim, dropout, d_model, **kwargs):
        """
        concatenate fc, attn, word to feed to the rnn
        """
        super().__init__(emb_dim, vocab_size, fc_emb_dim, attn_emb_dim, dropout, d_model, **kwargs)
        attn_size = kwargs.get("attn_size", self.d_model)
        self.model = getattr(nn, self.rnn_type)(
            input_size=self.emb_dim * 3,
            hidden_size=self.d_model,
            batch_first=True,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional)
        self.attn = Seq2SeqAttention(self.attn_emb_dim + 447,
                                     self.d_model * (self.bidirectional + 1) * self.num_layers,
                                     attn_size)
        self.fc_proj = nn.Linear(self.fc_emb_dim, self.emb_dim)
        self.ctx_proj = nn.Linear(self.attn_emb_dim + 447, self.emb_dim)
        #self.sed_prob_proj = nn.Linear(447, self.emb_dim)
        self.apply(init)
    
    def forward(self, input_dict):
        word = input_dict["word"]
        state = input_dict.get("state", None) # [n_layer * n_dire, bs, d_model]
        fc_embs = input_dict["fc_embs"]
        attn_embs = input_dict["attn_embs"]
        attn_emb_lens = input_dict["attn_emb_lens"]
        word = word.to(fc_embs.device)
        embed = self.in_dropout(self.word_embedding(word))
        attn_embs = torch.cat([attn_embs, input_dict["sed_prob"]], -1)
        
        # embed: [N, 1, embed_size]
        if state is None:
            state = self.init_hidden(word.size(0), fc_embs.device)
        if self.rnn_type == "LSTM":
            query = state[0].transpose(0, 1).flatten(1)
        else:
            query = state.transpose(0, 1).flatten(1)
        c, attn_weight = self.attn(query, attn_embs, attn_emb_lens)

        p_fc_embs = self.fc_proj(fc_embs)
        p_ctx = self.ctx_proj(c)

        rnn_input = torch.cat((embed, p_ctx.unsqueeze(1), p_fc_embs.unsqueeze(1)), dim=-1)

        out, state = self.model(rnn_input, state)
        
        output = {
            "state": state,
            "embeds": out,
            "logits": self.classifier(out),
            "weights": attn_weight
        }
        return output 
class BahAttnSedprobAttnDecoder(RnnDecoder):

    def __init__(self, emb_dim, vocab_size, fc_emb_dim, attn_emb_dim, dropout, d_model, **kwargs):
        """
        concatenate fc, attn, word to feed to the rnn
        """
        super().__init__(emb_dim, vocab_size, fc_emb_dim, attn_emb_dim, dropout, d_model, **kwargs)
        attn_size = kwargs.get("attn_size", self.d_model)
        self.model = getattr(nn, self.rnn_type)(
            input_size=self.emb_dim * 3,
            hidden_size=self.d_model,
            batch_first=True,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional)
        self.attn = Seq2SeqAttention(self.attn_emb_dim,
                                     self.d_model * (self.bidirectional + 1) * self.num_layers,
                                     attn_size)
        self.sed_prob_attn = Seq2SeqAttention(447, 
                                     self.d_model * (self.bidirectional + 1) * self.num_layers,
                                     attn_size)
        self.fc_proj = nn.Linear(self.fc_emb_dim, self.emb_dim)
        self.ctx_proj = nn.Linear(self.attn_emb_dim, self.emb_dim)
        self.sed_prob_proj = nn.Linear(447, self.emb_dim)
        self.apply(init)
    
    def forward(self, input_dict):
        word = input_dict["word"]
        state = input_dict.get("state", None) # [n_layer * n_dire, bs, d_model]
        fc_embs = input_dict["fc_embs"]
        attn_embs = input_dict["attn_embs"]
        attn_emb_lens = input_dict["attn_emb_lens"]
        word = word.to(fc_embs.device)
        embed = self.in_dropout(self.word_embedding(word))
        sed_prob = input_dict["sed_prob"]

        # embed: [N, 1, embed_size]
        if state is None:
            state = self.init_hidden(word.size(0), fc_embs.device)
        if self.rnn_type == "LSTM":
            query = state[0].transpose(0, 1).flatten(1)
        else:
            query = state.transpose(0, 1).flatten(1)
        attn_c, attn_weight = self.attn(query, attn_embs, attn_emb_lens)

        p_fc_embs = self.fc_proj(fc_embs)
        p_ctx = self.ctx_proj(attn_c)

        sed_prob_c, _ = self.sed_prob_attn(p_ctx, sed_prob, attn_emb_lens)
        sed_prob_ctx = self.sed_prob_proj(sed_prob_c)
        
        rnn_input = torch.cat((embed, sed_prob_ctx.unsqueeze(1), p_fc_embs.unsqueeze(1)), dim=-1)

        out, state = self.model(rnn_input, state)

        output = {
            "state": state,
            "embeds": out,
            "logits": self.classifier(out),
            "weights": attn_weight
        }
        return output
class BahAttnAddinfoDecoder1(RnnDecoder):

    def __init__(self, emb_dim, vocab_size, add_vocab_size, fc_emb_dim, attn_emb_dim, \
            dropout, d_model, **kwargs):
        """
        concatenate attn, word, to feed to the rnn
        """
        super().__init__(emb_dim, vocab_size, fc_emb_dim, attn_emb_dim, dropout, d_model, **kwargs)
        attn_size = kwargs.get("attn_size", self.d_model)
        self.start_by_fc = kwargs.get("start_by_fc", False)
        if self.start_by_fc:
            assert self.emb_dim == self.fc_emb_dim
        self.model = getattr(nn, self.rnn_type)(
            input_size=self.emb_dim + attn_emb_dim,
            hidden_size=self.d_model,
            batch_first=True,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional)
        self.attn = Seq2SeqAttention(self.attn_emb_dim,
                                     self.d_model * (self.bidirectional + 1) * self.num_layers,
                                     attn_size)
        self.ctx_proj = lambda x: x
        self.addinfo_embedding = nn.Linear(add_vocab_size, attn_emb_dim)
        self.apply(init)

    def forward(self, input_dict):
        word = input_dict["word"]
        state = input_dict.get("state", None) # [n_layer * n_dire, bs, d_model]
        fc_embs = input_dict["fc_embs"]
        attn_embs = input_dict["attn_embs"]
        attn_emb_lens = input_dict["attn_emb_lens"]
        add_info = input_dict["add_info"]

        if word.size(-1) == self.fc_emb_dim: # fc_embs
            embed = word.unsqueeze(1)
        elif word.size(-1) == 1: # word
            word = word.to(fc_embs.device)
            embed = self.in_dropout(self.word_embedding(word))
        else:
            raise Exception(f"problem with word input size {word.size()}")

        add_info_embs = self.addinfo_embedding(add_info)
        attn_embs = torch.cat((add_info_embs.unsqueeze(1), attn_embs), dim=1)
        # embed: [N, 1, embed_size]
        if state is None:
            state = self.init_hidden(word.size(0), fc_embs.device)
        if self.rnn_type == "LSTM":
            query = state[0].transpose(0, 1).flatten(1)
        else:
            query = state.transpose(0, 1).flatten(1)

        c, attn_weight = self.attn(query, attn_embs, attn_emb_lens)

        p_ctx = self.ctx_proj(c)
        
        rnn_input = torch.cat((embed, p_ctx.unsqueeze(1)), dim=-1)

        out, state = self.model(rnn_input, state)

        output = {
            "state": state,
            "embeds": out,
            "logits": self.classifier(out),
            "weights": attn_weight
        }
        return output
class BahAttnAddinfoCatonehotDecoder(RnnDecoder):

    def __init__(self, emb_dim, vocab_size, add_vocab_size, fc_emb_dim, attn_emb_dim, \
            dropout, d_model, **kwargs):
        """
        concatenate attn, word, to feed to the rnn
        """
        super().__init__(emb_dim, vocab_size, fc_emb_dim, attn_emb_dim, dropout, d_model, **kwargs)
        attn_size = kwargs.get("attn_size", self.d_model)
        self.start_by_fc = kwargs.get("start_by_fc", False)
        if self.start_by_fc:
            assert self.emb_dim == self.fc_emb_dim
        self.model = getattr(nn, self.rnn_type)(
            input_size=self.emb_dim + attn_emb_dim,
            hidden_size=self.d_model,
            batch_first=True,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional)
        self.attn = Seq2SeqAttention(self.attn_emb_dim,
                                     self.d_model * (self.bidirectional + 1) * self.num_layers,
                                     attn_size)
        self.ctx_proj = lambda x: x
        self.add_info_embedding = nn.Linear(add_vocab_size, attn_emb_dim)
        self.apply(init)

    def forward(self, input_dict):
        word = input_dict["word"]
        state = input_dict.get("state", None) # [n_layer * n_dire, bs, d_model]
        fc_embs = input_dict["fc_embs"]
        attn_embs = input_dict["attn_embs"]
        ori_attn_emb_lens = input_dict["ori_attn_emb_lens"]
        attn_emb_lens = input_dict["attn_emb_lens"]
        add_info_lens = input_dict["add_info_lens"]
        add_info = input_dict["add_info"]

        if word.size(-1) == self.fc_emb_dim: # fc_embs
            embed = word.unsqueeze(1)
        elif word.size(-1) == 1: # word
            word = word.to(fc_embs.device)
            embed = self.in_dropout(self.word_embedding(word))
        else:
            raise Exception(f"problem with word input size {word.size()}")

        N = attn_embs.shape[0]
        cated_embs = torch.zeros(N, add_info.size(1) + attn_embs.size(1), 
                attn_embs.shape[-1]).to(fc_embs.device)
        add_emb = self.add_info_embedding(add_info)

        for idx in range(N):
            emb_len = ori_attn_emb_lens[idx]
            add_len = add_info_lens[idx]
            cated_embs[idx, :emb_len, :] = attn_embs[idx, :emb_len, :]
            cated_embs[idx, emb_len: (emb_len + add_len), :] = add_emb[idx, :add_len, :]
        # embed: [N, 1, embed_size]
        if state is None:
            state = self.init_hidden(word.size(0), fc_embs.device)
        if self.rnn_type == "LSTM":
            query = state[0].transpose(0, 1).flatten(1)
        else:
            query = state.transpose(0, 1).flatten(1)
        c, attn_weight = self.attn(query, cated_embs, attn_emb_lens)
        p_ctx = self.ctx_proj(c)
        
        rnn_input = torch.cat((embed, p_ctx.unsqueeze(1)), dim=-1)

        out, state = self.model(rnn_input, state)

        output = {
            "state": state,
            "embeds": out,
            "logits": self.classifier(out),
            "weights": attn_weight
        }
        return output

class BahAttnAddinfoCatattnDecoder(RnnDecoder):

    def __init__(self, emb_dim, vocab_size, add_vocab_size, fc_emb_dim, attn_emb_dim, \
            dropout, d_model, **kwargs):
        """
        concatenate attn, word, to feed to the rnn
        """
        super().__init__(emb_dim, vocab_size, fc_emb_dim, attn_emb_dim, dropout, d_model, **kwargs)
        attn_size = kwargs.get("attn_size", self.d_model)
        self.start_by_fc = kwargs.get("start_by_fc", False)
        if self.start_by_fc:
            assert self.emb_dim == self.fc_emb_dim
        self.model = getattr(nn, self.rnn_type)(
            input_size=self.emb_dim + attn_emb_dim * 2,
            hidden_size=self.d_model,
            batch_first=True,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional)
        self.attn = Seq2SeqAttention(self.attn_emb_dim,
                                     self.d_model * (self.bidirectional + 1) * self.num_layers,
                                     attn_size)
        self.add_info_attn = Seq2SeqAttention(self.attn_emb_dim,
                                     self.d_model * (self.bidirectional + 1) * self.num_layers,
                                     attn_size)
        self.ctx_proj = lambda x: x
        self.addinfo_embedding = nn.Embedding(add_vocab_size, attn_emb_dim)
        self.apply(init)

    def forward(self, input_dict):
        word = input_dict["word"]
        state = input_dict.get("state", None) # [n_layer * n_dire, bs, d_model]
        fc_embs = input_dict["fc_embs"]
        attn_embs = input_dict["attn_embs"]
        attn_emb_lens = input_dict["attn_emb_lens"]
        add_info = input_dict["add_info"]
        add_info_lens = input_dict["add_info_lens"]

        if word.size(-1) == self.fc_emb_dim: # fc_embs
            embed = word.unsqueeze(1)
        elif word.size(-1) == 1: # word
            word = word.to(fc_embs.device)
            embed = self.in_dropout(self.word_embedding(word))
        else:
            raise Exception(f"problem with word input size {word.size()}")

        # embed: [N, 1, embed_size]
        if state is None:
            state = self.init_hidden(word.size(0), fc_embs.device)
        if self.rnn_type == "LSTM":
            query = state[0].transpose(0, 1).flatten(1)
        else:
            query = state.transpose(0, 1).flatten(1)
        c, attn_weight = self.attn(query, attn_embs, attn_emb_lens)

        p_ctx = self.ctx_proj(c)
        
        add_info_embs = self.addinfo_embedding(add_info)
        add_info_c, add_info_weight = self.add_info_attn(query, add_info_embs.squeeze(-2), add_info_lens)
        #add_info_ctx = self.ctx_proj(add_info_c)
        rnn_input = torch.cat((embed, p_ctx.unsqueeze(1), add_info_c.unsqueeze(1)), dim=-1)

        out, state = self.model(rnn_input, state)

        output = {
            "state": state,
            "embeds": out,
            "logits": self.classifier(out),
            "weights": attn_weight
        }
        return output

class BahAttnAddinfoCopyInstDecoder(RnnDecoder):

    def __init__(self, emb_dim, vocab_size, add_vocab_size, add_info_map, fc_emb_dim, attn_emb_dim, \
            dropout, d_model, **kwargs):
        super().__init__(emb_dim, vocab_size, fc_emb_dim, attn_emb_dim, dropout, d_model, **kwargs)
        self.add_info_map = add_info_map,
        attn_size = kwargs.get("attn_size", self.d_model)
        self.start_by_fc = kwargs.get("start_by_fc", False)
        if self.start_by_fc:
            assert self.emb_dim == self.fc_emb_dim
        self.model = getattr(nn, self.rnn_type)(
            input_size=self.emb_dim + attn_emb_dim,
            hidden_size=self.d_model,
            batch_first=True,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional)
        self.attn = Seq2SeqAttention(self.attn_emb_dim,
                                     self.d_model * (self.bidirectional + 1) * self.num_layers,
                                     attn_size)
        self.point = PointNet(self.attn_emb_dim,
                                     self.d_model * (self.bidirectional + 1) * self.num_layers,
                                     attn_size)
        self.ctx_proj = lambda x: x
        self.addinfo_embedding = nn.Embedding(add_vocab_size, attn_emb_dim)
        self.apply(init)
        self.p_gen_i = nn.Linear(self.emb_dim + attn_emb_dim, 1)
        self.p_gen_s = nn.Linear(self.d_model, 1)

    def forward(self, input_dict):
        word = input_dict["word"]
        state = input_dict.get("state", None) # [n_layer * n_dire, bs, d_model]
        fc_embs = input_dict["fc_embs"]
        attn_embs = input_dict["attn_embs"]
        attn_emb_lens = input_dict["attn_emb_lens"]
        add_info = input_dict["add_info"]
        add_info_lens = input_dict["add_info_lens"]

        if word.size(-1) == self.fc_emb_dim: # fc_embs
            embed = word.unsqueeze(1)
        elif word.size(-1) == 1: # word
            word = word.to(fc_embs.device)
            embed  = self.in_dropout(self.word_embedding(word))
        else:
            raise Exception(f"problem with word input size {word.size()}")

        # embed: [N, 1, embed_size]
        if state is None:
            state = self.init_hidden(word.size(0), fc_embs.device)
        if self.rnn_type == "LSTM":
            query = state[0].transpose(0, 1).flatten(1)
        else:
            query = state.transpose(0, 1).flatten(1)
        c, attn_weight = self.attn(query, attn_embs, attn_emb_lens)
        p_ctx = self.ctx_proj(c)
        
        add_info_embs = self.addinfo_embedding(add_info)
        _, point_weight = self.point(query, add_info_embs.squeeze(2), add_info_lens)
        rnn_input = torch.cat((embed, p_ctx.unsqueeze(1)), dim=-1)
        out, state = self.model(rnn_input, state)
        logits = self.classifier(out)
        
        logits_copy = torch.zeros(logits.shape).to(logits.device)
        add_info_map = self.add_info_map[0].to(logits.device)
        for i in range(logits_copy.shape[0]):
            index = torch.gather(add_info_map, 0, add_info.squeeze(-1)[i])                                                                             
            logits_copy[i][0].scatter_(0, index, point_weight[i]) 
            
        p_gen_s = self.p_gen_s(state)
        p_gen_i = self.p_gen_i(rnn_input)
        p_gen = torch.sigmoid(p_gen_s.squeeze(0) + p_gen_i.squeeze(1))

        logits = (1. - p_gen) * logits + p_gen * logits_copy
        
        output = {
            "state": state,
            "embeds": out,
            "logits": logits,
            "weights": attn_weight,
            "p_gen": p_gen.squeeze(-1),
            "logits_copy": logits_copy,
            "p_gen_s": p_gen_s,
            "p_gen_i": p_gen_i,
        }
        return output


class BahAttnAddinfoCopyBatchDecoder(RnnDecoder):

    def __init__(self, emb_dim, vocab_size, add_vocab_size, add_info_map, fc_emb_dim, attn_emb_dim, \
            dropout, d_model, **kwargs):
        super().__init__(emb_dim, vocab_size, fc_emb_dim, attn_emb_dim, dropout, d_model, **kwargs)
        self.add_info_map = add_info_map,
        attn_size = kwargs.get("attn_size", self.d_model)
        self.start_by_fc = kwargs.get("start_by_fc", False)
        if self.start_by_fc:
            assert self.emb_dim == self.fc_emb_dim
        self.model = getattr(nn, self.rnn_type)(
            input_size=self.emb_dim + attn_emb_dim,
            hidden_size=self.d_model,
            batch_first=True,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional)
        self.attn = Seq2SeqAttention(self.attn_emb_dim,
                                     self.d_model * (self.bidirectional + 1) * self.num_layers,
                                     attn_size)
        self.point = PointNet(self.attn_emb_dim,
                                     self.d_model * (self.bidirectional + 1) * self.num_layers,
                                     attn_size)
        self.ctx_proj = lambda x: x
        self.addinfo_embedding = nn.Embedding(add_vocab_size, attn_emb_dim)
        self.apply(init)
        self.p_gen_i = nn.Linear(self.emb_dim + attn_emb_dim, 1)
        self.p_gen_s = nn.Linear(self.d_model, 1)
    def forward(self, input_dict):
        word = input_dict["word"]
        state = input_dict.get("state", None) # [n_layer * n_dire, bs, d_model]
        fc_embs = input_dict["fc_embs"]
        attn_embs = input_dict["attn_embs"]
        attn_emb_lens = input_dict["attn_emb_lens"]
        add_info = input_dict["add_info"]
        add_info_lens = input_dict["add_info_lens"]

        if word.size(-1) == self.fc_emb_dim: # fc_embs
            embed = word.unsqueeze(1)
        elif word.size(-1) == 1: # word
            word = word.to(fc_embs.device)
            embed  = self.in_dropout(self.word_embedding(word))
        else:
            raise Exception(f"problem with word input size {word.size()}")

        # embed: [N, 1, embed_size]
        if state is None:
            state = self.init_hidden(word.size(0), fc_embs.device)
        if self.rnn_type == "LSTM":
            query = state[0].transpose(0, 1).flatten(1)
        else:
            query = state.transpose(0, 1).flatten(1)
        c, attn_weight = self.attn(query, attn_embs, attn_emb_lens)
        p_ctx = self.ctx_proj(c)
        
        add_info_embs = self.addinfo_embedding(add_info)
        _, point_weight = self.point(query, add_info_embs.squeeze(2), add_info_lens)
        rnn_input = torch.cat((embed, p_ctx.unsqueeze(1)), dim=-1)
        out, state = self.model(rnn_input, state)
        logits = self.classifier(out)

        logits_copy = torch.zeros(logits.shape).to(logits.device).squeeze(1)
        add_info_map = self.add_info_map[0].to(add_info.device)
        index = torch.gather(add_info_map, 0, add_info.view(-1)).view(add_info.shape)
        logits_copy.scatter_(1, index.squeeze(-1), point_weight)

        p_gen_s = self.p_gen_s(state)
        p_gen_i = self.p_gen_i(rnn_input)
        p_gen = torch.sigmoid(p_gen_s.squeeze(0) + p_gen_i.squeeze(1))
        logits = (1. - p_gen) * logits + p_gen * logits_copy.unsqueeze(1)
        
        output = {
            "state": state,
            "embeds": out,
            "logits": logits,
            "weights": attn_weight,
            "logits_copy": logits_copy,
        }
        return output

class BahAttnAddinfoCopyBatchPlusDecoder(RnnDecoder):

    def __init__(self, emb_dim, vocab_size, add_vocab_size, add_info_map, fc_emb_dim, attn_emb_dim, \
            dropout, d_model, **kwargs):
        super().__init__(emb_dim, vocab_size, fc_emb_dim, attn_emb_dim, dropout, d_model, **kwargs)
        self.add_info_map = add_info_map,
        attn_size = kwargs.get("attn_size", self.d_model)
        self.start_by_fc = kwargs.get("start_by_fc", False)
        if self.start_by_fc:
            assert self.emb_dim == self.fc_emb_dim
        self.model = getattr(nn, self.rnn_type)(
            input_size=self.emb_dim + attn_emb_dim,
            hidden_size=self.d_model,
            batch_first=True,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional)
        self.attn = Seq2SeqAttention(self.attn_emb_dim,
                                     self.d_model * (self.bidirectional + 1) * self.num_layers,
                                     attn_size)
        self.point = PointNet(self.attn_emb_dim,
                                     self.d_model * (self.bidirectional + 1) * self.num_layers,
                                     attn_size)
        self.ctx_proj = lambda x: x
        self.addinfo_embedding = nn.Embedding(add_vocab_size, attn_emb_dim)
        self.apply(init)
        self.p_gen_i = nn.Linear(self.emb_dim + attn_emb_dim, 1)
        self.p_gen_s = nn.Linear(self.d_model, 1)

    def forward(self, input_dict):
        word = input_dict["word"]
        state = input_dict.get("state", None) # [n_layer * n_dire, bs, d_model]
        fc_embs = input_dict["fc_embs"]
        attn_embs = input_dict["attn_embs"]
        attn_emb_lens = input_dict["attn_emb_lens"]
        add_info = input_dict["add_info"]
        add_info_lens = input_dict["add_info_lens"]

        if word.size(-1) == self.fc_emb_dim: # fc_embs
            embed = word.unsqueeze(1)
        elif word.size(-1) == 1: # word
            word = word.to(fc_embs.device)
            embed  = self.in_dropout(self.word_embedding(word))
        else:
            raise Exception(f"problem with word input size {word.size()}")

        # embed: [N, 1, embed_size]
        if state is None:
            state = self.init_hidden(word.size(0), fc_embs.device)
        if self.rnn_type == "LSTM":
            query = state[0].transpose(0, 1).flatten(1)
        else:
            query = state.transpose(0, 1).flatten(1)
        c, attn_weight = self.attn(query, attn_embs, attn_emb_lens)
        p_ctx = self.ctx_proj(c)
        
        add_info_embs = self.addinfo_embedding(add_info)
        _, point_weight = self.point(query, add_info_embs.squeeze(2), add_info_lens)
        rnn_input = torch.cat((embed, p_ctx.unsqueeze(1)), dim=-1)
        out, state = self.model(rnn_input, state)
        logits = self.classifier(out)

        logits_copy = torch.zeros(logits.shape).to(logits.device).squeeze(1)
        add_info_map = self.add_info_map[0].to(add_info.device)
        index = torch.gather(add_info_map, 0, add_info.view(-1)).view(add_info.shape)
        logits_copy.scatter_(1, index.squeeze(-1), point_weight)

        #p_gen_s = self.p_gen_s(state)
        #p_gen_i = self.p_gen_i(rnn_input)
        #p_gen = torch.sigmoid(p_gen_s.squeeze(0) + p_gen_i.squeeze(1))
        logits = logits + logits_copy.unsqueeze(1)
        
        output = {
            "state": state,
            "embeds": out,
            "logits": logits,
            "weights": attn_weight,
            "logits_copy": logits_copy,
        }
        return output

class BahAttnAddinfoCopyDecoder(RnnDecoder):

    def __init__(self, emb_dim, vocab_size, add_vocab_size, add_info_map, fc_emb_dim, attn_emb_dim, \
            dropout, d_model, **kwargs):
        super().__init__(emb_dim, vocab_size, fc_emb_dim, attn_emb_dim, dropout, d_model, **kwargs)
        self.add_info_map = add_info_map,
        attn_size = kwargs.get("attn_size", self.d_model)
        self.start_by_fc = kwargs.get("start_by_fc", False)
        if self.start_by_fc:
            assert self.emb_dim == self.fc_emb_dim
        self.model = getattr(nn, self.rnn_type)(
            input_size=self.emb_dim + attn_emb_dim,
            hidden_size=self.d_model,
            batch_first=True,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional)
        self.attn = Seq2SeqAttention(self.attn_emb_dim,
                                     self.d_model * (self.bidirectional + 1) * self.num_layers,
                                     attn_size)
        self.point = PointNet(self.attn_emb_dim,
                                     self.d_model * (self.bidirectional + 1) * self.num_layers,
                                     attn_size)
        self.ctx_proj = lambda x: x
        self.addinfo_embedding = nn.Embedding(add_vocab_size, attn_emb_dim)
        self.apply(init)
        self.p_gen_i = nn.Linear(self.emb_dim + attn_emb_dim, 1)
        self.p_gen_s = nn.Linear(self.d_model, 1)

    def forward(self, input_dict):
        word = input_dict["word"]
        state = input_dict.get("state", None) # [n_layer * n_dire, bs, d_model]
        fc_embs = input_dict["fc_embs"]
        attn_embs = input_dict["attn_embs"]
        attn_emb_lens = input_dict["attn_emb_lens"]
        add_info = input_dict["add_info"]
        add_info_lens = input_dict["add_info_lens"]

        if word.size(-1) == self.fc_emb_dim: # fc_embs
            embed = word.unsqueeze(1)
        elif word.size(-1) == 1: # word
            word = word.to(fc_embs.device)
            embed  = self.in_dropout(self.word_embedding(word))
        else:
            raise Exception(f"problem with word input size {word.size()}")

        # embed: [N, 1, embed_size]
        if state is None:
            state = self.init_hidden(word.size(0), fc_embs.device)
        if self.rnn_type == "LSTM":
            query = state[0].transpose(0, 1).flatten(1)
        else:
            query = state.transpose(0, 1).flatten(1)
        c, attn_weight = self.attn(query, attn_embs, attn_emb_lens)
        p_ctx = self.ctx_proj(c)
        
        add_info_embs = self.addinfo_embedding(add_info)
        _, point_weight = self.point(query, add_info_embs.squeeze(2), add_info_lens)
        rnn_input = torch.cat((embed, p_ctx.unsqueeze(1)), dim=-1)
        out, state = self.model(rnn_input, state)
        logits = self.classifier(out)
        
        logits_copy = torch.zeros(logits.shape).to(logits.device)
        add_info_map = self.add_info_map[0]
        add_info_map = add_info_map.squeeze(-1).cpu().numpy()
        for i in range(logits_copy.shape[0]):
            for j in range(add_info_lens[i]):
                index = add_info_map[add_info[i][j]]
                #if len(index) != 0:
                index = torch.as_tensor(index).to(logits.device)
                    #gather(add_info_map, 0, add_info.squeeze(-1)[i])
                logits_copy[i][0].scatter_(0, index, point_weight[i][j])
            
        p_gen_s = self.p_gen_s(state)
        p_gen_i = self.p_gen_i(rnn_input)
        p_gen = torch.sigmoid(p_gen_s.squeeze(0) + p_gen_i.squeeze(1))

        logits = (1. - p_gen) * logits + p_gen * logits_copy
        
        output = {
            "state": state,
            "embeds": out,
            "logits": logits,
            "weights": attn_weight,
            "p_gen": p_gen.squeeze(-1),
            "logits_copy": logits_copy,
            "p_gen_s": p_gen_s,
            "p_gen_i": p_gen_i,

        }
        return output

class BahAttnAddinfoCopyBugDecoder(RnnDecoder):

    def __init__(self, emb_dim, vocab_size, add_vocab_size, add_info_map, fc_emb_dim, attn_emb_dim, \
            dropout, d_model, **kwargs):
        super().__init__(emb_dim, vocab_size, fc_emb_dim, attn_emb_dim, dropout, d_model, **kwargs)
        self.add_info_map = add_info_map,
        attn_size = kwargs.get("attn_size", self.d_model)
        self.start_by_fc = kwargs.get("start_by_fc", False)
        if self.start_by_fc:
            assert self.emb_dim == self.fc_emb_dim
        self.model = getattr(nn, self.rnn_type)(
            input_size=self.emb_dim + attn_emb_dim,
            hidden_size=self.d_model,
            batch_first=True,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional)
        self.attn = Seq2SeqAttention(self.attn_emb_dim,
                                     self.d_model * (self.bidirectional + 1) * self.num_layers,
                                     attn_size)
        self.point = PointNet(self.attn_emb_dim,
                                     self.d_model * (self.bidirectional + 1) * self.num_layers,
                                     attn_size)
        self.ctx_proj = lambda x: x
        self.addinfo_embedding = nn.Embedding(add_vocab_size, attn_emb_dim)
        self.apply(init)
        self.p_gen_i = nn.Linear(self.emb_dim + attn_emb_dim, 1)
        self.p_gen_s = nn.Linear(self.d_model, 1)

    def forward(self, input_dict):
        word = input_dict["word"]
        state = input_dict.get("state", None) # [n_layer * n_dire, bs, d_model]
        fc_embs = input_dict["fc_embs"]
        attn_embs = input_dict["attn_embs"]
        attn_emb_lens = input_dict["attn_emb_lens"]
        add_info = input_dict["add_info"]
        add_info_lens = input_dict["add_info_lens"]

        if word.size(-1) == self.fc_emb_dim: # fc_embs
            embed = word.unsqueeze(1)
        elif word.size(-1) == 1: # word
            word = word.to(fc_embs.device)
            embed  = self.in_dropout(self.word_embedding(word))
        else:
            raise Exception(f"problem with word input size {word.size()}")

        # embed: [N, 1, embed_size]
        if state is None:
            state = self.init_hidden(word.size(0), fc_embs.device)
        if self.rnn_type == "LSTM":
            query = state[0].transpose(0, 1).flatten(1)
        else:
            query = state.transpose(0, 1).flatten(1)
        c, attn_weight = self.attn(query, attn_embs, attn_emb_lens)
        p_ctx = self.ctx_proj(c)
        
        add_info_embs = self.addinfo_embedding(add_info)
        _, point_weight = self.point(query, add_info_embs.squeeze(2), add_info_lens)
        rnn_input = torch.cat((embed, p_ctx.unsqueeze(1)), dim=-1)
        out, state = self.model(rnn_input, state)
        logits = self.classifier(out)
       
        #batch
        logits_copy1 = torch.zeros(logits.shape).to(logits.device).squeeze(1)
        add_info_map = self.add_info_map[0].to(add_info.device)
        index = torch.gather(add_info_map, 0, add_info.view(-1)).view(add_info.shape)
        logits_copy1.scatter_(1, index.squeeze(-1), point_weight)

        #instance
        logits_copy2 = torch.zeros(logits.shape).to(logits.device)
        add_info_map = self.add_info_map[0].to(logits.device)
        for i in range(logits_copy2.shape[0]):
            index = torch.gather(add_info_map, 0, add_info.squeeze(-1)[i])                                                                             
            logits_copy2[i][0].scatter_(0, index, point_weight[i]) 

        logits_copy3 = torch.zeros(logits.shape).to(logits.device)
        add_info_map = self.add_info_map[0]
        add_info_map = add_info_map.squeeze(-1).cpu().numpy()
        for i in range(logits_copy3.shape[0]):
            for j in range(add_info_lens[i]):
                index = add_info_map[add_info[i][j]]
                #if len(index) != 0:
                index = torch.as_tensor(index).to(logits.device)
                    #gather(add_info_map, 0, add_info.squeeze(-1)[i])
                logits_copy3[i][0].scatter_(0, index, point_weight[i][j])
           
        import pdb; pdb.set_trace()
        p_gen_s = self.p_gen_s(state)
        p_gen_i = self.p_gen_i(rnn_input)
        p_gen = torch.sigmoid(p_gen_s.squeeze(0) + p_gen_i.squeeze(1))

        logits = (1. - p_gen) * logits + p_gen * logits_copy1
        
        output = {
            "state": state,
            "embeds": out,
            "logits": logits,
            "weights": attn_weight,
            "p_gen": p_gen.squeeze(-1),
        }
        return output

class BahAttnAddinfoCatattnControldimDecoder(RnnDecoder):

    def __init__(self, emb_dim, vocab_size, add_vocab_size, fc_emb_dim, attn_emb_dim, \
            dropout, d_model, **kwargs):
        """
        concatenate attn, word, to feed to the rnn
        """
        super().__init__(emb_dim, vocab_size, fc_emb_dim, attn_emb_dim, dropout, d_model, **kwargs)
        attn_size = kwargs.get("attn_size", self.d_model)
        self.start_by_fc = kwargs.get("start_by_fc", False)
        if self.start_by_fc:
            assert self.emb_dim == self.fc_emb_dim
        self.model = getattr(nn, self.rnn_type)(
            input_size=self.emb_dim * 3,
            hidden_size=self.d_model,
            batch_first=True,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional)
        self.attn = Seq2SeqAttention(self.attn_emb_dim,
                                     self.d_model * (self.bidirectional + 1) * self.num_layers,
                                     attn_size)
        self.add_info_attn = Seq2SeqAttention(self.attn_emb_dim,
                                     self.d_model * (self.bidirectional + 1) * self.num_layers,
                                     attn_size)
        self.ctx_proj = nn.Linear(self.attn_emb_dim, emb_dim)
        self.addinfo_proj = nn.Linear(self.attn_emb_dim, emb_dim)
        self.addinfo_embedding = nn.Linear(add_vocab_size, attn_emb_dim)
        self.apply(init)

    def forward(self, input_dict):
        word = input_dict["word"]
        state = input_dict.get("state", None) # [n_layer * n_dire, bs, d_model]
        fc_embs = input_dict["fc_embs"]
        attn_embs = input_dict["attn_embs"]
        attn_emb_lens = input_dict["attn_emb_lens"]
        add_info = input_dict["add_info"]
        add_info_lens = input_dict["add_info_lens"]

        if word.size(-1) == self.fc_emb_dim: # fc_embs
            embed = word.unsqueeze(1)
        elif word.size(-1) == 1: # word
            word = word.to(fc_embs.device)
            embed = self.in_dropout(self.word_embedding(word))
        else:
            raise Exception(f"problem with word input size {word.size()}")

        # embed: [N, 1, embed_size]
        if state is None:
            state = self.init_hidden(word.size(0), fc_embs.device)
        if self.rnn_type == "LSTM":
            query = state[0].transpose(0, 1).flatten(1)
        else:
            query = state.transpose(0, 1).flatten(1)
        c, attn_weight = self.attn(query, attn_embs, attn_emb_lens)

        p_ctx = self.ctx_proj(c)
        
        add_info_embs = self.addinfo_embedding(add_info)
        add_info_c, add_info_weight = self.add_info_attn(query, add_info_embs, add_info_lens)
        add_info_ctx = self.addinfo_proj(add_info_c)
        rnn_input = torch.cat((embed, p_ctx.unsqueeze(1), add_info_ctx.unsqueeze(1)), dim=-1)

        out, state = self.model(rnn_input, state)

        output = {
            "state": state,
            "embeds": out,
            "logits": self.classifier(out),
            "weights": attn_weight
        }
        return output

class BahAttnAddinfoGtDecoder(BahAttnAddinfoCatattnDecoder):

    def __init__(self, emb_dim, vocab_size, add_vocab_size, fc_emb_dim, attn_emb_dim, \
            dropout, d_model, **kwargs):
        super().__init__(emb_dim, vocab_size, add_vocab_size, fc_emb_dim, attn_emb_dim, dropout, d_model, **kwargs)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: [T, N, E]
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerDecoder(BaseDecoder):

    def __init__(self, emb_dim, vocab_size, fc_emb_dim, attn_emb_dim, dropout, **kwargs):
        super().__init__(emb_dim, vocab_size, fc_emb_dim, attn_emb_dim, dropout=dropout)
        self.d_model = emb_dim
        self.nhead = kwargs.get("nhead", self.d_model // 64)
        self.nlayers = kwargs.get("nlayers", 2)
        self.dim_feedforward = kwargs.get("dim_feedforward", self.d_model * 4)

        self.pos_encoder = PositionalEncoding(self.d_model, dropout)
        layer = nn.TransformerDecoderLayer(d_model=self.d_model,
                                           nhead=self.nhead,
                                           dim_feedforward=self.dim_feedforward,
                                           dropout=dropout)
        self.model = nn.TransformerDecoder(layer, self.nlayers)
        self.classifier = nn.Linear(self.d_model, vocab_size)
        # self.attn_proj = nn.Sequential(
            # nn.Linear(self.attn_emb_dim, self.d_model),
            # nn.ReLU(),
            # nn.Dropout(dropout),
            # nn.LayerNorm(self.d_model)
        # )
        self.attn_proj = lambda x: x
        self.init_params()

    def init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_square_subsequent_mask(self, max_length):
        mask = (torch.triu(torch.ones(max_length, max_length)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, input_dict):
        word = input_dict["word"]
        attn_embs = input_dict["attn_embs"]
        attn_emb_lens = input_dict["attn_emb_lens"]
        caps_padding_mask = input_dict["caps_padding_mask"]
       
        p_attn_embs = self.attn_proj(attn_embs)
        p_attn_embs = p_attn_embs.transpose(0, 1) # [T_src, N, emb_dim]
        word = word.to(attn_embs.device)
        embed = self.in_dropout(self.word_embedding(word)) * math.sqrt(self.emb_dim) # [N, T, emb_dim]
        # embed = self.word_embedding(word) * math.sqrt(self.emb_dim) # [N, T, emb_dim]
        embed = embed.transpose(0, 1) # [T, N, emb_dim]
        embed = self.pos_encoder(embed)

        tgt_mask = self.generate_square_subsequent_mask(embed.size(0)).to(attn_embs.device)
        memory_key_padding_mask = ~generate_length_mask(attn_emb_lens, attn_embs.size(1)).to(attn_embs.device)
        output = self.model(embed, p_attn_embs, tgt_mask=tgt_mask,
                            tgt_key_padding_mask=caps_padding_mask,
                            memory_key_padding_mask=memory_key_padding_mask)
        output = output.transpose(0, 1)
        output = {
            "embeds": output,
            "logits": self.classifier(output),
        }
        return output


class TransformerTemporalDecoder(TransformerDecoder):

    def __init__(self, emb_dim, vocab_size, fc_emb_dim, attn_emb_dim, dropout, **kwargs):
        super().__init__(emb_dim, vocab_size, fc_emb_dim, attn_emb_dim, dropout=dropout, **kwargs)
        self.temporal_embedding = nn.Embedding(4, emb_dim)
        self.init_params()

    def forward(self, input_dict):
        word = input_dict["word"]
        attn_embs = input_dict["attn_embs"]
        attn_emb_lens = input_dict["attn_emb_lens"]
        caps_padding_mask = input_dict["caps_padding_mask"]
        temporal_label = input_dict["temporal_label"]
         
        
        p_attn_embs = self.attn_proj(attn_embs)
        p_attn_embs = p_attn_embs.transpose(0, 1) # [T_src, N, emb_dim]

        temporal_embed = self.in_dropout(self.temporal_embedding(temporal_label)).unsqueeze(1)
        word = word.to(attn_embs.device)
        word_embed = self.in_dropout(self.word_embedding(word))  #[N, T, emb_dim]
        embed = torch.cat((temporal_embed, word_embed[:, 1:, :]), dim=1) * math.sqrt(self.emb_dim)
        # embed = self.word_embedding(word) * math.sqrt(self.emb_dim) # [N, T, emb_dim]
        embed = embed.transpose(0, 1) # [T, N, emb_dim]
        embed = self.pos_encoder(embed)

        tgt_mask = self.generate_square_subsequent_mask(embed.size(0)).to(attn_embs.device)
        memory_key_padding_mask = ~generate_length_mask(attn_emb_lens, attn_embs.size(1)).to(attn_embs.device)
        output = self.model(embed, p_attn_embs, tgt_mask=tgt_mask,
                            tgt_key_padding_mask=caps_padding_mask,
                            memory_key_padding_mask=memory_key_padding_mask)
        output = output.transpose(0, 1)
        output = {
            "embeds": output,
            "logits": self.classifier(output),
        }
        return output
class TransformerAddinfoDecoder(BaseDecoder):

    def __init__(self, emb_dim, vocab_size, add_vocab_size, fc_emb_dim, attn_emb_dim, dropout, **kwargs):
        super().__init__(emb_dim, vocab_size, fc_emb_dim, attn_emb_dim, dropout=dropout)
        self.d_model = emb_dim
        self.nhead = kwargs.get("nhead", self.d_model // 64)
        self.nlayers = kwargs.get("nlayers", 2)
        self.dim_feedforward = kwargs.get("dim_feedforward", self.d_model * 4)

        self.pos_encoder = PositionalEncoding(self.d_model, dropout)
        layer = TransformerAddinfoDecoderLayer(d_model=self.d_model,
                                           nhead=self.nhead,
                                           dim_feedforward=self.dim_feedforward,
                                           dropout=dropout)
        self.model = TransformerAddinfoBaseDecoder(layer, self.nlayers)
        self.addinfo_embedding = nn.Embedding(add_vocab_size, emb_dim)
        self.classifier = nn.Linear(self.d_model, vocab_size)
        # self.attn_proj = nn.Sequential(
            # nn.Linear(self.attn_emb_dim, self.d_model),
            # nn.ReLU(),
            # nn.Dropout(dropout),
            # nn.LayerNorm(self.d_model)
        # )
        self.attn_proj = nn.Linear(attn_emb_dim, emb_dim)
        self.init_params()

    def init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_square_subsequent_mask(self, max_length):
        mask = (torch.triu(torch.ones(max_length, max_length)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, input_dict):
        word = input_dict["word"]
        attn_embs = input_dict["attn_embs"]
        attn_emb_lens = input_dict["attn_emb_lens"]
        caps_padding_mask = input_dict["caps_padding_mask"]
        add_info = input_dict["add_info"]
        add_info_lens = input_dict["add_info_lens"]
        add_info_embs = self.addinfo_embedding(add_info)
        p_add_info_embs = add_info_embs.transpose(0, 1) # [L, N, emb_dim]
        
        p_attn_embs = self.attn_proj(attn_embs)
        p_attn_embs = p_attn_embs.transpose(0, 1) # [T_src, N, emb_dim]
        word = word.to(attn_embs.device)
        embed = self.in_dropout(self.word_embedding(word)) * math.sqrt(self.emb_dim) # [N, T, emb_dim]
        # embed = self.word_embedding(word) * math.sqrt(self.emb_dim) # [N, T, emb_dim]
        embed = embed.transpose(0, 1) # [T, N, emb_dim]
        embed = self.pos_encoder(embed)
        tgt_mask = self.generate_square_subsequent_mask(embed.size(0)).to(attn_embs.device)
        memory_key_padding_mask = ~generate_length_mask(attn_emb_lens, attn_embs.size(1)).to(attn_embs.device)
        add_info_memory_key_padding_mask =  ~generate_length_mask(add_info_lens, add_info_embs.size(1)).to(add_info_embs.device)
        output = self.model(embed, p_attn_embs, p_add_info_embs, tgt_mask=tgt_mask,
                            tgt_key_padding_mask=caps_padding_mask,
                            memory_key_padding_mask=memory_key_padding_mask,
                            add_info_memory_key_padding_mask=add_info_memory_key_padding_mask)
        output = output.transpose(0, 1)
        output = {
            "embeds": output,
            "logits": self.classifier(output),
        }
        return output

class M2TransformerDecoder(BaseDecoder):

    def __init__(self, vocab_size, fc_emb_dim, attn_emb_dim, dropout=0.1, **kwargs):
        super().__init__(attn_emb_dim, vocab_size, fc_emb_dim, attn_emb_dim, dropout=dropout)
        try:
            from m2transformer.models.transformer import MeshedDecoder
        except:
            raise ImportError("meshed-memory-transformer not installed; please run `pip install git+https://github.com/ruotianluo/meshed-memory-transformer.git`")
        del self.word_embedding
        del self.in_dropout

        self.d_model = attn_emb_dim
        self.nhead = kwargs.get("nhead", self.d_model // 64)
        self.nlayers = kwargs.get("nlayers", 2)
        self.dim_feedforward = kwargs.get("dim_feedforward", self.d_model * 4)
        self.model = MeshedDecoder(vocab_size, 100, self.nlayers, 0,
                                   d_model=self.d_model,
                                   h=self.nhead,
                                   d_ff=self.dim_feedforward,
                                   dropout=dropout)
        self.init_params()

    def init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_dict):
        word = input_dict["word"]
        attn_embs = input_dict["attn_embs"]
        attn_emb_mask = input_dict["attn_emb_mask"]
        word = word.to(attn_embs.device)
        embeds, logits = self.model(word, attn_embs, attn_emb_mask)
        output = {
            "embeds": embeds,
            "logits": logits,
        }
        return output

from torch import Tensor
from typing import Optional, Any
class TransformerAddinfoBaseDecoder(nn.Module):
    r"""TransformerAddinfoDecoder is a stack of N decoder layers

    Args:
        decoder_layer: an instance of the TransformerAddinfoDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> decoder_layer = nn.TransformerAddinfoDecoderLayer(d_model=512, nhead=8)
        >>> transformer_decoder = nn.TransformerAddinfoDecoder(decoder_layer, num_layers=6)
        >>> memory1 = torch.rand(10, 32, 512)
        >>> memory2 = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = transformer_decoder(tgt, memory)
    """
    __constants__ = ['norm']
    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerAddinfoBaseDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt: Tensor, memory: Tensor, add_info_memory: Tensor, tgt_mask: Optional[Tensor] = None,
            memory_mask: Optional[Tensor] = None, add_info_memory_mask: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None,
            add_info_memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:

        """
        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            add_info_memory
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            add_info_memory_mask
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = tgt

        for mod in self.layers:
            output = mod(output, memory, add_info_memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask, add_info_memory_mask=add_info_memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask,
                         add_info_memory_key_padding_mask=add_info_memory_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerAddinfoDecoderLayer(nn.Module):
    """
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> memory = torch.rand(10, 32, 512)
        >>> add_info_memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = decoder_layer(tgt, memory)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerAddinfoDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.add_info_multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.add_info_dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = nn.functional.relu
        super(TransformerAddinfoDecoderLayer, self).__setstate__(state)

    def forward(self, tgt: Tensor, memory: Tensor, add_info_memory: Tensor, tgt_mask: Optional[Tensor] = None, 
            memory_mask: Optional[Tensor] = None, add_info_memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, 
                memory_key_padding_mask: Optional[Tensor] = None,
                add_info_memory_key_padding_mask: Optional[Tensor] = None)  -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            add_info_memory:
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            add_info_memory_mask
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        add_info_memory = add_info_memory.squeeze(-2)
        tgt2 = self.add_info_multihead_attn(tgt, add_info_memory, add_info_memory, 
                attn_mask=add_info_memory_mask, 
                key_padding_mask=add_info_memory_key_padding_mask)[0]
        tgt = tgt + self.add_info_dropout2(tgt2)

        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

import copy
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return torch.nn.functional.relu
    elif activation == "gelu":
        return torch.nn.functional.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))
