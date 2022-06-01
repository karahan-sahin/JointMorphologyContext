import os
import math
import pickle
import torch
from torch import Tensor,nn
from torch.nn.utils.rnn import pad_sequence
from _utils__.config import device


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)

    src_padding_mask = (src == 2).transpose(0, 1)
    tgt_padding_mask = (tgt == 2).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + 
                            self.pos_embedding[:token_embedding.size(0),:])

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size
    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


def generate_square_subsequent_mask(sz: int):
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)



def Vocab(vocab_file):
    return  pickle.load(open(vocab_file, "rb"))


class DataLoader():
     
    def __init__(self, data_path, raw_vocab, morph_vocab):
        self.data_path = data_path
        self.batch = 0
        self.raw_vocab =  raw_vocab
        self.morph_vocab =  morph_vocab


    def get_batch(self):
        loaded = torch.load(f'train_data/train_batch_{self.batch}_db')
        raw_batch, morph_batch, next  = loaded["raw"],loaded["morph"],loaded["next"]

        r_batch = [[] for _ in range(len(raw_batch))]
        for r in raw_batch:
            for idx in range(r.shape[1]):
                r_batch[idx].append(r[:,idx])

        raw_batch = [pad_sequence(r, padding_value=self.raw_vocab["<pad>"]) for r in r_batch]

        r_batch = [[] for _ in range(len(morph_batch))]
        for r in morph_batch:
            for idx in range(r.shape[1]):
                r_batch[idx].append(r[:,idx])

        morph_batch = [pad_sequence(r, padding_value=self.morph_vocab["<pad>"]) for r in r_batch]

        self.batch += 1

        return raw_batch, morph_batch, next