import re
import numpy as np
from tqdm import tqdm
from config import device, bptt

import argparse
import os
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from _utils___ import Vocab

import torch
from torch import Tensor

from torch.utils.data import dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


def enc(morph):
    return " ".join(morph.split("+")[0]) + " " + " ".join(morph.split("+")[1:])


def sentence_parser(sentence):
    sentence = re.sub("<.+?>", "", sentence[0])
    tokens = [t for t in sentence.split("\n") if t]
    pairs = []
    for idx, token in enumerate(tokens):
        raw, morph = token.split("\t")[:2]
        try:
            pairs.append((raw,
                            morph,
                            tokens[idx+1].split("\t")[0]))
        except:
            pairs.append((raw,
                            morph,
                            "<eos>"))
    return pairs

def batchify(data: Tensor, bsz: int):
    """Divides the data into bsz separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Args:
        data: Tensor, shape [N]
        bsz: int, batch size

    Returns:
        Tensor of shape [N // bsz, bsz]
    """
    seq_len = data.size(0) // bsz
    data = data[:seq_len * bsz]
    data = data.view(bsz, seq_len).t().contiguous()

    return data.to(device)

def get_iter(sents):
    raw_next_iter = []
    morph_next_iter = []
    for d in sents:
        raw = []
        morph = []
        if sentence_parser(d):
            for i in sentence_parser(d):
                raw.append(i[0])
                morph.append(i[1])
            raw_next_iter.append(raw)
            morph_next_iter.append(morph)

    return raw_next_iter, morph_next_iter

def get_batch(raw: Tensor, morph: Tensor, i: int):
    """
    Args:
        source: Tensor, shape [full_seq_len, batch_size]
        i: int

    Returns:
        tuple (data, target), where data has shape [seq_len, batch_size] and
        target has shape [seq_len * batch_size]
    """
    seq_len = min(bptt, len(raw) - 1 - i)
    data = raw[i:i+seq_len]
    morph = morph[i:i+seq_len]
    target = raw[i+1:i+1+seq_len]
    return data, morph, target

def data_process(raw_text_iter, vocab):
    """Converts raw text into a flat Tensor."""
    data = []
    for d in raw_text_iter:
        sent = []
        for i in d:
            try:
                sent.append([vocab[i]])
            except:
                pass

        data.append(torch.tensor(sent, dtype=torch.long))


    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="CMPE58T Application Project",add_help=True,formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-d","--dataType", default='',type=str,help="Select the data type (train/test)")
    parser.add_argument("-b","--batch", default=20,type=int,help="Select the batch size")
    parser.add_argument("-p","--path", default='',type=str,help="Select the data path")
    args=parser.parse_args()

    data = open(args.path, "r", encoding="utf-8").read()
    sents = re.findall("<S>\+BSTag((.|\n)+?)</S>\+ESTag", data)

    raw_next_iter, morph_next_iter =  get_iter(sents)

    raw_vocab, morph_vocab =  Vocab("../vocab/raw_vocab"), Vocab("../vocab/morph_vocab")
    raw_next_vocab, morph_next_vocab =  Vocab(f"../vocab/{args.dataType}/raw_next_vocab"), Vocab(f"../vocab/{args.dataType}/morph_next_vocab")

    raw_train_data = data_process(raw_next_iter, raw_next_vocab)
    morph_train_data = data_process(morph_next_iter, morph_next_vocab)

    raw_train_data = batchify(raw_train_data, args.batch)
    morph_train_data = batchify(morph_train_data, args.batch)

    raw_iter = []
    morph_iter = []
    next_iter = []
    batch = 0
    try:
        os.mkdir(f"{args.dataType}_data")
    except:
        pass
    for i in tqdm(range(0, 
                        raw_train_data.size(0) - 1, 
                        bptt)):
    
        data, morph, next = get_batch(raw_train_data, morph_train_data,i)

        raw_batch = []
        morph_batch = []

        for id in range(data.size(0)):
                        
            r = pad_sequence([torch.cat([torch.tensor([raw_vocab["<bos>"]]), torch.tensor([raw_vocab[x] for x in raw_next_vocab.get_itos()[i.item()]], dtype=torch.long), torch.tensor([raw_vocab["<eos>"]])], dim=0) for i in data[id,:]], padding_value=raw_vocab["<pad>"])
            m = pad_sequence([torch.cat([torch.tensor([morph_vocab["<bos>"]]), torch.tensor([morph_vocab[x] for x in enc(morph_next_vocab.get_itos()[i.item()]).split()], dtype=torch.long), torch.tensor([morph_vocab["<eos>"]])], dim=0) for i in morph[id,:]], padding_value=raw_vocab["<pad>"])
            raw_batch.append(r)
            morph_batch.append(m)
        
        next_batch = next.reshape(-1)

        torch.save({
        "raw": raw_iter,
        "morph": morph_iter,
        "next": next_iter,
        }, f"{args.dataType}_data/{args.dataType}_batch_{batch}_db")
        batch +=1