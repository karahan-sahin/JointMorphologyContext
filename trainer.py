import torch
from torch import nn, Tensor
from tqdm import tqdm
from _utils__.config import params, device, bptt, ntokens

from torch.utils.data import dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from _utils__.utils_ import generate_square_subsequent_mask, DataLoader, Vocab
from models.LanguageModel import TransformerLM
from models.MorphologicalAnalyzer import SentenceMorphologicalParser as rnn_parser
from models.MorphologicalAnalyzer_tr import SentenceMorphologicalParser as tr_parser
import argparse
import time
import os

def train(model: nn.Module, raw, morph,  model_type="rnn", ntokens=ntokens,):
    model.train()  # turn on train mode
    total_loss = 0.
    src_mask = generate_square_subsequent_mask(bptt).to(device)
    batch =0
    loader = DataLoader("train_data",
                        raw,
                        morph)

    
    for _ in tqdm(range(len(os.listdir("train_data")))):

        raw_batch, morph_batch, next = loader.get_batch()
        morph_out, next_out = model(raw_batch, morph_batch, src_mask)
        next_loss = criterion(next_out.view(-1, ntokens), next.reshape(-1))
        total_morph_loss = 0
        for m_out, m_true in zip(morph_out, morph_batch):
            output = m_out.view(-1, m_out.shape[-1])
            trg = m_true[:-1, :].reshape(-1) if model_type == "tr" else m_true.reshape(-1)
            morph_loss = criterion(output, trg)
            total_morph_loss += morph_loss

        loss = (total_morph_loss + next_loss) / 2

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()

        batch +=1


if __name__ ==  "__main__":

    parser = argparse.ArgumentParser(description="CMPE58T Application Project",add_help=True,formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-t","--encoderType", default='rnn',type=str,help="Select the encoder type (rnn for LSTM-based encoder-decoder or tr for Transformer-based)",)
    parser.add_argument("-e","--epochs",default=10,type=int,help="Define Number of Epochs")
    parser.add_argument("-p","--path", default='./',type=str,help="Select the path to save weights",)
    args=parser.parse_args()

    raw_vocab, morph_vocab =  Vocab("vocab/raw_vocab"), Vocab("vocab/morph_vocab")
    raw_next_vocab, morph_next_vocab =  Vocab(f"vocab/train/raw_next_vocab"), Vocab(f"vocab/train/morph_next_vocab")

    if args.encoderType == "rnn":
        parser=rnn_parser
    
    elif args.encoderType == "tr":
        parser=tr_parser
    
    model = TransformerLM(parser).to(device)

    criterion = nn.CrossEntropyLoss()
    lr = 5.0  # learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    for epoch in range(1, args.epochs + 1):

        epoch_start_time = time.time()
        train(model, raw_vocab, morph_vocab, model_type=args.encoderType)

        scheduler.step()

        checkpoint = {
        'model_state_dict': model.state_dict(),
        'source': raw_vocab.vocab,
        'target': morph_vocab.vocab
        }

        torch.save(checkpoint, f'{args.path}/morph-lm-epoch={epoch}.pth')

        elapsed = time.time() - epoch_start_time