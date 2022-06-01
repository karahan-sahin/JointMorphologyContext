import imp
import torch
import random
from _utils__.config import params, device


from torch import nn, Tensor
from torch.nn.utils.rnn import pad_sequence

from _utils__.utils_ import PositionalEncoding, TokenEmbedding
from torch.nn import (TransformerEncoder, TransformerDecoder,
                      TransformerEncoderLayer, TransformerDecoderLayer)


class CharTransformerEncoder(nn.Module):
    def __init__(self):

        super(CharTransformerEncoder, self).__init__()

                                        
        self.src_tok_emb = TokenEmbedding(params["CharTrEncoder"]["src_vocab_size"], 
                                          params["CharTrEncoder"]["embedding_size"])

        self.positional_encoding = PositionalEncoding(params["CharTrEncoder"]["embedding_size"], 
                                                      dropout=params["CharTrEncoder"]["dropout"])

        encoder_layer = TransformerEncoderLayer(d_model=params["CharTrEncoder"]["embedding_size"], 
                                                nhead=params["CharTrEncoder"]["NHEAD"],
                                                dim_feedforward=params["CharTrEncoder"]["feed_forward_dim"])

        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=params["CharTrEncoder"]["NUM_ENCODER_LAYERS"])


    def forward(self, src: Tensor, src_mask: Tensor, src_padding_mask: Tensor):
      
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        memory = self.transformer_encoder(src_emb, 
                                          src_mask, 
                                          src_padding_mask)

        return memory


class CharTransformerDecoder(nn.Module):
    def __init__(self):

        super(CharTransformerDecoder, self).__init__()

        self.tgt_tok_emb = TokenEmbedding(params["CharTrDecoder"]["tgt_vocab_size"], 
                                          params["CharTrDecoder"]["embedding_size"])

        self.positional_encoding = PositionalEncoding(emb_size=params["CharTrDecoder"]["embedding_size"],
                                                      dropout=params["CharTrDecoder"]["dropout"])

        decoder_layer = TransformerDecoderLayer(d_model=params["CharTrDecoder"]["embedding_size"], 
                                                nhead=params["CharTrDecoder"]["NHEAD"],
                                                dim_feedforward=params["CharTrDecoder"]["feed_forward_dim"])
        
        self.transformer_decoder = TransformerDecoder(decoder_layer, 
                                                      num_layers=params["CharTrDecoder"]["NUM_DECODER_LAYERS"])
                
        self.generator = nn.Linear(params["CharTrDecoder"]["embedding_size"], params["CharTrDecoder"]["tgt_vocab_size"])

    def forward(self, memory: Tensor, tgt: Tensor, tgt_mask: Tensor, 
                tgt_padding_mask: Tensor, memory_key_padding_mask: Tensor):

        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        outs = self.transformer_decoder(tgt_emb,
                                        memory, 
                                        tgt_mask, 
                                        None,
                                        tgt_padding_mask, 
                                        memory_key_padding_mask)
        return self.generator(outs)