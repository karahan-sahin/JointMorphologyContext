import torch
from torch import nn, Tensor

from _utils__.config import params
from _utils__.utils_ import TokenEmbedding, PositionalEncoding

from torch.nn import TransformerEncoderLayer, TransformerEncoder


class TransformerLM(nn.Module):
    
    def __init__(self,parser):
      

        super(TransformerLM, self).__init__()

        self.morph_parser = parser()

        self.pos_encoder = PositionalEncoding(params["TransformerLanguageModel"]["embedding_size"], 
                                              params["TransformerLanguageModel"]["dropout"], )

        encoder_layers = TransformerEncoderLayer(params["TransformerLanguageModel"]["embedding_size"],
                                                 params["TransformerLanguageModel"]["nhead"], 
                                                 params["TransformerLanguageModel"]["hidden_dim"], 
                                                 params["TransformerLanguageModel"]["dropout"])

        self.transformer_encoder = TransformerEncoder(encoder_layers, 
                                                      params["TransformerLanguageModel"]["n_layers"])

        self.decoder = nn.Linear(params["TransformerLanguageModel"]["embedding_size"],
                                 params["TransformerLanguageModel"]["tgt_vocab_size"])

        self.init_weights()

    def init_weights(self): # Look at init
        initrange = 0.1
        # self.transformer_encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Tensor):
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        morph_out, last_cell = self.morph_parser(src,tgt)

        src = self.pos_encoder(last_cell)

        output = self.transformer_encoder(src, src_mask)

        output = self.decoder(output)

        return morph_out ,output