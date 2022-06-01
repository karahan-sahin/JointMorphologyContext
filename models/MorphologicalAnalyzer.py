import torch
from torch import nn
from _utils__.utils_ import TokenEmbedding, PositionalEncoding
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from _utils__.config import params


from models.CharEncoderDecoder import CharacterLSTMDecoder, CharacterLSTMEncoder


class SentenceMorphologicalParser(nn.Module):
    
  def __init__(self):
    
      super(SentenceMorphologicalParser, self).__init__()

      self.encoder = CharacterLSTMEncoder()

      self.positional_encoding = PositionalEncoding(params["CharacterContextEncoder"]["PositionalEncoding"]["embedding_size"], 
                                                    params["CharacterContextEncoder"]["PositionalEncoding"]["dropout"])

      encoder_layer = TransformerEncoderLayer(params["CharacterContextEncoder"]["TransformerEncoder"]["embedding_size"], 
                                              params["CharacterContextEncoder"]["TransformerEncoder"]["nhead"],
                                              params["CharacterContextEncoder"]["TransformerEncoder"]["dim_feedforward"])
      

      self.transformer_encoder = TransformerEncoder(encoder_layer, 
                                                    params["CharacterContextEncoder"]["TransformerEncoder"]["num_encoder_layers"])


      self.decoder = CharacterLSTMDecoder()


  def init_weights(self):
    initrange = 0.1
    self.encoder.weight.data.uniform_(-initrange, initrange)
    self.decoder.bias.data.zero_()
    self.decoder.weight.data.uniform_(-initrange, initrange)

  def forward(self, sources, targets):
    # Character Encoder
    # Put each token of the sentence into character encoder
    char_out = []
    hidden_out = []
    for source, target in zip(sources, targets):
      (hidden, cell) = self.encoder(source)
      char_out.append(cell)
      hidden_out.append(hidden)

    char_out = torch.cat(char_out, dim=0)

    # # Char Context
    src_emb = self.positional_encoding(char_out)
    context_out = self.transformer_encoder(src_emb)

    # # Residual Connections from Character Encoder
    # char_res = torch.cat((context_out,char_out),1)

    predictions = []
    last_cells = []
    for cell, hidden, target in zip(context_out, hidden_out, targets):
      cell = torch.unsqueeze(cell, dim=0)
      prediction, outs = self.decoder(target, hidden, cell)
      predictions.append(prediction)
      last_cells.append(outs)

    return predictions, torch.cat(last_cells, dim=0)