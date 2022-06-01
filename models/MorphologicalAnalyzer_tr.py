import torch
from torch import nn, Tensor
from torch.nn.utils.rnn import pad_sequence
from _utils__.utils_ import TokenEmbedding, PositionalEncoding, create_mask
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch import nn

from _utils__.config import params

from models.TransformerEncDec  import CharTransformerEncoder, CharTransformerDecoder


class SentenceMorphologicalParser(nn.Module):
    
  def __init__(self):
    
      super().__init__()

      self.encoder = CharTransformerEncoder()
    
      self.context_in = nn.LazyLinear(256) 
      
      self.positional_encoding = PositionalEncoding(params["CharacterContextEncoder"]["PositionalEncoding"]["embedding_size"], 
                                                    params["CharacterContextEncoder"]["PositionalEncoding"]["dropout"])

      encoder_layer = TransformerEncoderLayer(params["CharacterContextEncoder"]["TransformerEncoder"]["embedding_size"], 
                                              params["CharacterContextEncoder"]["TransformerEncoder"]["nhead"],
                                              params["CharacterContextEncoder"]["TransformerEncoder"]["dim_feedforward"])


      self.transformer_encoder = TransformerEncoder(encoder_layer, 
                                                    params["CharacterContextEncoder"]["TransformerEncoder"]["num_encoder_layers"])

      self.flat = nn.Flatten(1,2)
      
      self.context_out = nn.Linear(256, 512*25) 


      self.decoder = CharTransformerDecoder()


  def forward(self, src: Tensor, tgt: Tensor):

    # Character Encoder
    # Put each token of the sentence into character encoder
    char_out = []
    for source, target in zip(src, tgt):
      for _ in range(source.size(0), 26):
        source = torch.cat([source, torch.unsqueeze(torch.tensor([1 for _ in range(20)]), 0)], dim=0)
      if source.size(0) > 26:
            source = source[:26,:]
      tgt_input = target[:-1, :]
      src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(source, tgt_input)
      memory = self.encoder(source, src_mask, src_padding_mask)
      memory = memory.transpose(0, 1)
      memory = self.flat(memory) # Flat encoded chars-of-word [batch_len, seq_len*hidden_dim ] torch.Size([18, 20, 512])

      char_out.append(memory) #

    char_out = pad_sequence(char_out, batch_first=True, padding_value=1)
    char_out = self.context_in(char_out)

    # # Char Context
    src_emb = self.positional_encoding(char_out)
    context_out = self.transformer_encoder(src_emb)
    context_dec = self.context_out(context_out)

    # # Residual Connections from Character Encoder
    # char_res = torch.cat((context_out,char_out),1)

    predictions = []
    for source, target, ctx in zip(src, tgt, context_dec):
      for _ in range(source.size(0), 25):
        source = torch.cat([source, torch.unsqueeze(torch.tensor([1 for _ in range(20)]), 0)], dim=0)
      if source.size(0) > 25:
        source = source[:25,:]
      tgt_input = target[:-1, :]
      src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(source, tgt_input)
      memory = torch.reshape(ctx, 
                            (source.shape[1], source.shape[0], 512))
      
      memory = memory.transpose(1, 0)

      outs = self.decoder(memory, tgt_input, tgt_mask, 
                          tgt_padding_mask, src_padding_mask)

      predictions.append(outs)

    return predictions, context_out