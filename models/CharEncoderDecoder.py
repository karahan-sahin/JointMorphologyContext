import imp
import torch
import random
from _utils__.config import params, device


from torch import nn, Tensor
from torch.nn.utils.rnn import pad_sequence


class CharacterLSTMEncoder(nn.Module):
  """The character encoder module where a given a word is feed through the Bidirectional LSTM model to provide
  character-level word representation.

  Args:
      nn (_type_): _description_
  """

  def __init__(self):

    super(CharacterLSTMEncoder, self).__init__()

    self.embedding = nn.Embedding(params["CharacterEncoder"]["EmbeddingLayer"]["vocab_len"], 
                                  params["CharacterEncoder"]["EmbeddingLayer"]["embedding_dim"])

    self.rnn = nn.LSTM(params["CharacterEncoder"]["LSTM"]["embedding_dim"], 
                       params["CharacterEncoder"]["LSTM"]["hidden_dim"], 
                       params["CharacterEncoder"]["LSTM"]["n_layers"], 
                       dropout=params["CharacterEncoder"]["LSTM"]["dropout"])

    self.dropout = nn.Dropout(params["CharacterEncoder"]["LSTM"]["dropout"])

  def forward(self, input_batch):
        
    embed = self.dropout(self.embedding(input_batch))
    outputs, (hidden, cell) = self.rnn(embed)

    return (hidden, 
            cell) # This is the final state


class CharacterStateDecoder(nn.Module):

  
  def __init__(self):
    
      super().__init__()


      # self.input_output_dim will be used later

      self.input_output_dim = params["CharacterDecoder"]["CharacterStateDecoder"]["LSTM"]["vocab_len"]

      self.embedding = nn.Embedding(params["CharacterDecoder"]["CharacterStateDecoder"]["EmbeddingLayer"]["vocab_len"], 
                                    params["CharacterDecoder"]["CharacterStateDecoder"]["EmbeddingLayer"]["embedding_dim"])

      self.rnn = nn.LSTM(params["CharacterDecoder"]["CharacterStateDecoder"]["LSTM"]["embedding_dim"], 
                         params["CharacterDecoder"]["CharacterStateDecoder"]["LSTM"]["hidden_dim"], 
                         params["CharacterDecoder"]["CharacterStateDecoder"]["LSTM"]["n_layers"], 
                         dropout=params["CharacterDecoder"]["CharacterStateDecoder"]["LSTM"]["dropout"])

      self.fc = nn.Linear(params["CharacterDecoder"]["CharacterStateDecoder"]["Linear(Softmax)"]["hidden_dim"], 
                          params["CharacterDecoder"]["CharacterStateDecoder"]["Linear(Softmax)"]["target_dim"]) # len(morph_vocab)

    
      self.dropout = nn.Dropout(params["CharacterEncoder"]["LSTM"]["dropout"])


  def forward(self, target_token, hidden, cell):
        
      target_token = target_token.unsqueeze(0)

      # Embedding Layer
      embedding_layer = self.dropout(self.embedding(target_token)) # Embedding Layer with Dropout (?)

      output, (hidden, cell) = self.rnn(embedding_layer, 
                                        (hidden, cell)) # LSTM Model

      linear = self.fc(output.squeeze(0)) # Final Softmax

      return linear, hidden, cell


class CharacterLSTMDecoder(nn.Module):
      
  def __init__(self):
    
      super().__init__()

      self.one_step_decoder = CharacterStateDecoder()

  def forward(self, target, hidden, cell):

    target_len, batch_size = target.shape[0], target.shape[1]

    target_vocab_size = params["CharacterDecoder"]["CharacterStateDecoder"]["Linear(Softmax)"]["target_dim"]
    predictions = torch.zeros(target_len, batch_size, target_vocab_size).to(device)
    input = target[0, :]
        
    last_cell = None
    for t in range(target_len):
          
        predict, hidden, cell = self.one_step_decoder(input, hidden, cell)

        last_cell = cell
        predictions[t] = predict
        input= predict.argmax(1)        

        # Teacher forcing
        do_teacher_forcing = random.random() < params["CharacterDecoder"]["teaching_force_ratio"]   

        input = target[t] if do_teacher_forcing else input

    return predictions, last_cell