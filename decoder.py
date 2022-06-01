import torch
from _utils__.utils_ import Vocab
from _utils__.config import device


def MorphologyDecoder(model, raw_batch):
    _, morph_vocab =  Vocab("vocab/raw_vocab"), Vocab("vocab/morph_vocab")
    
    model.eval()

    char_out = []
    hidden_out = []
    for source in raw_batch:
        (hidden, cell) = model.morph_parser.encoder(source)
        char_out.append(cell)
        hidden_out.append(hidden)

    char_out = torch.cat(char_out, dim=0)

    # # Char Context
    src_emb = model.morph_parser.positional_encoding(char_out)
    context_out = model.morph_parser.transformer_encoder(src_emb)

    predictions = []
    for t,(cell, hidden) in enumerate(zip(context_out, hidden_out)):

        with torch.no_grad():
            # U
            cell = torch.unsqueeze(torch.unsqueeze(cell[t], dim=0), dim=0)
            hidden = torch.unsqueeze(hidden, dim=0)

            trg_index = [morph_vocab['<bos>']]
            next_token = torch.LongTensor(trg_index).to(device)
            outputs = []
            for _ in range(20):

                output, hidden, cell = model.morph_parser.decoder.one_step_decoder(next_token,hidden, cell)

                # Take the most probable word
                next_token = output.argmax(1)
                predicted = morph_vocab.get_itos()[output.argmax(1).item()]
                # print(predicted)
                if predicted == '<eos>':
                    break
                elif output.argmax(1).item():
                    outputs.append(predicted)
        
        pred = "".join(outputs)
        predictions.append(pred)
    
    return predictions