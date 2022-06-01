import torch

device = torch.device('cpu')
bptt = 20
ntokens = 111463

params = {

    "CharacterEncoder": {

        "EmbeddingLayer": {

            "vocab_len": 168,
            "embedding_dim": 256

        },

        "LSTM": {
            "embedding_dim": 256,
            "hidden_dim": 256,
            "n_layers": 1,
            "dropout": 0.5,
            "bidirectional": True
        }
    },

    "CharTrEncoder": {
        "NHEAD": 8, 
        "NUM_ENCODER_LAYERS": 3, 
        "embedding_size": 512, 
        "src_vocab_size": 97, 
        "feed_forward_dim": 512,
        "dropout": 0.5
    },

    "CharacterContextEncoder": {
        
        "PositionalEncoding": {

            "embedding_size": 256,
            "dropout": 0.1

        },

        "TransformerEncoder": {

            "embedding_size": 256,
            "nhead": 6,

            "src_vocab_size": 256,
            "tgt_vocab_size": 256,

            "num_encoder_layers": 2,

            "nhead": 8,
            "hidden_dim": 512,
            "dim_feedforward": 1024,
            "n_layers": 1

        }

    },


    "CharTrDecoder": {
        "NHEAD": 8, 
        "NUM_DECODER_LAYERS": 3, 
        "embedding_size": 512, 
        "src_vocab_size": 97, 
        "tgt_vocab_size": 249,
        "feed_forward_dim": 512,
        "dropout": 0.5
    },


    "CharacterDecoder": {

        "teaching_force_ratio": 0.5,

        "CharacterStateDecoder": {

            "EmbeddingLayer": {
    
                "vocab_len": 256,
                "embedding_dim": 256
    
            },
    
            "LSTM": {
    
                "vocab_len": 256,
                "embedding_dim": 256,
                "hidden_dim": 256,
                "n_layers": 1,
                "dropout": 0.5,
    
            },
            
            "Linear(Softmax)": {
                "hidden_dim": 256,
                "target_dim": 249
            }
    
        }

    },

    "TransformerLanguageModel": {

        "embedding_size": 256,
        "dropout": 0.5,

        "tgt_vocab_size": 111463,

        "nhead": 8,
        "hidden_dim": 512,
        "dim_feedforward": 1,
        "n_layers": 8

    }
}