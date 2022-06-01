## Joint Learning of Morphology and Context with Transformers

## Requirements

- python3.6 >
- torch==1.11.0
- torchtext=0.8.0
- tqdm



## Installation

You need to first create a virtual environment and import necessary libraries using the terminal commands below

```bash
python3 -m venv morphology_venv
source morphology_venv/bin/activate
pip install -r requiremetts.txt
```



## Usage 

If you don't have the training data you can either create one from scratch using `DataLoader.py` module with the following command

```bash
python3 DataLoader.py -d <train/test> 
					  -b <batch_size>
                      -p <path/to/datafile>
```

After your training/test data is generated then you can train the model. The training is done with the `trainer.py` module. Usage is given below

```bash
python3 trainer.py -t <encoderType (rnn/tr)> # rnn for LSTM-based encoder-decoder and tr Transformer-based encoder-decoder
				   -e <number_of_epochs>
				   -p <path/to/checkpoints> 
```

