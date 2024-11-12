---
toc: true
categories:
- machine-learning
- exposition
date: '2024-11-09'
description: A step-by-step demonstration
title: Sequence-to-Sequence Transformer in Pytorch
---

> This post benefited from the generous help of [Liben Chen](https://sites.google.com/view/liben-chen) and [Hejia Liu](https://carlsonschool.umn.edu/faculty/hejia-liu)

The transformer architecture is the bedrock for much of today's Generative AI and Large Language Models. Despite its obvious importance, I have not been able to find a tutorial that satisfies the following criteria:

- Uses popular deep learning framework such as ```pytorch``` but doesn't require a super deep mastery of it. Andrej Karpathy's tutorial ["Let's reproduce GPT-2"](https://www.youtube.com/watch?v=l8pRSuU81PU) is perhaps one of the best resources online (and will easily be several magnitudes better than what I can come up with), but it requires a pretty good understanding of ```pytorch``` to fully grasp.
- Abstracts away "peripheral" issues such as tokenization / padding / masking and focuses on the core transformer components. Many tutorials I can find (including the ones on ```pytorch``` official website) spend way too much energy just to prepare the datasets (which can be frustrating for beginners).
- Provides a step-by-step / exhuastive explanation of what happens. By this, I include explanations on seemingly trivial things such as the shape of tensors at each step (which can be hard to keep track for beginners).

This blog post is an attempt to fill in this gap. It takes a simple (artificial) learning task and implements a transformer model with standard ```pytorch``` classes, and explains every step along the way.

The following python packages / modules will be used:

```python
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Transformer
```

## Task and Synthetic Data Generation
The task is to predict a numerical output based on a number of input features. Treating a number as a sequence of digits (where each digit is a token), then this task is essentially a sequence-to-sequence prediction task. Even though such "numeric prediction" task is typically not where transformers are applied, it does offer a few advantages as a tutorial / demonstration: 

- The vocabulary is very restricted (all 10 single digits + blank space);
- Each input can be represented as a fixed length sequence, thereby removing the need for padding / masking (except for a specific masking at decoder -- will be discussed later).

As the first step, let's simulate the data used for training and evaluation:

- $X_1$, ..., $X_{10}$: 10 numerical input features, each randomly sampled from a uniform distribution.
- $Y = \frac{1}{10} \sum_i X_i$: the numerical output is simply the average value.
- $N=5000$: 5000 samples, 4000 for training and 1000 for evaluation.

```python
# set random seed for reproducibility
np.random.seed(123)
X = np.random.uniform(size = (5000, 10))
Y = np.mean(X, axis = 1)
X_train = X[:4000]
X_test = X[4000:]
Y_train = Y[:4000]
Y_test = Y[4000:]
#print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
```

## Constructing Vocabulary and Datasets
Importantly, ```pytorch``` does not take these raw values / arrays as input. We need to tokenize them and convert them into indices in the vocabulary.

```python
# vocab has single-digits, space, start, end
VOCAB = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ' ', 's', 'e']
# for simplicity, we restrict each input/output number to 8 digits
MAX_DIGITS = 8
```

With such a restricted vocabulary, tokenizing each number is the same as splitting it into a sequence of single digits. Note that, because both inputs and outputs take value between 0 and 1, every number starts with "0." (followed by 8 decimal digits). Therefore, as a further simplification, we don't need to keep track of the "0." for each number.

The following ```CustomDataset``` class performs basic processing and tokenization of input features and output values. It will allow us to convert the raw numpy arrays ```X``` and ```Y``` into a format that can be ingested by the transformer model.

```python
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, vocab):
        self.X = X
        self.Y = Y
        self.vocab = vocab
        # the "index" method is defined below
        self.X_indexed = self.index(X, 'X')
        self.Y_indexed = self.index(Y, 'Y')

    # The "index" method converts either an input vector or an output value to a sequence of token indices
    def index(self, data, type):
        data_indexed = []
        for row in data:
            if type == 'Y':
                # in this case, row is a scalar, we convert it to a string and remove the "0." prefix
                # the '{:8f}'.format(...) part ensures the number has 8 digits after the decimal point, and converts it to a string
                # the '[2:]' part removes the "0." prefix
                row_str = '{:.8f}'.format(row)[2:]
            if type == 'X':
                # in this case, we do the same processing to each feature value, then concatenate them to a longer sequence, separated by blank spaces
                row_str = ' '.join(['{:.8f}'.format(x)[2:] for x in row])
            # also need to prepend 's' and append 'e' to the sequence
            row_str = 's' + row_str + 'e'
            # convert to indices in vocabulary
            row_idx = [self.vocab.index(c) for c in row_str]
            data_indexed.append(row_idx)
        return np.array(data_indexed)

    def __len__(self):
        # this is a required method in custom dataset classes, it should return size of data (i.e., number of rows)
        return len(self.X_indexed)

    def __getitem__(self, idx):
        # this is also a required method, it should return the item at the given index
        src = torch.tensor(self.X_indexed[idx], dtype=torch.long)
        tgt = torch.tensor(self.Y_indexed[idx], dtype=torch.long)
        return src, tgt
```

Now, we can create the datasets that can be used for training and validation:

```python
train_dataset = CustomDataset(X_train, Y_train, VOCAB)
test_dataset = CustomDataset(X_test, Y_test, VOCAB)
#print(len(train_dataset), len(test_dataset))
```

You can also print out the first data point to see what it looks like
```python
print("raw inputs:", X_train[0])
print("raw output:", Y_train[0])
print("tokenized input sequence:", train_dataset[0][0])
print("tokenized output sequence:", train_dataset[0][1])
```

## Constructing the Transformer Model
We are now ready to construct the transformer model. This include several modules:
- A ```TokenEmbedding``` class that projects each token to its (trainable) embedding representation;
- A ```PositionalEncoding``` class that adds the positional encoding to the token embeddings;
- A ```Seq2SeqTransformer``` that implements the actual transformer architecture.

We will do them one at a time.

```python
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        """
        :param vocab_size: the size of the vocabulary
        :param d_model: the embedding dimension
        """
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, tokens):
        """
        :param tokens: the input tensor with shape (batch_size, seq_len)
        :return: the tensor after token embedding with shape (batch_size, seq_len, d_model)
        """
        return self.embedding(tokens)
```

If you apply the ```TokenEmbedding``` module to the first input sequence in the training set, you should get a tensor of shape (1, seq_len, d_model)

```python
test_input = train_dataset[0][0].unsqueeze(0)
test_emb = TokenEmbedding(len(VOCAB), 512)(test_input)
print(test_emb.size())
```

Next, the ```PositionalEncoding``` class makes use of the positional encoding approach initially proposed in [Attention Is All You Need](https://user.phil.hhu.de/~cwurm/wp-content/uploads/2020/01/7181-attention-is-all-you-need.pdf).

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        """
        :param d_model: the embedding dimension
        :param max_len: the maximum length of the sentence
        """
        super(PositionalEncoding, self).__init__()
        # setting max_len to 100 here, because the largest input sequence is 91 tokens long (10 * 8 digits + 9 spaces + 1 start + 1 end), so 100 is enough
        # intialize the positional encoding, pe.shape = (max_len, d_model)        
        pe = torch.zeros(max_len, d_model)
        # generate a tensor of shape (max_len, 1), with values from 0 to max_len - 1, to represent all unique positions
        # the unsqueeze(1) operation adds a dimension after the first dimension, so the shape changes from (max_len,) to (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # calculate scaling factors for each dimension of the positional encoding, see the formula in the first section of this notebook
        scaling_factors = torch.tensor([10000.0 ** (-2 * i / d_model) for i in range(d_model // 2)])
        # now populate the positional encoding tensor with values, even indices use sine functions, odd indices use cosine functions
        pe[:, 0::2] = torch.sin(position * scaling_factors)  # pe[:, 0::2].shape = (max_len, d_model/2)
        pe[:, 1::2] = torch.cos(position * scaling_factors)  # pe[:, 1::2].shape = (max_len, d_model/2)
        # add a batch dimension to the positional encoding tensor so that it's compatible with the input tensor. pe.shape = (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        # register the positional encoding tensor as a buffer, so that it will be stored as part of the model's "states" and won't be updated during training
        # this is desirable because we don't want the positional encoding to be trained, we want it to be fixed
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        :param x: the input tensor with shape (batch_size, seq_len, d_model)
        :return: the tensor after adding positional encoding with shape (batch_size, seq_len, d_model)
        """
        # for a given input tensor x, add the positional encoding to it
        # x.size(1) gets the second dimensions of x, which is dimension that contains the element indices in the sequence
        x = x + self.pe[:, :x.size(1)]
        return x
```

Again, you can apply this on the first input sequence to see its effect.

```python
test_emb_with_pe = PositionalEncoding(512)(test_emb)
print(test_emb_with_pe.size())
```

Next we have the actual ```Seq2SeqTransformer``` module. Things like multi-head attention, feed-foward layers, layer normalziation, and residual connections are all encapsulated in pytorch's ```Transformer``` module, which makes it very straightforward to build. For a more theoretical discussion of these transformer components, you can check out my [teaching materials](https://github.com/mochenyang/MSBA6461-Advanced-AI/blob/main/5_Transformer.ipynb) on this topic.

```python
class Seq2SeqTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, vocab_size):
        """
        :param d_model: the embedding dimension
        :param nhead: the number of heads in multi-head attention
        :param num_encoder_layers: the number of blocks in the encoder
        :param num_decoder_layers: the number of blocks in the decoder
        :param dim_feedforward: the dimension of the feedforward network
        """
        super(Seq2SeqTransformer, self).__init__()
        # note that, in many other tasks (e.g., translation), you need two different token embeddings for the source and target languages
        # here, however, because both input and output use the same vocabulary, we can use the same token embedding for both
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        # the transformer model is constructed with the Transformer module, which takes care of all the details
        # the batch_first=True argument means the input and output tensors are of shape (batch_size, seq_len, d_model)
        self.transformer = Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, batch_first=True)
        # the generator is a simple linear layer that projects the transformer output to the vocabulary size
        # it generates the logits for each token in the vocabulary, will be used for computing loss and making predictions
        self.generator = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        """
        :param src: the sequence to the encoder (required). with shape (batch_size, seq_len, d_model)
        :param tgt: the sequence to the decoder (required). with shape (batch_size, seq_len, d_model)
        :param src_mask: the additive mask for the src sequence (optional). with shape (batch_size, seq_len, seq_len)
        :param tgt_mask: the additive mask for the tgt sequence (optional). with shape (batch_size, seq_len, seq_len)
        :param src_padding_mask: the additive mask for the src sequence (optional). with shape (batch_size, 1, seq_len)
        :param tgt_padding_mask: the additive mask for the tgt sequence (optional). with shape (batch_size, 1, seq_len)
        :param memory_key_padding_mask: the additive mask for the encoder output (optional). with shape (batch_size, 1, seq_len)
        :return: the decoder output tensor with shape (batch_size, seq_len, d_model)
        """
        # separately embed the source and target sequences
        src_emb = self.positional_encoding(self.tok_emb(src))
        tgt_emb = self.positional_encoding(self.tok_emb(tgt))
        # Important: we don't need any masks for source sequence, or any padding masks, nor do we need a mask for decoder attending to the encoder
        # but we do need a mask for the target sequence -- this is a "causal mask", which prevents the decoder from attending to subsequent tokens during training
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1))
        outs = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask)
        return self.generator(outs)
    
    # The transformer also have an encode method and a decode method
    # the encode method takes the source sequence and produce the context vector (which pytorch calls "memory")
    # the decoder method takes the target sequence and the context vector, and produce the output sequence
    def encode(self, src):
        """
        :param src: the sequence to the encoder (required). with shape (batch_size, seq_len, d_model)
        :return: the encoder output tensor with shape (batch_size, seq_len, d_model)
        """
        return self.transformer.encoder(self.positional_encoding(self.tok_emb(src)))
    
    def decode(self, tgt, memory):
        """
        :param tgt: the sequence to the decoder (required). with shape (batch_size, seq_len, d_model)
        :param memory: the sequence from the last layer of the encoder (required). with shape (batch_size, seq_len, d_model)
        :return: the decoder output tensor with shape (batch_size, seq_len, d_model)
        """
        return self.transformer.decoder(self.positional_encoding(self.tok_emb(tgt)), memory)
```

## Training the Transformer Model
Training a transformer model usually starts with specifying the model's parameters, initializing the model, and choosing the optimizer as well as loss function.

```python
# specify model parameters and training parameters
VOCAB_SIZE = len(VOCAB)
EMB_SIZE = 256
NHEAD = 4
FFN_HID_DIM = 128
BATCH_SIZE = 32
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
NUM_EPOCHS = 25

# instantiate the model
model = Seq2SeqTransformer(EMB_SIZE, NHEAD, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, FFN_HID_DIM, VOCAB_SIZE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Create DataLoader for batching
# for eval_loader, we load data one at a time for better demonstration of what happens -- in practice you can also batch it
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
eval_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
```

We can now start the actual training and validation process.

```python
# training
for epoch in range(NUM_EPOCHS):
    # start model training
    model.train()
    # initialize total loss for the epoch
    total_loss = 0
    for src, tgt in train_loader:
        optimizer.zero_grad()        
        # Separate the input and target sequences for teacher forcing
        # tgt_input has everything except the last token
        # tgt_output has everything except the first token
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        # Forward pass with teacher forcing, logits has shape (batch_size, seq_len, vocab_size)
        logits = model(src, tgt_input)
        # Calculate loss. The .reshape(-1) flattens the logits to (batch_size * seq_len, vocab_size)
        outputs = logits.reshape(-1, logits.shape[-1])
        # also flatten the ground truth outputs to shape (batch_size * seq_len)
        tgt_out = tgt_output.reshape(-1)
        loss = criterion(outputs, tgt_out)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(f"Epoch: {epoch}, Training Loss: {total_loss}")
    
    # monitor loss test set
    model.eval()
    test_loss = 0      
    with torch.no_grad():
        for src, tgt in eval_loader:
            encoder_output = model.encode(src)
            # decoding starts with the "start" token
            tgt_idx = [VOCAB.index('s')]
            pred_num = '0.'
            for i in range(MAX_DIGITS):
                # prepare the input tensor for the decoder, adding the batch dimension
                decoder_input = torch.LongTensor(tgt_idx).unsqueeze(0)
                # the decoder output has shape (1, seq_len, d_model) and the last position in sequence is the prediction for next token
                decoder_output = model.decode(decoder_input, encoder_output)
                # the predicted logits has shape (1, seq_len, vocab_size)
                logits = model.generator(decoder_output)
                # calculate test loss based on most recent token prediction, that is logits[:, -1, :]
                test_loss += criterion(logits[:, -1, :], tgt[0][i].unsqueeze(0)).item()
                # the actual predicted token is the one with highest logit score
                # here, .argmax(2) makes sure the max is taken on the last dimension, which is the vocabulary dimension, and [:, -1] makes sure that we are looking at the last position in the sequence
                pred_token = logits.argmax(2)[:,-1].item()
                # append the predicted token to target sequence as you go
                tgt_idx.append(pred_token)
                pred_num += VOCAB[pred_token]
                if pred_token == VOCAB.index('e'):
                    break            
            # Convert the predicted sequence to a number - if you want, you can use it to compute other metrics such as RMSE
            try:
                pred_num = float(pred_num)  # Convert the accumulated string to a float
            except ValueError:
                pred_num = 0.0  # Handle any conversion errors gracefully
    print("Test Loss: ", test_loss)
```

Finally, it is worth noting that even though the entire architecture is relatively straightforward to set up, training it to achieve a good performance is highly non-trivial. In fact, if you take the above code and simply run it, the validation loss would be really bad. Sovling this training challenge is not easy, and would likely require a combination of having high-quality data, expertise in training deep neural nets, and a lot of computing power.