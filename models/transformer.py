import torch
import torch.nn as nn
import math

class EncoderBlock(nn.Module):
    """Combines the self attention and FFNN to form the encoder block"""
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        # create the self attention and ffnn layers
        self.attention = MultiHeadSelfAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)

        # create the normalization layers
        self.normal_one = nn.LayerNorm(d_model)
        self.normal_two = nn.LayerNorm(d_model)

    def forward(self, x):
        # calculate self attention output and normalize
        self_attention = self.attention(x)
        x = self.normal_one(x + self_attention)

        # calculate feed forward output and normalize
        ffnn_output = self.feed_forward(x)
        x = self.normal_two(x + ffnn_output)

        # return the output
        return x

class MultiHeadSelfAttention(nn.Module):
    """Implements the multi-head self attention mechanism"""

    def __init__(self, d_model, num_heads):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_key = d_model / num_heads

        self.query_mat = nn.Linear(d_model, d_model)
        self.key_mat = nn.Linear(d_model, d_model)
        self.value_mat = nn.Linear(d_model, d_model)
    
    def calculateAttention(self, x):
        batch_size, seq_len = x.shape

        # create the query, key, and value matrices
        Q = self.query_mat(x)
        K = self.key_mat(x)
        V = self.value_mat(x)

        # reshape to account for multi heads
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_key)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_key)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_key)

        # swap the sequence length and number of head dimensions (we want num heads to be outside of sequence length)
        Q = Q.transpose(2, 1)
        K = K.tranpose(2, 1)
        V = V.transpose(2, 1)


        # performs self attention score calculation
        output = torch.softmax((Q @ K.transpose(3, 2)) / math.sqrt(self.d_key), dim = 3) @ V

        return output
    

class FeedForward(nn.Module):
    """Implements the feedforward neural network step"""

    def __init__(self, d_model, d_ff):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff

        # create the linear transformations for each layer
        self.lin_one = nn.Linear(d_model, d_ff)
        self.lin_two = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        # pass through each linear transformation and ReLU layer
        x = self.lin_one(x)
        x = torch.relu(x)
        x = self.lin_two(x)

        # return the result
        return x