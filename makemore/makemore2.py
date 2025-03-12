import torch
import torch.nn as nn
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    block_size: Optional[int] = None # Length of the input sequences of integers
    vocab_size: Optional[int] = None # The input integers are in range[0, vocab_size)

    # Parameters below control the sizes of each model slightly differently
    n_layer: int = 4
    n_embd: int = 64
    n_embd2: int = 64
    n_head: int = 4

# Transformer Language Model (*exactly* as used in GPT-2)

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo
    (identical to OpenAI GPT). 
    
    Reference: Gaussian Error Linear Units (GELU) paper:
    https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

