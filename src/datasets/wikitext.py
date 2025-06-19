from typing import Any, Dict, List, Literal, Tuple

import torch
from torch.utils.data import Dataset
import tiktoken
from torchtune.data._messages import Message
from torchtune.datasets import wikitext_dataset
from torchtune.modules.transforms.tokenizers import ModelTokenizer

class GPT2Tokenizer(ModelTokenizer):
    def __init__(self, block_size: int = 1024):
        self.enc = tiktoken.get_encoding('gpt2')
        self.max_length = block_size
    
    def tokenize_messages(self, messages):
        tokenized = []
        for message in messages:
            input_ids = self.enc.encode(message)[:self.max_length]
            tokenized.append({'input_ids': input_ids})
        return tokenized

class WikiText(Dataset):
    def __init__(self,
        split: Literal['train', 'validation', 'test'],
        version: Literal['2', '103'],
        block_size: int = 1024
    ):
        super().__init__()
        self.split = split
        self.version = version
        self.block_size = block_size

        self.data = wikitext_dataset(
            tokenizer=GPT2Tokenizer(self.block_size),
            source='EleutherAI/wikitext_document_level',
            subset=f'wikitext-{self.version}-raw-v1',
            max_seq_len=self.block_size,
            packed=False,
            split=self.split
        )