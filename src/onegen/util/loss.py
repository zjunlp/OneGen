# contrastive loss
import torch
from typing import List, Tuple

def info_nce_loss(
    temperature:float, 
    hidden_states, 
    embedding_index: Tuple[torch.LongTensor, torch.LongTensor], 
    embedding_index_split_flag: List
):
    """
    embedding_index:
        Tuple(row_index: torch.LongTensor, column_index: torch.LongTensor)
            which token's output will be used for representation
    
    embedding_index_split_flag:
        how many example for generation, anchor, positive, and negative
    """

    if embedding_index is None:
        return 0.
    assert sum(embedding_index_split_flag) == len(embedding_index[0])
    raise NotImplementedError()

def bpr_loss(
    temperature: float,
    hidden_states,
    embedding_index: Tuple[torch.LongTensor, torch.LongTensor],
    embedding_index_split_flag: List
):
    raise NotImplementedError()