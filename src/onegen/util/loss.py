# contrastive loss
import torch
import torch.nn as nn
from typing import List, Tuple
import torch.nn.functional as F

def info_nce_loss(
    hidden_states, 
    onegen_config,
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
    
    # Step 1. Get the anchor, positive and negative respectively
    row_index, column_index = embedding_index
    row_index = row_index.to(hidden_state.device)
    column_index = column_index.to(hidden_state.device)
    all_mention_representation = hidden_state[row_index, column_index]
    assert len(row_index) == sum(embedding_index_split_flag)
    # anchors = [0, sum(embedding_index_split_flag[0:1]), sum(embedding_index_split_flag[0:2]), sum(embedding_index_split_flag[0:3])]
    # anchors = [0, sum(embedding_index_split_flag[0:1]), sum(embedding_index_split_flag[0:2]), sum(embedding_index_split_flag[0:3]), sum(embedding_index_split_flag[0:4])][1:]
    anchors = [0, sum(embedding_index_split_flag[0:1]), sum(embedding_index_split_flag[0:2]), sum(embedding_index_split_flag[0:3])]
    anchor_mention_representation = all_mention_representation[anchors[0]:anchors[1]]
    description_mention_positive_representation = all_mention_representation[anchors[1]:anchors[2]]
    description_mention_negative_representation = all_mention_representation[anchors[2]:anchors[3]]
    assert len(anchor_mention_representation) == len(description_mention_positive_representation)
    assert len(anchor_mention_representation) * onegen_config.n_neg_per_pos == len(description_mention_negative_representation)

    # Step 2. Concate postive and negative
    description_mention_representation = torch.cat(
        (description_mention_positive_representation,description_mention_negative_representation), dim=0
    ) 

    # Step 3. Calculate similarity and loss
    similarity_scores = sim_matrix(anchor_mention_representation, description_mention_representation)
    logits = similarity_scores / onegen_config.info_nce_temperature
    N = anchor_mention_representation.size(0)
    labels = torch.arange(N).to(logits.device)
    loss = F.cross_entropy(logits, labels)
    return loss

def bpr_loss(
    hidden_states_unnorm,
    onegen_config,
    embedding_index: Tuple[torch.LongTensor, torch.LongTensor],
    embedding_index_split_flag: List,
    eps=1e-8,
):
    if embedding_index is None or len(embedding_index) == 0:
        return 0.
    # Step 1. Normalize
    h_n = hidden_state_unnorm.norm(dim=1)[:, None]
    hidden_state = hidden_state_unnorm / torch.max(h_n, eps*torch.ones_like(h_n))

    # Step 2. Get the anchor, positive and negative respectively
    row_index, column_index = embedding_index
    row_index = row_index.to(hidden_state.device)
    column_index = column_index.to(hidden_state.device)
    all_mention_representation = hidden_state[row_index, column_index]
    assert len(row_index) == sum(embedding_index_split_flag), f"\n{row_index}\n{embedding_index_split_flag}"
    # anchors = [0, sum(embedding_index_split_flag[0:1]), sum(embedding_index_split_flag[0:2]), sum(embedding_index_split_flag[0:3])]
    anchors = [
        0, 
        sum(embedding_index_split_flag[0:1]), 
        sum(embedding_index_split_flag[0:2]), 
        sum(embedding_index_split_flag[0:3]),
    ]
    anchor_mention_representation = all_mention_representation[anchors[0]:anchors[1]]
    description_mention_positive_representation = all_mention_representation[anchors[1]:anchors[2]]
    description_mention_negative_representation = all_mention_representation[anchors[2]:anchors[3]]
    assert len(anchor_mention_representation) == len(description_mention_positive_representation)
    assert len(anchor_mention_representation) * onegen_config.n_neg_per_pos == len(description_mention_negative_representation),\
        f"{anchors}\n{len(anchor_mention_representation)} * {onegen_config.n_neg_per_pos} != {len(description_mention_negative_representation)}"

    # Step 3. Concate postive and negative
    description_mention_representation = torch.cat(
        (description_mention_positive_representation,description_mention_negative_representation), dim=0
    ) 

    # Step 4. Calculate Loss
    n_pos = anchor_mention_representation.shape[0]
    n_dim = anchor_mention_representation.shape[1]
    # [n_pos, dim] * [n_pos, dim] -> [n_pos, 1]
    positive_scores = (anchor_mention_representation * description_mention_representation[0:n_pos]).sum(-1).unsqueeze(-1)
    # [n_pos, 1, dim] * [n_pos, n_neg_per_pos, dim] -> [n_pos, n_neg_per_pos, dim] -> [n_pos, n_neg_per_pos]
    negative_scores = (anchor_mention_representation[0:n_pos].unsqueeze(dim=1) * description_mention_representation[n_pos:].view(n_pos, -1, n_dim)).sum(-1)
    return -F.logsigmoid(positive_scores - negative_scores).mean()