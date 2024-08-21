from typing import List, Dict, Any, Union, Tuple, Optional
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclass import dataclass

from .config import DataConfig, TrainingConfig, OneGenConfig
from .util import EnumContrastiveLoss

class CausalLMOutputWithPast:
    def __init__(self, loss, logits, past_key_values, hidden_states, attentions):
        self.loss = loss
        self.logits = logits
        self.past_key_values = past_key_values
        self.hidden_states = hidden_states
        self.attentions = attentions
    
    def __contains__(self, key):
        return hasattr(self, key)

class OneGenModel(AutoModelForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.train_mode: bool = False
        self.cl_loss_mapping:dict = EnumContrastiveLoss.get_loss_mapping()

    