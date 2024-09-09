import sys
sys.path.append('../')

from typing import List, Dict, Any, Union, Tuple, Optional
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer 
from transformers import LlamaForCausalLM, Qwen2ForCausalLM, MistralForCausalLM
from dataclasses import dataclass

from onegen.config import DataConfig, TrainingConfig, OneGenConfig, SpecialTokenConfig
from onegen.util import EnumContrastiveLoss, _print
from onegen.tokenizer import Tokenizer
from torch.nn import CrossEntropyLoss

class CausalLMOutputWithPast:
    def __init__(self, loss, logits, past_key_values, hidden_states, attentions):
        self.loss = loss
        self.logits = logits
        self.past_key_values = past_key_values
        self.hidden_states = hidden_states
        self.attentions = attentions
    
    def __contains__(self, key):
        return hasattr(self, key)

def create_onegen_model_class(cls):

    assert cls in [LlamaForCausalLM, Qwen2ForCausalLM, MistralForCausalLM], \
        "We now only test the following model class:\n1. LlamaForCausalLM\n2. Qwen2ForCausalLM\n3. MistralForCausalLM\n"+\
        "If you want to use the new class, please add this class in the `src/model/model.py`"

    class OneGenModel(cls):
        """
        This class is only used at training time.
        AutoModelForCausalLM is still a prefered class at inference time.
        """
        def __init__(self, config):
            super().__init__(config)
            self.train_mode: bool = False
            self.cl_loss_mapping:dict = EnumContrastiveLoss.get_loss_mapping()

        # must execute manually
        def load_train_config(
            self, 
            onegen_config:OneGenConfig,
        ):
            self.train_mode = True
            self.onegen_config:OneGenConfig = onegen_config
            assert self.onegen_config.loss_type in self.cl_loss_mapping, \
                f"The current loss `{self.onegen_config.loss_type}` is not supported."

        # must execute manually
        def resize_and_initialize(
            self, 
            tokenizer:Tokenizer,
            special_token_config:SpecialTokenConfig,
        ):
            # Expand the token embedding and lm_head
            _print(f"before.embedding.shape={self.model.embed_tokens.weight.shape}")
            _print(f"before.lm_head.shape={self.lm_head.weight.shape}")
            self.resize_token_embeddings(len(tokenizer))
            _print(f"now.embedding.shape={self.model.embed_tokens.weight.shape}")
            _print(f"now.lm_head.shape={self.lm_head.weight.shape}")

            # Initialize
            with torch.no_grad():
                for idx, token in enumerate(reversed(special_token_config.get_all_tokens()), start=1):
                    description = special_token_config.get_desciption(token)
                    tokenized = tokenizer.tokenize(description)
                    tokenized_ids = tokenizer.convert_tokens_to_ids(tokenized)

                    # embedding layer
                    new_embedding = self.model.embed_tokens.weight[tokenized_ids].mean(axis=0)
                    self.model.embed_tokens.weight[-idx, :] = new_embedding.clone().detach().requires_grad_(True)

                    # lm_head layer
                    last_embedding = self.lm_head.weight[tokenized_ids].mean(axis=0)
                    self.lm_head.weight[-idx, :] = last_embedding.clone().detach().requires_grad_(True)

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values = None, # Optional[Union[Cache, List[torch.FloatTensor]]]
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            # ========onegen add========
            embedding_index: Tuple[torch.LongTensor] = None,
            embedding_index_split_flag: List = None,
            # ========onegen add========
            **kwargs
        ):
            """
            embedding_index:
                Tuple(row_index: torch.LongTensor, column_index: torch.LongTensor)
                    which tokens' output will be used for represenation
            
            embedding_index_split_flag:
                how many example for generation, anchor, positive, and negative.
                [len(anchor), len(positive), len(negative)]
                    - [len(anchor_info['row_index']), len(positive_info['row_index'], len(negative_info['row_index']))]
                
                make sure that all index in embedding_index must be continous.
                we handle it by `start:end` to ditinguash anchor embedding, positive embedding, and negative embedding.
                by the way, the positive for first anchor embedding is the first.
            """
            # Prepare
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            # Forward without lm_head
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )
            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            logits = logits.float()

            loss = None
            # Generation Loss
            if labels is not None:
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Ensure tensors are on the same device
                shift_labels = shift_labels.to(shift_logits.device)
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(shift_logits, shift_labels)
            npt_loss = loss.item() if loss != None else 0.
            
            # Contrastive Loss
            cl_loss = 0.
            if loss == None:
                loss = 0.
            if self.train_mode and self.onegen_config.lambda_r != 0:
                if sum(embedding_index_split_flag[1:]) == 0:
                    loss = loss
                    assert embedding_index==None or len(embedding_index[0]) == 0
                else:
                    contrastive_loss = self.cl_loss_mapping[self.onegen_config.loss_type](
                        hidden_states, self.onegen_config, embedding_index, embedding_index_split_flag
                    )
                    cl_loss = contrastive_loss.item()
                    loss = loss * self.onegen_config.lambda_g + contrastive_loss * self.onegen_config.lambda_r
            
            if not return_dict:
                output = (logits,) + outputs[1:]
                return (loss,) + output if loss is not None else output
            
            if self.train_mode:
                return dict(
                    loss=loss,
                    logits=logits,
                    past_key_values=outputs.past_key_values,
                    hidden_states=outputs.hidden_states,
                    attentions=outputs.attentions,
                    cl_loss=cl_loss,
                    npt_loss=npt_loss
                )
            else:
                return CausalLMOutputWithPast(
                    loss=loss,
                    logits=logits,
                    past_key_values=outputs.past_key_values,
                    hidden_states=outputs.hidden_states,
                    attentions=outputs.attentions,
                )

        @torch.no_grad()
        def get_embedding(
            self,
            input_ids,
            embedding_index: List
        ):
            outputs = self.model(input_ids=input_ids)
            hidden_states = outputs[0]
            row_index, column_index = embedding_index
            row_index = row_index.to(hidden_states.device)
            column_index = column_index.to(hidden_states.device)
            all_mention_representation = hidden_states[row_index, column_index] # [bs, dim]
            return all_mention_representation

        # @classmethod
        # def from_pretrained(cls, *args, **kwargs):
        #     model = super(OneGenModel, cls).from_pretrained(*args, **kwargs)
        #     model.__class__ = cls
        #     model.train_mode: bool = False
        #     model.cl_loss_mapping:dict = EnumContrastiveLoss.get_loss_mapping()
        #     return model

    return OneGenModel