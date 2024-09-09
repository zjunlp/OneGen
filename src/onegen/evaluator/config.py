import sys
sys.path.append('../')
from onegen.templator import Templator
from onegen.util import FileReader, _print, sim_matrix, FileWriter
from onegen.util import DEFAULT_GENERATION_CONFIG, MAX_NEW_TOKENS, MAX_RETRIEVAL_CNT
from onegen.dataset import padding_input_ids

from dataclasses import dataclass
from typing import Dict, Tuple, Any, Union, List, Optional
from transformers import GenerationConfig
from transformers import AutoModel, AutoModelForCausalLM
import torch

@dataclass
class InferenceConfig:
    max_new_tokens:int 
    generation_config:GenerationConfig
    embed_batch_size:int
    sentence_connector:str
    max_retrieval_cnt:int

    # only for database
    skip_repr_token_cnt:int


    def __post_init__(self):
        if self.max_new_tokens == None:
            self.max_new_tokens:int = MAX_NEW_TOKENS
        if self.generation_config == None:
            self.generation_config:GenerationConfig = DEFAULT_GENERATION_CONFIG
        if self.embed_batch_size == None:
            self.embed_batch_size:int = 16
        if self.sentence_connector == None:
            self.sentence_connector:str = ""
        if self.max_retrieval_cnt == None:
            self.max_retrieval_cnt:int = MAX_RETRIEVAL_CNT
        if self.skip_repr_token_cnt == None:
            self.skip_repr_token_cnt:int = 0

@dataclass
class FileConfig:
    test_file_path: str
    db_file_path: str
    db_cache_embedding_path: str
    test_templator: Templator
    db_templator: Templator

    def __post_init__(self):
        self.db_embedding:torch.Tensor = None
        self.db_label:List = None
        self.online_retrieval: bool = False
        self.pre_cache: bool = False

        if self.db_cache_embedding_path == None:
            self.db_cache_embedding_path = ""
        
        assert FileReader.is_existed(self.test_file_path)
        if self.db_file_path != None:
            assert FileReader.is_existed(self.db_file_path)

        if FileReader.is_existed(self.db_cache_embedding_path):
            # use cached embedding
            _print(f"loading embedding from the cached file `{self.db_cache_embedding_path}`")
            self.db_embedding, self.db_label = FileReader.read_pickle(self.db_cache_embedding_path)

        if self.db_embedding == None and not FileReader.is_existed(self.db_file_path):
            # online retrieve
            self.online_retrieval = True
        elif self.db_embedding == None and FileReader.is_existed(self.db_file_path):
            # need offline embedding
            self.pre_cache = True
        
        # need online retrieval / need pre_cache / loading the cached embedding
        assert self.online_retrieval or self.pre_cache or self.db_embedding != None

@dataclass
class ComponentConfig:
    model_class:Any
    model_path:str
    tokenizer_path:str
    # [torch.float16, torch.bfloat16, torch.float32]
    torch_dtype:Any
    special_token_list:List[str]
    padding_side:str
    padding_token:str
    # the following variables are used for generator
    stop_token_list:List[str]
    repr_token_list:List[str]
    # this variable is only used for retrieval then generation task
    concatenate_template:str = "{history}{document}"
    add_prefix_space: bool = False
    add_eos_token:bool = False
    add_bos_token:bool = False

    def __post_init__(self):
        assert self.model_class in [AutoModel, AutoModelForCausalLM]
        assert FileReader.is_existed(self.model_path)
        if self.tokenizer_path == None:
            self.tokenizer_path = self.model_path
        assert FileReader.is_existed(self.tokenizer_path)
        if self.special_token_list != None and len(self.special_token_list) == 0:
            self.special_token_list = None
        if self.stop_token_list == None:
            self.stop_token_list = []
        if self.repr_token_list == None:
            self.repr_token_list = []
        # this value will be updated in the method `__init__` of the class `Backend` automatically.
        self.stop_token_id_list:List[int] = []
        self.repr_token_id_list:List[int] = []
        assert self.padding_side in ['left', 'right']
        self.padding_token_id:int = None
        assert "{history}" in self.concatenate_template
        assert "{document}" in self.concatenate_template

    def __str__(self) -> str:
        return f"model_class: {self.model_class}\nmodel_path: {self.model_path}\ntokenizer_path: {self.tokenizer_path}\ntorch_dtype: {self.torch_dtype}\nspecial_token_list: {self.special_token_list}" + \
            f"add_prefix_space: {self.add_prefix_space}\nadd_eos_token: {self.add_eos_token}\nadd_bos_token: {self.add_bos_token}\nstop_token_list: {self.stop_token_list}\nrepr_token_list: {self.repr_token_list}\n" + \
            f"padding_side: {self.padding_side}"
