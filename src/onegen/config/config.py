
from dataclasses import dataclass, field
from typing import List
from .util import EnumContrastiveLoss, _print
from tokenizer import Tokenizer
# from templator import Templator

# training config
@dataclass
class TrainingConfig:
    gradient_checkpointing: bool
    learning_rate: float
    optimizer: str
    save_steps: int
    report_to: str
    save_total_limit: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    epochs: int
    bf16: bool
    fp16: bool
    output_dir: str

# db/train config
@dataclass
class DataConfig:
    file_path: str
    cache_file_path: str
    mask_token_from_to: List
    repr_token: List
    max_length: int
    templator


class SpecialTokenConfig:
    def __init__(
        self,
        ctx_token_list:List[str],
        gen_token_list:List[str],
        ret_token_list:List[str],
        tokenizer: Tokenizer = None,
    ):
        self.ctx_token_list: List[str] = ctx_token_list
        self.gen_token_list: List[str] = gen_token_list
        self.ret_token_list: List[str] = ret_token_list

        self.ctx_token_id_list: List[int] = None
        self.gen_token_id_list: List[int] = None
        self.ret_toekn_id_list: List[int] = None

        self.tokenizer = tokenizer
        if self.tokenizer != None:
            self._update_token_id()
    
    def update_tokenizer(self, tokenizer:Tokenizer):
        self.tokenizer:Tokenizer = tokenizer
    
    def _update_token_id(self):
        for token_list in [self.ctx_token_list, self.gen_token_list, self.ret_token_list]:
            for token_id_list in [self.ctx_token_id_list, self.gen_token_id_list, self.ret_toekn_id_list]:
                token_id_list:List[int] = []
                for token in token_list:
                    token_id_list.append(self.tokenizer.convert_tokens_to_ids(token))

        

# onegen config
@dataclass
class OneGenConfig:
    loss_type:str
    info_nce_temperature: float
    n_pos_per_sent: int
    n_neg_per_pos: int
    lambda_r: float
    lambda_g: float

    def __post_init__(self):
        assert self.loss_type in EnumContrastiveLoss.to_list(),  f"`{self.loss_type}` is not supported. The supported loss functions are in the list `{EnumContrastiveLoss.to_list()}`"
        _print(str(self))
    
    def __repr__(self):
        return str(self)
    
    def __str__(self):
        common_output = f"""Generation Loss Weight: {self.lambda_g}\nRetrieval Loss Weight: {self.lambda_r}\nN positive per sentence: {self.n_pos_per_sent}\nN negative per positive: {self.n_neg_per_pos}"""
        if self.loss_type == EnumContrastiveLoss.InfoNCE:
            common_output = f"""Loss Type: {self.loss_type}\nTemperature: {self.info_nce_temperature}\n{common_output}"""
        elif self.loss_type == EnumContrastiveLoss.BPR:
            common_output = f"""Loss Type: {self.loss_type}\n{common_output}"""
        else:
            assert False, f"`{self.loss_type}` is not supported. The supported loss functions are in the list `{EnumContrastiveLoss.to_list()}`"
        return common_output