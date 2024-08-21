
from dataclasses import dataclass
from typing import List
from .util import EnumContrastiveLoss, _print

# training config
@dataclass
class TrainingConfig:
    pass

# db/train config
@dataclass
class DataConfig:
    file_path:str
    special_token_id_list: List
    mode: str
    overlength_last_token_id_list: List
    mask_token_id_flag_from_to: List

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