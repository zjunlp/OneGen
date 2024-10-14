import sys
sys.path.append('../')
from dataclasses import dataclass, field
from typing import List
from onegen.util import EnumContrastiveLoss, _print, FileReader
from onegen.tokenizer import Tokenizer
from onegen.templator import *
from typing import List, Dict, Tuple, Any

@dataclass
class TrainingConfig:
    gradient_checkpointing: bool
    num_train_epochs: int
    learning_rate: float
    bf16: bool
    logging_steps:int
    do_eval:bool
    optim: str
    save_steps: int
    output_dir: str
    load_best_model_at_end: bool
    deepspeed: str
    save_total_limit: int
    report_to: str
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    fp16: bool
    save_only_model: bool

    def to_dict(self):
        return vars(self)

# db/train config
@dataclass
class DataConfig:
    file_path: str
    cache_file_path: str
    mask_token_from_to: List
    repr_token: List
    max_length: int
    templator: Any
    hf_path: Dict

    def __post_init__(self):
        if isinstance(self.mask_token_from_to, list):
            if len(self.mask_token_from_to) == 0:
                self.mask_token_from_to = None

class PaddingConfig:
    def __init__(
        self,
        padding_side:str,
        padding_max_length:int,
        padding_label_id:int,
        padding_input_id:int
    ):
        self.padding_side: str = padding_side
        self.padding_max_length: int = padding_max_length
        self.padding_label_id: int = padding_label_id
        self.padding_input_id: int = padding_input_id

class SpecialTokenConfig:
    def __init__(
        self,
        ctx_token_dict:Dict[str, str],
        gen_token_dict:Dict[str, str],
        ret_token_dict:Dict[str, str],
        tokenizer: Tokenizer = None,
    ):
        self.ctx_token_dict: Dict[str, str] = ctx_token_dict
        self.gen_token_dict: Dict[str, str] = gen_token_dict
        self.ret_token_dict: Dict[str, str] = ret_token_dict

        self.description_dict: Dict[str, str] = dict()
        for _dict in [ctx_token_dict, gen_token_dict, ret_token_dict]:
            self.description_dict.update(_dict)
        assert len(self.description_dict) == \
            len(self.ctx_token_dict) + len(self.gen_token_dict) + len(self.ret_token_dict)
        
        self.ctx_token_list: List[str] = list(ctx_token_dict.keys())
        self.gen_token_list: List[str] = list(gen_token_dict.keys())
        self.ret_token_list: List[str] = list(ret_token_dict.keys())

        self.ctx_token_id_list: List[int] = None
        self.gen_token_id_list: List[int] = None
        self.ret_token_id_list: List[int] = None
        
        self.tokenizer = tokenizer
        if self.tokenizer != None:
            self._update_token_id()
    
    def update_tokenizer(self, tokenizer:Tokenizer):
        self.tokenizer:Tokenizer = tokenizer
        self._update_token_id()
    
    def _update_token_id(self):
        for token_list in [self.ctx_token_list, self.gen_token_list, self.ret_token_list]:
            for token_id_list in [self.ctx_token_id_list, self.gen_token_id_list, self.ret_token_id_list]:
                token_id_list:List[int] = []
                for token in token_list:
                    token_id_list.append(self.tokenizer.convert_tokens_to_ids(token))
    
    def get_desciption(self, token:str) -> str:
        assert token in self.description_dict
        return self.description_dict[token]
    
    def get_all_tokens_id(self) -> List[int]:
        return [self.tokenizer.convert_tokens_to_ids(token) for token in self.get_all_tokens]
    
    def get_all_tokens(self) -> List[str]:
        return list(self.description_dict.keys())
        
@dataclass
class OneGenConfig:
    loss_type:str
    info_nce_temperature: float
    n_pos_per_sent: int
    n_neg_per_pos: int
    lambda_r: float
    lambda_g: float
    model_path: str
    tokenizer_path: str
    model_type: str
    model_class: str

    def __post_init__(self):
        assert self.loss_type in EnumContrastiveLoss.to_list(),  f"`{self.loss_type}` is not supported. The supported loss functions are in the list `{EnumContrastiveLoss.to_list()}`"
        if self.tokenizer_path == None:
            self.tokenizer_path = model_path
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

def parse_workflow(file_name:str) -> Tuple[TrainingConfig, DataConfig, DataConfig, PaddingConfig, SpecialTokenConfig, OneGenConfig, str]:
    data:dict = FileReader.read_json(file_name=file_name)

    resume_checkpoint_path = data['resume_checkpoint_path']
    max_length = data['info-model']['max_length']
    training_config = TrainingConfig(**data['train'])

    data_db_config = DataConfig(
        file_path=data['info-data-db']['file_path'],
        cache_file_path=data['info-data-db']['cache_file_path'],
        mask_token_from_to=data['info-data-db']['mask_token_from_to'],
        repr_token=data['info-data-db']['repr_token'],
        max_length=max_length,
        templator=eval(data['info-data-db']['templator']),
        hf_path=data['info-data-db']['hf_path']
    )

    data_train_config = DataConfig(
        file_path=data['info-data-train']['file_path'],
        cache_file_path=data['info-data-train']['cache_file_path'],
        mask_token_from_to=data['info-data-train']['mask_token_from_to'],
        repr_token=data['info-data-train']['repr_token'],
        max_length=max_length,
        templator=eval(data['info-data-train']['templator']),
        hf_path=data['info-data-train']['hf_path']
    )

    padding_config = PaddingConfig(
        padding_side=data['info-model']['padding_side'],
        padding_max_length=max_length,
        padding_label_id=data['info-model']['padding_label_id'],
        padding_input_id=data['info-model']['padding_input_id']
    )

    special_token_config = SpecialTokenConfig(
        ctx_token_dict=data['special_token_list']['CTX'],
        gen_token_dict=data['special_token_list']['GEN'],
        ret_token_dict=data['special_token_list']['RET'],
        tokenizer=None
    )

    onegen_config = OneGenConfig(
        loss_type=data['onegen']['loss_type'],
        info_nce_temperature=data['onegen']['info_nce_temperature'],
        n_pos_per_sent=data['onegen']['n_pos_per_sent'],
        n_neg_per_pos=data['onegen']['n_neg_per_pos'],
        lambda_r=data['onegen']['lambda_r'],
        lambda_g=data['onegen']['lambda_g'],
        model_path=data['info-model']['model_path'],
        tokenizer_path=data['info-model']['tokenizer_path'],
        model_type=data['info-model']['model_type'],
        model_class=data['info-model']['model_class']
    )

    return training_config, data_train_config, data_db_config, padding_config, special_token_config, onegen_config, resume_checkpoint_path