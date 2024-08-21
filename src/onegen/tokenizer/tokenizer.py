import sys
sys.path.append('../')
from transformers import AutoTokenizer
from typing import List
from util import _print

class Tokenizer:
    """
    method:
        - add_special_token
        - convert_tokens_to_ids
    """
    def __init__(
        self, 
        tokenizer_path: str,
        special_token_list: List[str]=None,
        bos_token:str = None,
        eos_token:str = None,
    ):
        self.tokenizer:AutoTokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if special_token_list != None:
            self.add_special_token(special_token_list)
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.bos_token_id = None if self.bos_token == None else self.tokenizer.convert_tokens_to_ids(self.bos_token)
        self.eos_token_id = None if self.eos_token == None else self.tokenizer.convert_tokens_to_ids(self.eos_token)
        
    def add_special_token(self, special_token_list: List[str]):
        _print("expanding tokenizer ...")
        num_added_tokens = self.tokenizer.add_tokens(special_token_list, special_tokens=True)
        assert num_added_tokens == len(special_token_list)
        _print(f"{num_added_tokens} tokens have been added including {special_token_list}.")
        return self.tokenizer
    
    
