import sys
sys.path.append('../')
from transformers import AutoTokenizer
from typing import List
from util import _print
from constant import IGNORE_LABEL_ID

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
        # add_prefix_space=False
        self.tokenizer:AutoTokenizer = AutoTokenizer.from_pretrained(tokenizer_path, add_prefix_space=False)
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

    def tokenize(
        self,
        structured_input:List[str],
        special_token_id_list_for_repr:List[int],
        max_length:int, 
        mask_token_id_from_to:List[List, int]=None,
        train_on_input:bool=False,
        check_consistent:bool=False
    ) -> Dict:
        """
        args:
            - structured_input: List[str]
                [user, model, user, model, ...]
                The reason for splitting this is that we need to mask the label for the user
            - special_token_id_list_for_repr: List[int]
                to get the `embedding_index`
            - max_length: int
            - train_on_input: bool
            - check_consistent: bool
                check `tokenizer("".join(structured_input))['input_id'] == tokenizer(segament)['input_id'] for segement in structured_input`
            - mask_token_id_from_to: List[List, int]
                e.g. [<paragraph>, </paragraph>]
                e.g. [[<p1>, <p2>], [<p3>, <p4>]]
        procedure:
            Step 1. tokenize for each segement
            Step 2. create default labels
            Step 3. adjust labels' value according to the `mask_token_id_from_to`
            Step 4. locate the `special_token_id_list_for_repr`
        """


    
    
