from typing import Dict, List
from copy import deepcopy

def padding_item(item:Dict, padding_side:str, label_padding_id:int, input_padding_id:int, max_length:int) -> Dict:
    """
    padding manually.
    args:
        - item: Dict
            from tokenizer.tokenize()
        - padding_side: str
            - left
            - right
        - label_padding_id: int
        - input_padding_id: int
        - max_length: int
    """
    for key in ['input_ids', 'labels']:
        assert key in item
    assert padding_side in ['left', 'right'], \
        f"Current padding_side `{padding_side}` is invalid."
    copy_item = deepcopy(item)
    
    remainder_for_label = [label_padding_id] * (max_length - len(copy_item['labels'])) 
    remainder_for_input = [input_padding_id] * (max_length - len(copy_item['input_ids'])) 
            
    if padding_side == 'left':
        copy_item['input_ids'] = remainder_for_input + copy_item['input_ids']
        copy_item['labels'] = remainder_for_label + copy_item['labels']
    else:
        copy_item['input_ids'] = copy_item['input_ids'] + remainder_for_input
        copy_item['labels'] = copy_item['labels'] + remainder_for_label
    return copy_item
