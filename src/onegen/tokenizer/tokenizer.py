import sys
sys.path.append('../')
from transformers import AutoTokenizer
from typing import List, Dict
from util import _print
from util.constant import IGNORE_LABEL_ID
from copy import deepcopy

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
        add_prefix_space:bool = False,
    ):
        self.tokenizer:AutoTokenizer = AutoTokenizer.from_pretrained(tokenizer_path, add_prefix_space=add_prefix_space)
        if special_token_list != None:
            self.add_special_token(special_token_list)
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.bos_token_id = None if self.bos_token == None else self.tokenizer.convert_tokens_to_ids(self.bos_token)
        self.eos_token_id = None if self.eos_token == None else self.tokenizer.convert_tokens_to_ids(self.eos_token)
        self.tokenizer.add_eos_token = False
        self.tokenizer.add_bos_token = False
        
    def add_special_token(self, special_token_list: List[str]):
        _print("expanding tokenizer ...")
        num_added_tokens = self.tokenizer.add_tokens(special_token_list, special_tokens=True)
        assert num_added_tokens == len(special_token_list)
        _print(f"{num_added_tokens} tokens have been added including {special_token_list}.")
        return self.tokenizer

    def tokenize(
        self,
        structured_input:List[str],
        max_length:int, 
        special_token_id_list_for_repr:List[int],
        train_on_input:bool=False,
        mask_token_id_from_to:List=None,
        check_consistency:bool=False,
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
            - check_consistency: bool
                check `tokenizer("".join(structured_input))['input_id'] == tokenizer(segement)['input_id'] for segement in structured_input`
            - mask_token_id_from_to: List[List, int]
                e.g. [<paragraph>, </paragraph>]
                e.g. [[<p1>, <p2>], [<p3>, <p4>]]
        procedure:
            Step 1. tokenize for each segement
            Step 2. create default labels
            Step 3. adjust labels' value according to the `mask_token_id_from_to`
            Step 4. truncate the labels and input_ids according to the `max_length`
            Step 5. locate the `special_token_id_list_for_repr`. here we just pay attention to the model's output part.
        """

        # Step 1. tokenize for each segement
        tokenized_input_id_list:List = []
        for segement in structured_input:
            if segement != "":
                tokenized_input_id_list.append(
                    self.tokenizer(segement, return_tensors=None, padding=False)['input_ids']
                )
            else:
                # ensure the first part in tokenized_input_id_list is from user
                tokenized_input_id_list.append(list())
        if check_consistency:
            whole_input: str = "".join(structured_input)
            tokenized_whole_input = self.tokenizer(whole_input, return_tensors=None, padding=False)['input_ids']
            tokenized_whole_input_from_segement = []
            for input_ids in tokenized_input_id_list:
                tokenized_whole_input_from_segement.extend(input_ids)
            assert tokenized_whole_input_from_segement == tokenized_whole_input, \
                f"consistency check failed.\n{tokenized_whole_input_from_segement}\n{tokenized_whole_input}"
            _print("consistency check successfully.")

        # Step 2. create default labels
        tokenized_label_list:List = deepcopy(tokenized_input_id_list)
        if not train_on_input:
            for i in range(len(tokenized_label_list)):
                if i % 2 == 0:
                    # mask!
                    tokenized_label_list[i] = [IGNORE_LABEL_ID] * len(tokenized_label_list[i])
        
        # Step 3. adjust labels' value according to the `mask_token_id_from_to`
        if mask_token_id_from_to != None:
            if isinstance(mask_token_id_from_to[0], int):
                assert len(mask_token_id_from_to) == 2
                mask_token_id_from_to:List[List[int]] = [mask_token_id_from_to]
            mask_token_id_start_list = [pair[0] for pair in mask_token_id_from_to]
            mask_token_id_end_list = [pair[1] for pair in mask_token_id_from_to]
            for i in range(len(tokenized_label_list)):
                if i % 2 == 1:
                    idx = 0
                    mask_start_id = -1
                    while idx < len(tokenized_label_list[i]):
                        if tokenized_label_list[i][idx] in mask_token_id_start_list and mask_start_id == -1:
                            # trigger!
                            locate_start = idx
                            mask_start_id = tokenized_label_list[i][idx]
                            mask_end_id = mask_token_id_end_list[mask_token_id_start_list.index(mask_start_id)]
                            while idx < len(tokenized_label_list[i]) and tokenized_label_list[i][idx] != mask_end_id:
                                idx += 1
                            assert tokenized_label_list[i][idx] == mask_end_id
                            mask_start_id = -1
                            tokenized_label_list[i][locate_start:idx+1] = [IGNORE_LABEL_ID] * len(tokenized_label_list[i][locate_start:idx+1])
                            assert len(tokenized_input_id_list[i]) == len(tokenized_label_list[i])
                        idx += 1
                    assert mask_start_id == -1

        # Step 4. truncate the labels and input_ids according to the `max_length`
        final_item:Dict = {
            "input_ids": [],
            "labels": [],
            "embedding_index": []
        }
        user_indicator:List[bool] = []
        assert len(tokenized_label_list) == len(tokenized_input_id_list)
        for idx, (labels, input_ids) in enumerate(zip(tokenized_label_list, tokenized_input_id_list)):
            if idx % 2 == 0:
                user_indicator += [True] * len(labels)
            else:
                user_indicator += [False] * len(labels)
            final_item['labels'].extend(labels)
            final_item['input_ids'].extend(input_ids)
        assert len(final_item['labels']) == len(final_item['input_ids'])
        final_item['input_ids'] = final_item['input_ids'][0:max_length]
        final_item['labels'] = final_item['labels'][0:max_length]
        user_indicator = user_indicator[0:max_length]

        # Step 5. locate the `special_token_id_list_for_repr`. here we just pay attention to the model's output part.
        if len(special_token_id_list_for_repr) != 0 or special_token_id_list_for_repr != None:
            for i in range(len(final_item['input_ids'])):
                if final_item['input_ids'][i] in special_token_id_list_for_repr and user_indicator[i] == False:
                    if i + 1 < len(final_item['labels']):
                        final_item['labels'][i+1] = IGNORE_LABEL_ID 
                    else:
                        assert final_item['input_ids'][-1] in special_token_id_list_for_repr
                    final_item['embedding_index'].append(i)
        return final_item

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

if __name__ == '__main__':
    tokenizer = Tokenizer(
        "/disk/disk_20T/share/Llama-3-8B-Instruct",
        special_token_list=["[RQ]", "[RD]", "[CON]", "<paragraph>", "</paragraph>"]
    )
    result = tokenizer.tokenize(
        structured_input=[
            '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nsystem prompt 1<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nuser input 1<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n', 
            '[RQ]<paragraph>123</paragraph>model output 1<|eot_id|>', 
            '<|start_header_id|>user<|end_header_id|>\n\nuser input 2<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n', 
            '[RQ]<paragraph>456</paragraph>model output 2<|eot_id|>', 
            ''
        ],
        max_length=1024, 
        special_token_id_list_for_repr=[tokenizer.tokenizer.convert_tokens_to_ids("[RQ]")],
        train_on_input=False,
        mask_token_id_from_to=[tokenizer.tokenizer.convert_tokens_to_ids("<paragraph>"), tokenizer.tokenizer.convert_tokens_to_ids("</paragraph>")],
        check_consistency=True,
    )
    print(result)
    seq = []
    for label, input_ids in zip(result['labels'], result['input_ids']):
        if label == -100:
            if len(seq) != 0:
                print(tokenizer.tokenizer.decode(seq))
                seq.clear()
        else:
            seq.append(input_ids)
    print(tokenizer.decode(seq))
        