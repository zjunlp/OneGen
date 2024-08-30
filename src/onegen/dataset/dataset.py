import sys
sys.path.append('../')
import torch
from onegen.config import DataConfig, SpecialTokenConfig
from onegen.tokenizer import Tokenizer
from onegen.util import FileReader, FileWriter
from onegen.util import _print
from tqdm import tqdm
import jsonlines
from onegen.templator import DocumentTemplator
import random
from typing import *

class AutoDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        db_file_config: DataConfig,
        train_file_config: DataConfig,
        # special_token_config: SpecialTokenConfig,
        tokenizer: Tokenizer,
    ):
        self.tokenizer: Tokenizer = tokenizer
        self._db_file_config: DataConfig = db_file_config
        self._train_file_config: DataConfig = train_file_config
        # self._special_token_config: SpecialTokenConfig = special_token_config

        # final data
        self.db_data: Dict = {}
        self.train_data: List = []
        self.read_db_file(
            tokenization=True, progress_bar=True,
            train_on_input=False, check_consistency=True,
            overwrite=True
        )
        self.read_train_file(
            tokenization=True, progress_bar=True,
            train_on_input=False, check_consistency=True,
            overwrite=True
        )
        self.check_positive_in_db()
        self.all_uid_list:List[str] = list(self.db_data.keys())
        
    def __getitem__(self, idx:int) -> Tuple[int, List[List], List[List]]:
        return idx, self.train_data[idx]['positive'], self.train_data[idx]['negative']
    
    def __len__(self):
        return len(self.train_data)

    def read_db_file(
        self, 
        tokenization:bool, 
        progress_bar:bool,
        train_on_input:bool=False,
        check_consistency:bool=True,
        overwrite:bool=True,
    ):
        """
        uid: {
            "content": [],
            "meta": "",
            "tokenized": {
                "input_ids": [],
                "labels": [],
                "embedding_index": []
            } 
        }
        """
        # Prepare
        templator = self._db_file_config.templator
        cache_file_path = self._db_file_config.cache_file_path
        file_path = self._db_file_config.file_path
        max_length = self._db_file_config.max_length
        _repr_token:List[str] = self._db_file_config.repr_token
        _mask_token_from_to:List = self._db_file_config.mask_token_from_to
        special_token_id_list_for_repr:List = [self.tokenizer.convert_tokens_to_ids(token) for token in _repr_token]
        mask_token_id_from_to:List = []
        if _mask_token_from_to != None and len(_mask_token_from_to) != 0:
            if isinstance(_mask_token_from_to[0], int):
                assert len(_mask_token_from_to) == 2
                for token in _mask_token_from_to:
                    mask_token_id_from_to.append(
                        self.tokenizer.convert_tokens_to_ids(token)
                    )
            else:
                for token_list in _mask_token_from_to:
                    mask_token_id_from_to.append(list())
                    for token in token_list:
                        mask_token_id_from_to[-1].append(
                            self.tokenizer.convert_tokens_to_ids(token)
                        )
        invalid = 0
        
        # load cache if the file is existed
        if cache_file_path!= None and FileReader.is_existed(cache_file_path):
            _print(f"loading tokenized db file from `{cache_file_path}`")
            self.db_data:Dict = FileReader.read_pickle(cache_file_path)
            _print(f"Example Document:\n{self.tokenizer.decode(self.db_data[list(self.db_data.keys())[0]]['tokenized']['input_ids'])}")
            return
        
        # read file_path and then tokenize it
        assert FileReader.is_existed(file_path), \
            f"The database file `{file_path}` is not existed."
        _print(f"reading database data from `{file_path}` ...")
        if progress_bar:
            pbar = tqdm(total=FileReader.get_num_of_line(file_path))
        else:
            pbar = tqdm(total=1)
        with jsonlines.open(file_path, 'r') as reader:
            for item in reader:
                pbar.update(1)
                uid: str = self.get_db_id(item)
                messages:List[Dict] = self.get_db_messages(item)
                structured_input:List[str] = templator.wrap(messages)
                new_item = {
                    'meta': self.get_db_meta_data(item),
                    'content': "".join(structured_input)
                }
                if tokenization:
                    new_item['tokenized'] = self.tokenizer.onegen_tokenize(
                        structured_input=structured_input,
                        max_length=max_length,
                        special_token_id_list_for_repr=special_token_id_list_for_repr,
                        mask_token_id_from_to=mask_token_id_from_to,
                        train_on_input=train_on_input,
                        check_consistency=check_consistency
                    )
                if tokenization and len(new_item['tokenized']['embedding_index']) == 0:
                    # maybe, due to the truncation
                    invalid += 1
                else:
                    self.db_data[uid] = new_item
                pbar.set_description(f"invalid: {str(invalid)}")
                pbar.update(1)
        pbar.close()

        # cache tokenized data
        # TODO: multi-cpu scenario
        if cache_file_path != None:
            _print(f"start saving tokenized file for database in the file `{cache_file_path}` ...")
            FileWriter.write_pickle(self.db_data, cache_file_path, overwrite=overwrite)
            _print(f"saving done!")

    def read_train_file(
        self, 
        tokenization:bool, 
        progress_bar:bool,
        train_on_input:bool=False,
        check_consistency:bool=True,
        overwrite:bool=True,
    ):
        # Prepare
        templator = self._train_file_config.templator
        cache_file_path = self._train_file_config.cache_file_path
        file_path = self._train_file_config.file_path
        max_length = self._train_file_config.max_length
        _repr_token_list:List[str] = self._train_file_config.repr_token
        _mask_token_from_to:List = self._train_file_config.mask_token_from_to
        special_token_id_list_for_repr:List = [self.tokenizer.convert_tokens_to_ids(token) for token in _repr_token_list]
        mask_token_id_from_to:List = []
        if _mask_token_from_to != None and len(_mask_token_from_to) != 0:
            if not isinstance(_mask_token_from_to[0], list):
                assert len(_mask_token_from_to) == 2
                for token in _mask_token_from_to:
                    mask_token_id_from_to.append(
                        self.tokenizer.convert_tokens_to_ids(token)
                    )
            else:
                for token_list in _mask_token_from_to:
                    mask_token_id_from_to.append(list())
                    for token in token_list:
                        mask_token_id_from_to[-1].append(
                            self.tokenizer.convert_tokens_to_ids(token)
                        )
        invalid = 0

        # load cache if the file existed
        if cache_file_path != None and FileReader.is_existed(cache_file_path):
            _print(f"loading tokenized train file from `{cache_file_path}`")
            self.train_data: List = FileReader.read_pickle(cache_file_path)
            _print(f"Example Case for Training:\n{self.train_data[0]}")
            return
        
        # read file_path and tokenize it
        assert FileReader.is_existed(file_path), \
            f"The train file `{file_path}` is not existed."
        _print(f"reading training data from `{file_path}`")
        if progress_bar:
            pbar = tqdm(total=FileReader.get_num_of_line(file_path))
        else:
            pbar = tqdm(total=1)
        with jsonlines.open(file_path, 'r') as reader:
            for item in reader:
                pbar.update(1)
                messages: List[Dict] = self.get_train_messages(item)
                structured_input: List[str] = templator.wrap(messages)
                positive_list:List[List[str]] = self.get_train_positive(item, repr_token_list=_repr_token_list)
                negative_list:List[List[str]] = self.get_train_negative(item, repr_token_list=_repr_token_list)
                assert len(positive_list) == len(negative_list), \
                    "Please check the training data."
                new_item = {
                    'meta': self.get_train_meta_data(item),
                    'positive': positive_list,
                    'negative': negative_list,
                    'content': "".join(structured_input)
                }
                if tokenization:
                    new_item['tokenized']:Dict = self.tokenizer.onegen_tokenize(
                        structured_input=structured_input,
                        max_length=max_length,
                        special_token_id_list_for_repr=special_token_id_list_for_repr,
                        mask_token_id_from_to=mask_token_id_from_to,
                        train_on_input=train_on_input,
                        check_consistency=check_consistency
                    )
                    assert len(new_item['tokenized']['embedding_index']) <= len(new_item['positive'])
                self.train_data.append(new_item)
            pbar.close()

        # cache tokenized data
        if cache_file_path != None:
            _print(f"start saving tokenized file for database in the file `{cache_file_path}` ...")
            FileWriter.write_pickle(self.train_data, cache_file_path, overwrite=overwrite)
            _print(f"saving done!")

    def check_positive_in_db(self):
        pbar = tqdm(total=len(self.train_data))
        invalid = 0
        for item in self.train_data:
            for pos_list in item['positive']:
                if len(pos_list) == 0:
                    continue
                for pos in pos_list:
                    if pos == None:
                        invalid += 1
                    else:
                        doc_uid: str = self.get_doc_id_for(pos)
                        sent_id: int = self.get_sent_id_for(pos)
                        if doc_uid not in self.db_data:
                            invalid += 1
                        else:
                            if sent_id >= len(self.db_data[doc_uid]['tokenized']['embedding_index']):
                                invalid += 1
            pbar.set_description(str(invalid))
            pbar.update(1)
        pbar.close()
        _print(f"check done and the number of invalid positive examples is `{invalid}`")
            
    # custom function for parsing id of positive and negative
    def get_doc_id_for(self, uid:str) -> str:
        assert isinstance(uid, str)
        doc_id = uid
        if "-" in uid:
            doc_id = uid.split("-")[0]
        return doc_id

    def get_sent_id_for(self, uid:str) -> int:
        sent_id = 0
        assert isinstance(uid, str)
        if "-" in uid:
            sent_id = int(uid.split('-')[1])
        return sent_id
    
    def make_db_uid(self, doc_id, sent_id) -> str:
        # this is the default uid maker
        return f"{doc_id}-{sent_id}"

    # custom function for db data
    def get_db_id(self, item:dict) -> str:
        return str(item['uid'])
    
    def get_db_meta_data(self, item:dict) -> Dict:
        return item
    
    def get_db_messages(self, item:dict) -> List[Dict]:
        return item['messages']
    
    # custom function for train data
    def get_train_positive(self, item:dict, repr_token_list:List[str]=None) -> List[List[str]]:
        """repr_token_list is used for check."""
        # [["1", "2"], ["3"], ["5"]]
        positive_list:List = []
        for message in item['messages']:
            if message['role'] == 'assistant':
                for positive in message['positive']:
                    if isinstance(positive, list):
                        positive_list.append([str(p) for p in positive])
                    else:
                        if positive == None:
                            positive_list.append([])
                        else:
                            positive_list.append([str(positive)])
                if repr_token_list != None and len(repr_token_list) > 0:
                    total = 0
                    for token in repr_token_list:
                        total += message['content'].count(token)
                    if total != len(positive_list):
                        raise ValueError(f"{total} != {len(positive_list)}. Please make sure the number of special token and the number of positive being same.")
        return positive_list

    def get_train_negative(self, item:dict, repr_token_list:List[str]=None) -> List[List[str]]:
        """repr_token_list is used for check."""
        negative_list:List = []
        for message in item['messages']:
            if message['role'] == 'assistant':
                for negative in message['negative']:
                    assert isinstance(negative, list)
                    negative_list.append([str(n) if n != None else None for n in negative])
                if repr_token_list != None and len(repr_token_list) > 0:
                    total = 0
                    for token in repr_token_list:
                        total += message['content'].count(token)
                    if total != len(negative_list):
                        raise ValueError(f"{total} != {len(negative_list)}. Please make sure the number of special token and the number of positive being same.")
        return negative_list

    def get_train_meta_data(self, item:dict) -> Dict:
        return item
    
    def get_train_messages(self, item:dict) -> List[Dict]:
        return item['messages']

    # other functions which will be used in collator
    def get_tokenized_info_for_train_data(self, idx:int) -> Dict:
        return self.train_data[idx]['tokenized']

    def get_random_uid_list(self, n:int) -> List[str]:
        uid_list:List = []
        random_index = random.sample(
            range(0, len(self.db_data)), n
        )
        for index in random_index:
            doc_id:str = self.all_uid_list[index]
            sent_id = random.randint(
                0, len(self.db_data[doc_id]['tokenized']['embedding_index'])-1
            )
            uid_list.append(self.make_db_uid(doc_id, sent_id))
        return uid_list
    
    def get_tokenized_input(self, idx) -> Dict:
        return self.train_data[idx]['tokenized']

    def get_tokenized_db(self, uid:Union[str, List]) -> Union[Dict, List[Dict]]:
        if isinstance(uid, str):
            return self.db_data[uid]['tokenized']
        elif isinstance(uid, List):
            return [self.db_data[u]['tokenized'] for u in uid]
        else:
            raise ValueError("Invalid uid!")