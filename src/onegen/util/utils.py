import time
from typing import Any, List, Dict, Union
import json
import jsonlines
import os
import pickle
from tqdm import tqdm
import torch.distributed as dist
import torch

def _print(message:Any):
    print(f"[{time.ctime()}] {message}")

class FileWriter:
    @classmethod
    def get_current_time(cls) -> str:
        return time.ctime().replace("  "," ").replace(" ","-").replace(":","-")

    @classmethod
    def write_jsonl(cls, data:List[Dict], file_name:str, overwrite:bool=False, rewrite_name:bool=True):
        if FileReader.is_existed(file_name) == True and not overwrite and not rewrite_name:
            raise ValueError(f"The file `{file_name}` has existed. Please set the other `file_name` or make the `overwrite` True.")
        if FileReader.is_existed(file_name) == True and rewrite_name and not overwrite:
            file_name = f"{file_name}-{cls.get_current_time()}"
        if dist.get_rank() == 0:
            with jsonlines.open(file_name, 'w') as writer:
                pbar = tqdm(total=len(data))
                for item in data:
                    pbar.update(1)
                    writer.write(item)
                pbar.close()
        dist.barrier()

    @classmethod
    def write_json(cls, data:dict, file_name:str, overwrite:bool=False, rewrite_name:bool=True):
        if FileReader.is_existed(file_name) == True and not overwrite and not rewrite_name:
            raise ValueError(f"The file `{file_name}` has existed. Please set the other `file_name` or make the `overwrite` True.")
        if FileReader.is_existed(file_name) == True and rewrite_name and not overwrite:
            file_name = f"{file_name}-{cls.get_current_time()}"
        if dist.get_rank() == 0:
            with open(file_name, 'w') as writer:
                json.dump(data, writer)
        dist.barrier()

    @classmethod
    def write_pickle(cls, data, file_name:str, overwrite:bool=False, rewrite_name:bool=True):
        if FileReader.is_existed(file_name) == True and not overwrite and not rewrite_name:
            raise ValueError(f"The file `{file_name}` has existed. Please set the other `file_name` or make the `overwrite` True.")
        if FileReader.is_existed(file_name) == True and rewrite_name and not overwrite:
            file_name = f"{file_name}-{cls.get_current_time()}"
        if dist.get_rank() == 0:
            with open(file_name, 'wb') as writer:
                pickle.dump(data, writer)
        dist.barrier()

    @classmethod
    def write_pt(cls, data, file_name:str, overwrite:bool=False, rewrite_name:bool=True):
        if FileReader.is_existed(file_name) == True and not overwrite and not rewrite_name:
            raise ValueError(f"The file `{file_name}` has existed. Please set the other `file_name` or make the `overwrite` True.")
        if FileReader.is_existed(file_name) == True and rewrite_name and not overwrite:
            file_name = f"{file_name}-{cls.get_current_time()}"
        if dist.get_rank() == 0:
            torch.save(data, file_name)
        dist.barrier()

class FileReader:
    @classmethod
    def is_existed(cls, file_name:str) -> bool:
        if file_name == None:
            return False
        return os.path.exists(file_name)

    @classmethod
    def get_num_of_line(cls, file_name:str) -> int:
        """Get the number of row in the file"""
        cnt = 0
        with open(file_name, 'r') as lines:
            for line in lines:
                cnt += 1
        return cnt
    
    @classmethod
    def read_json(cls, file_name:str) -> Union[Dict,List[Dict]]:
        assert cls.is_existed(file_name), \
            f"The file `{file_name}` is not existed."
        with open(file_name, 'r') as file:
            results = json.load(file)
        return results

    @classmethod
    def read_jsonl(cls, file_name:str, progress_bar:bool=False) -> List[Dict]:
        assert cls.is_existed(file_name), \
            f"The file `{file_name}` is not existed."
        results: List = []
        if progress_bar:
            pbar = tqdm(total=cls.get_num_of_line(file_name))
        else:
            pbar = tqdm(total=1)
        with jsonlines.open(file_name, 'r') as reader:
            for item in reader:
                results.append(item)
                pbar.update(1)
        pbar.close()
        return results

    @classmethod
    def read_pickle(cls, file_name:str) -> Any:
        assert cls.is_existed(file_name), \
            f"The file `{file_name}` is not existed."
        with open(file_name, 'rb') as file:
            results = pickle.load(file)
        return results

    @classmethod
    def read_pt(cls, file_name:str) -> Any:
        assert cls.is_existed(file_name), \
            f"The file `{file_name}` is not existed."
        return torch.load(file_name)