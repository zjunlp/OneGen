import time
from typing import Any, List, Dict, Union
import json
import jsonlines
import os
import pickle
from tqdm import tqdm
import torch.distributed as dist
import torch
import faiss


def _print(message:Any):
    print(f"[{time.ctime()}] {message}")


def faiss_sim_matrix(query_embedding, doc_embedding):
    """
    Calculate similarity matrix using FAISS with inner product (cosine similarity proxy).
    
    query_embedding: Tensor, shape [1, dim]
    doc_embedding: Tensor, shape [n_doc, dim]
    max_retrieval_cnt: Maximum number of nearest documents to retrieve. If None, retrieve all documents.
    """
    # Normalize embeddings to use inner product as cosine similarity
    query_embedding = query_embedding / query_embedding.norm(dim=1, keepdim=True)

    # Seperate doc_embedding to multiple GPUs, regarding doc usually has a large size
    num_gpus = torch.cuda.device_count()
    doc_chunks = torch.chunk(doc_embedding, num_gpus)  # Break doc_embedding down into sub-tensors
    normalized_chunks = []

    for i in range(num_gpus):
        with torch.cuda.device(i):
            chunk = doc_chunks[i].to(i)  # Moving torches
            normalized_chunk = chunk / chunk.norm(dim=1, keepdim=True)  # Normalization
            normalized_chunks.append(normalized_chunk.cpu())  # Move the normalized tensors back to CPU, since the total size is large

    # Gather
    doc_embedding = torch.cat(normalized_chunks, dim=0)

    # Create faiss index with inner product metric
    res = faiss.StandardGpuResources()
    config = faiss.GpuIndexFlatConfig()
    config.device = 0 

    # Create inner product indexes on GPU
    index = faiss.GpuIndexFlatIP(res, doc_embedding.shape[1], config)
    doc_embedding = doc_embedding.to(torch.float32) # Make sure doc_embedding is float32
    index.add(doc_embedding.numpy())  # FAISS needs NumPy array

    # Search for the nearest documents based on inner product
    query_embedding = query_embedding.to(torch.float32) # Make sure query_embedding is float32
    _, nearest_idx = index.search(query_embedding.cpu().numpy(), 1)
    
    return nearest_idx[0]  # Return the indices of the nearest documents


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