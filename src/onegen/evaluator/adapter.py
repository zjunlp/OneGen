from .config import FileConfig, InferenceConfig
import sys
sys.path.append('../')
from onegen.util import FileReader, _print
from onegen.templator import Templator, DocumentTemplator
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Union
from tqdm import tqdm
from transformers import GenerationConfig
import torch

class Adapter:
    def __init__(
        self,
        file_config: FileConfig,
        inference_config: InferenceConfig,
        test_templator: Templator=None,
        db_templator: Templator=None,
        **kwargs
    ):
        raise NotImplementedError()
    
    def convert(self):
        raise NotImplementedError()
    
    def __getitem__(self, idx):
        raise NotImplementedError()

    def __len__(self) -> int:
        raise NotImplementedError()
    
class EntityLinkingAdapter(Adapter):
    def __init__(
        self,
        file_config:FileConfig,
        inference_config: InferenceConfig,
        **kwargs,
    ): 
        # Here, we didn't use the test_templator and db_templator
        # we just using the template in kwargs
        # kwargs: {input_template, db_template}
        self.file_config:FileConfig = file_config
        self.inference_config:InferenceConfig = inference_config
        self.test_templator = file_config.test_templator
        self.db_templator = file_config.db_templator
        self.kwargs = kwargs

        assert not self.file_config.online_retrieval
        self.db_meta_data = None
        self.db_data = None
        self.db_label = None
        
        if self.file_config.pre_cache:
            self.prepare_precache()
        else:
            self.convert()
    
    def prepare_precache(self):
        """
        {
            "uid": "",
            "messages": [
                {"role": "user", "content": ""},
                {"role": "assistant", "content": ""}
            ]
        }
        """
        # prepare the data for embedding
        self.db_meta_data:List[Dict] = FileReader.read_jsonl(
            self.file_config.db_file_path,
            progress_bar=True
        )
        self.db_label:List[str] = list()
        self.db_data:List[str] = list()
        pbar = tqdm(total=len(self.db_meta_data))
        for item in self.db_meta_data:
            self.db_data.append(
                self.get_db_prompt(item)
            )
            self.db_label.append(
                self.get_db_label(item)
            )
            pbar.update(1)
        pbar.close()
    
    def get_db_label(self, item:dict) -> str:
        return item['uid']

    def get_db_prompt(self, item:dict) -> str:
        return "<s>" + "".join(self.db_templator.wrap(item['messages']))
    
    def convert(self):
        """
        entity linking:
        {
            'id': 0,
            'title': '',
            'text': '',
            'labels': [
                {
                    'id': 0, 'span': [19, 24], 'entity_id': '', 'name': '',
                    'parent': None, 'children': [],
                    'optional': false, 'type': '',
                },
                {
                    'id': 1, 'span': [78, 86], 'entity_id': '', 'name': '',
                    'parent': None, 'children': [],
                    'optional': false, 'type': '',
                },
            ]
        }
        """
        self.meta_data:List[Dict] = FileReader.read_jsonl(
            self.file_config.test_file_path,
            progress_bar=True
        )
        self.data:List[Dict] = list()
        pbar = tqdm(total=len(self.data))
        for item in self.meta_data:
            doc_embedding, doc_embedding_label = self.get_embedding_and_label(item)
            candidate_list, meta_candidate_list = self.get_candidate_list(item)
            assert doc_embedding != None and doc_embedding_label != None
            self.data.append(dict(
                prompt=self.get_prompt(item),
                candidate_list=candidate_list,
                meta_candidate_list=meta_candidate_list,
                max_new_tokens=self.inference_config.max_new_tokens,
                input_ids=None,
                generation_config=self.inference_config.generation_config,
                embed_batch_size=self.inference_config.embed_batch_size,
                doc_embedding=doc_embedding,
                doc_embedding_label=doc_embedding_label,
                sentence_connector=self.inference_config.sentence_connector,
                max_retrieval_cnt=self.inference_config.max_retrieval_cnt,
                skip_repr_token_cnt=self.inference_config.skip_repr_token_cnt,
            ))
            pbar.update(1)
        pbar.close()

    def get_prompt(self, item:dict) -> str:
        question = item['text']
        return self.kwargs['input_template'].format(input=question)

    def get_candidate_list(self, item:dict) -> Tuple[
        Union[List[str], List[List[str]]],
        Union[List[str], List[List[str]]]
    ]:
        return None, None
    
    def get_embedding_and_label(self, item:dict) -> Tuple[torch.Tensor, List[str]]:
        return self.file_config.db_embedding, self.file_config.db_label

    def __getitem__(self, idx:int) -> Tuple[Dict, Dict]:
        return self.data[idx], self.meta_data[idx]
    
    def __len__(self) -> int:
        return len(self.data)

class EntityDisambiguationAdapter(EntityLinkingAdapter):
    def __init__(
        self,
        file_config:FileConfig,
        inference_config: InferenceConfig,
        **kwargs,
    ): 
        super().__init__(file_config, inference_config, **kwargs)
    
    def get_prompt(self, item:dict) -> str:
        return self.kwargs['input_template'].format(input=item['sentence']) + item['labeled_sentence']

class MultiHopQAAdapter(Adapter):
    def __init__(
        self, 
        file_config:FileConfig,
        inference_config: InferenceConfig,
        **kwargs,
    ):
        # the kwargs here is used for specific task
        # here maybe is:
        # kwargs['input_template'] = "{input}"
        self.file_config:FileConfig = file_config
        self.inference_config:InferenceConfig = inference_config
        self.test_templator = file_config.test_templator
        self.db_templator = file_config.db_templator
        self.kwargs:dict = kwargs

        assert self.file_config.online_retrieval
        assert self.file_config.db_file_path == None

        self.convert()

    def convert(self):
        """
        {
            "_id": "",
            "type": "",
            "question": "",
            "context": [
                [title, [sent1, sent2, ..., sent_n]]
            ],
            "entity_ids": "",
            "supporting_facts": [
                [title, sent_id], [title, sent_id]
            ],
            "evidences": [
                [head, relation, tail],
                []
            ],
            "answer": "",
            "evidences_id": [
                [head_qid, relation_qid, tail_qid],
                [head_qid, relation_qid, tail_qid],
            ],
            "answer_id": ""
        }
        """
        self.meta_data:List[Dict] = FileReader.read_json(
            self.file_config.test_file_path
        )
        self.data:List[Dict] = []
        pbar = tqdm(total=len(self.meta_data))
        for item in self.meta_data:
            doc_embedding, doc_embedding_label = self.get_embedding_and_label(item)
            candidate_list, meta_candidate_list = self.get_candidate_list(item)
            assert doc_embedding == None and doc_embedding_label == None
            self.data.append(dict(
                prompt=self.get_prompt(item),
                candidate_list=candidate_list,
                meta_candidate_list=meta_candidate_list,
                max_new_tokens=self.inference_config.max_new_tokens,
                input_ids=None,
                generation_config=self.inference_config.generation_config,
                embed_batch_size=self.inference_config.embed_batch_size,
                doc_embedding=doc_embedding,
                doc_embedding_label=doc_embedding_label,
                sentence_connector=self.inference_config.sentence_connector,
                max_retrieval_cnt=self.inference_config.max_retrieval_cnt,
                skip_repr_token_cnt=self.inference_config.skip_repr_token_cnt,
            ))
            pbar.update(1)
        pbar.close()

    def get_candidate_list(self, item:dict) -> Tuple[
        Union[List[str], List[List[str]]], 
        Union[List[str], List[List[str]]]
    ]:
        # used for embedding
        candidate_list:List = []
        # used for concatenate
        meta_candidate_list: List = []
        for context in item['context']:
            title, doc_list = context
            if 'embed_template' in self.kwargs:
                assert '{input}' in self.kwargs['embed_template']
                candidate_list.append(
                    [self.kwargs['embed_template'].format(input=f"wiki title: {title}\ncontent: {doc_list[0]}")]
                )
            else:
                candidate_list.append([f"wiki title: {title}\ncontent: {doc_list[0]}"])
            meta_candidate_list.append([f"wiki title: {title}\ncontent: {doc_list[0]}"])
            for sent in doc_list[1:]:
                candidate_list[-1].append(sent)
                meta_candidate_list[-1].append(sent)
        return candidate_list, meta_candidate_list
    
    def get_prompt(self, item:dict) -> str:
        question = item['question']
        return self.kwargs['input_template'].format(input=question)

    def get_embedding_and_label(self, item:dict) -> Tuple[torch.Tensor, List[str]]:
        return self.file_config.db_embedding, self.file_config.db_label

    def __getitem__(self, idx:int) -> Tuple[Dict, Dict]:
        return self.data[idx], self.meta_data[idx]
    
    def __len__(self) -> int:
        return len(self.data)

if __name__ == '__main__':
    pass