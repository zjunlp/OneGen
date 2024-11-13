from .config import ComponentConfig
from typing import Dict, List, Tuple, Union
from tqdm import tqdm
from dataclasses import dataclass
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig
import torch
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
import jsonlines
import re

import sys
sys.path.append('../')
from onegen.util import FileReader, _print, sim_matrix, faiss_sim_matrix, FileWriter
from onegen.util import DEFAULT_GENERATION_CONFIG, MAX_NEW_TOKENS, MAX_RETRIEVAL_CNT
from onegen.dataset import padding_input_ids

class AutoAddLogitsProcessor(LogitsProcessor):
    def __init__(self, rules:Union[List[int], List[List[int]]], nexts: Union[int, List[int]]):
        # List[int], int
        # List[List[int]] List[int]
        self.nexts = nexts
        if isinstance(nexts, int):
            self.nexts = [nexts]
        assert isinstance(self.nexts, list)
        self.rules = rules
        if isinstance(rules[0], int):
            self.rules = [rules]
        assert isinstance(self.rules, list) and isinstance(self.rules[0], list)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        n_beams = input_ids.shape[0]
        assert input_ids.shape[0] == scores.shape[0]
        for b in range(n_beams):
            for idx, rule in enumerate(self.rules):
                # print(input_ids[b, -len(rule):], rule)
                if input_ids[b, -len(rule):].tolist() == rule:
                    new_token_id = self.nexts[idx]
                    scores[b, 0:new_token_id] = -float('inf')
                    scores[b, new_token_id+1:] = -float('inf')
                    scores[b, 0:new_token_id] = 0.
                    scores[b, new_token_id+1:] = 0.
                    scores[b, new_token_id] = 1.
        return scores

class Backend:

    def __expand_tokenizer(self, special_token_list:List[str], tokenizer:AutoTokenizer) -> AutoTokenizer:
        num_added_tokens = tokenizer.add_tokens(
            special_token_list, special_tokens=True
        )
        _print(f"{num_added_tokens} tokens have been added including {special_token_list}")
        return tokenizer

    def load_model_and_tokenizer(self, component_config:ComponentConfig) -> Tuple[Union[AutoModel, AutoModelForCausalLM], AutoTokenizer]:
        model_path:str = component_config.model_path
        tokenizer_path:str = component_config.tokenizer_path
        model_class = component_config.model_class
        torch_dtype = component_config.torch_dtype
        special_token_list:List[str] = component_config.special_token_list
        add_prefix_space:bool = component_config.add_prefix_space
        add_eos_token:bool = component_config.add_eos_token
        add_bos_token:bool = component_config.add_bos_token

        # Step 1. Loading model
        _print(f"loading model from `{model_path}` ...")
        model = model_class.from_pretrained(
            model_path, torch_dtype=torch_dtype, device_map="auto"
        )

        # Step 2. Loading tokenizer
        _print(f"loading tokenizer from `{tokenizer_path}` ...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, add_prefix_space=add_prefix_space)
        
        # Step 3. Expanding tokenizer
        if special_token_list != None:
            tokenizer = self.__expand_tokenizer(special_token_list, tokenizer)
        tokenizer.add_eos_token = add_eos_token
        tokenizer.add_bos_token = add_bos_token

        return model, tokenizer
    
    def tokenize(self, prompt:str, tokenizer:AutoTokenizer) -> torch.Tensor:
        tokenized_item:dict = tokenizer(prompt, return_tensors=None)
        return torch.as_tensor(
            [tokenized_item['input_ids']],
            device="cuda"
        )

    def __init__(
        self,
        generator_config:ComponentConfig,
        retriever_config:ComponentConfig,
    ):
        self.generator, self.generator_tokenizer = self.load_model_and_tokenizer(generator_config)
        self.retriever, self.retriever_tokenizer = self.load_model_and_tokenizer(retriever_config)

        # update the stop_token_list
        self.generator_config:ComponentConfig = generator_config
        self.retriever_config:ComponentConfig = retriever_config
        self.generator_config.stop_token_id_list: List[int] = [self.generator_tokenizer.convert_tokens_to_ids(token) for token in self.generator_config.stop_token_list]
        self.generator_config.repr_token_id_list: List[int] = [self.generator_tokenizer.convert_tokens_to_ids(token) for token in self.generator_config.repr_token_list]
        assert self.generator_tokenizer.stop_token_list != None and len(self.generator_tokenizer.stop_token_list) > 0, \
            "generator must have the end of token"
        self.generator_config.padding_token_id = self.generator_tokenizer.convert_tokens_to_ids(self.generator_config.padding_token)

    @torch.no_grad()
    def encode_backend(self, batch_prompts:List[str], skip_repr_token_cnt:int=0) -> torch.Tensor:
        # the argument `skip_repr_token_cnt` will be only used in OneGen
        # this is the retriever named Contriever 
        # https://huggingface.co/facebook/contriever-msmarco
        inputs = self.retriever_tokenizer(
            batch_prompts, padding=True, truncation=True, return_tensors='pt'
        ).to('cuda')
        outputs = self.retriever(**inputs)
        def mean_pooling(token_embeddings, mask):
            token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
            sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
            return sentence_embeddings
        embeddings = mean_pooling(outputs[0], inputs['attention_mask'])
        return embeddings
        
    @torch.no_grad()
    def encode(
        self,
        texts:Union[str, List[str], List[List[str]]],
        batch_size:int=1,
        sentence_connector:str="",
        skip_repr_token_cnt:int=0
    ) -> torch.Tensor:
        """
        texts:
            - str. Single Sentence
            - List[str]. each item is a document to be encoded
            - List[List[str]] each item is a document consisted of many sentences
        - sentence_connector:
            - will be used when the type of texts is List[List[str]]
        - skip_repr_token_cnt:
            - will only be used in the OneGen
        """
        if isinstance(texts, str):
            texts = [texts]
        if isinstance(texts[0], list):
            _temp = []
            for doc_list in texts:
                _temp.append(sentence_connector.join(doc_list)+sentence_connector)
            texts = _temp
        assert isinstance(texts, list) and isinstance(texts[0], str)

        idx = 0
        batch = []
        embedding_list:List = []
        while idx < len(texts):
            batch.append(texts[idx])
            if len(batch) == batch_size:
                embedding = self.encode_backend(batch, skip_repr_token_cnt)
                embedding_list.append(embedding)
                batch.clear()
            idx += 1
        if len(batch) > 0:
            embedding = self.encode_backend(batch, skip_repr_token_cnt)
            embedding_list.append(embedding)
            batch.clear()
        embedding_cache = torch.cat(embedding_list, dim=0)  # [n, dim]
        return embedding_cache

    @torch.no_grad()
    def generate(
        self, 
        generation_config:GenerationConfig=DEFAULT_GENERATION_CONFIG,
        max_new_tokens:int=MAX_NEW_TOKENS,
        prompt:str=None,
        input_ids:torch.Tensor=None,
        past_key_values=None,
        logits_processor: LogitsProcessorList=None,
    ) -> Dict:
        # check
        assert prompt or input_ids
        if prompt:
            input_ids = self.tokenize(prompt, self.generator_tokenizer)
        
        # generate
        generation_output = self.generator.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            past_key_values=past_key_values,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
            eos_token_id=self.generator_config.stop_token_id_list, # + self.generator_config.repr_token_id_list,
            logits_processor=logits_processor,
        )

        # post handle
        past_key_values = generation_output.past_key_values
        state:bool = generation_output.sequences[0][-1] == self.generator_tokenizer.eos_token_id
        if prompt:
            output = self.generator_tokenizer.decode(
                generation_output.sequences[0]
            )
            increment = output[len(prompt):]
        else:
            new_output = generation_output.sequence[0][len(input_ids[0]):]
            prompt = self.generator_tokenizer.decode(input_ids[0])
            increment = self.generator_tokenizer.decode(new_output)
            output = prompt + increment
        
        # return
        return dict(
            past_key_values=past_key_values,
            output=output,
            increment=increment,
            finish=state,
            hidden_states=generation_output.hidden_states, 
            token_seq=generation_output.sequences,  # generation_output.sequences[0]
        )

    @torch.no_grad()
    def retrieve(
        self,
        query:str,
        candidate_list:Union[List[str], List[List[str]]],
        meta_candidate_list:Union[List[str], List[List[str]]]=None,
        batch_size:int=1,
        doc_embedding:torch.Tensor=None,
        past_key_values=None,
        mapping_func=None,
        sentence_connector:str="",
        skip_repr_token_cnt:int=0,
        use_faiss:bool=False,
    ) -> Tuple[Union[str, List[str]], torch.Tensor]: 
        # ATTENTION: 
        # if the doc_embedding is not None, we must make sure that candidate_list and doc_embedding have the one-to-one correspondence.

        # 1. embedding query
        query_embedding:torch.Tensor = self.encode(query, sentence_connector=sentence_connector, skip_repr_token_cnt=skip_repr_token_cnt)  # [1, dim]
        # 2. embedding document
        if doc_embedding == None:
            doc_embedding:torch.Tensor = self.encode(candidate_list, batch_size, sentence_connector=sentence_connector, skip_repr_token_cnt=skip_repr_token_cnt)  # [n, dim]
        assert len(doc_embedding) == len(candidate_list)
        # 3. calculating similarity
        if use_faiss:
            # use faiss when retrieve
            nearest_idx = faiss_sim_matrix(query_embedding, doc_embedding)
        else:
            # without faiss
            scores = sim_matrix(query_embedding, doc_embedding) # [1,n]
            nearest_idx = int(torch.argmax(scores))
        return meta_candidate_list[nearest_idx], doc_embedding

class OneGen(Backend):
    def __init__(
        self,
        generator_config: ComponentConfig,
    ): 
        # generator and tokenizer
        self.generator, self.generator_tokenizer = self.load_model_and_tokenizer(generator_config)
        # padding config
        self.padding_side = generator_config.padding_side
        self.padding_token_id = self.generator_tokenizer.convert_tokens_to_ids(generator_config.padding_token)
        # update the stop_token_list
        self.generator_config:ComponentConfig = generator_config
        self.generator_config.stop_token_id_list: List[int] = [self.generator_tokenizer.convert_tokens_to_ids(token) for token in self.generator_config.stop_token_list]
        self.generator_config.repr_token_id_list: List[int] = [self.generator_tokenizer.convert_tokens_to_ids(token) for token in self.generator_config.repr_token_list]
        assert self.generator_config.stop_token_list != None and len(self.generator_config.stop_token_list) > 0, \
            "generator must have the end of token"
        self.generator_config.padding_token_id = self.generator_tokenizer.convert_tokens_to_ids(
            self.generator_config.padding_token
        )

    # This is only used for encoding
    @torch.no_grad()
    def get_embedding(
        self,
        input_ids,
        embedding_index: List
    ) -> torch.Tensor:
        # AutoModelForCausalLM.model is the encoder without lm_head.
        outputs = self.generator.model(input_ids=input_ids)
        hidden_states = outputs[0]
        row_index, column_index = embedding_index
        row_index = row_index.to(hidden_states.device)
        column_index = column_index.to(hidden_states.device)
        all_mention_representation = hidden_states[row_index, column_index] # [bs, dim]
        return all_mention_representation

    # This is only used for encoding
    # Because we need the `embedding_index`
    def tokenize_batch(
        self, 
        batch_prompts:List[str],
        # due to some special token appeared at the input
        # we need to ignore them
        skip_repr_token_cnt: int=0,
        # TODO: there are other strategies like using the index
        # This may be needed to improve
    ) -> Dict:
        # ATTENTION: some tokens in self.generator_config.repr_token_id_list
        # are not allowed for embedding
        max_length = 0
        tokenized_items = []
        for prompt in batch_prompts:
            # 1. tokenize
            tokenized_item:dict = self.generator_tokenizer(
                prompt, return_tensors=None, padding=False,
                truncation=False
            )
            max_length = max(max_length, len(tokenized_item['input_ids']))
            
            flag = False
            for repr_token_id in self.generator_config.repr_token_id_list:
                if repr_token_id in tokenized_item['input_ids']:
                    flag = True
            assert flag, f"The current prompt `{prompt}` has no special token for embedding."

            # 2. locate
            embedding_index = []
            for position, token_id in enumerate(tokenized_item['input_ids']):
                if token_id in self.generator_config.repr_token_id_list:
                    embedding_index.append(position)
            embedding_index = embedding_index[skip_repr_token_cnt:]
            assert len(embedding_index) > 0
            tokenized_item['embedding_index'] = embedding_index
            tokenized_items.append(tokenized_item)

        input_ids = list()
        row_index, column_index = list(), list()
        for row_id, item in enumerate(tokenized_items):
            for column_id in item['embedding_index']:
                row_index.append(row_id)
                column_index.append(column_id)
            # 3. padding
            input_ids.append(
                padding_input_ids(
                    item, padding_side=self.generator_config.padding_side,
                    input_padding_id=self.generator_config.padding_token_id,
                    max_length=max_length
                )['input_ids']
            )

        return dict(
            input_ids=torch.as_tensor(input_ids, dtype=torch.int32).cuda(),
            embedding_index=[torch.as_tensor(row_index), torch.as_tensor(column_index)]
        )

    @torch.no_grad()
    def encode_backend(
        self, 
        batch_prompts:List[str],
        # due to some special token appeared at the input
        # we need to ignore them
        skip_repr_token_cnt: int=0,
        # TODO: there are other strategies like using the index
        # This may be needed to improve
    ) -> torch.Tensor:
        batch_input = self.tokenize_batch(
            batch_prompts=batch_prompts,
            skip_repr_token_cnt=skip_repr_token_cnt
        )
        embedding = self.get_embedding(**batch_input)
        return embedding

        
    @torch.no_grad()
    def encode(
        self,
        texts:Union[str, List[str], List[List[str]]],
        batch_size:int=1,
        sentence_connector:str="",
        skip_repr_token_cnt:int=0
    ) -> torch.Tensor:
        """
        texts:
            - str. Single Sentence
            - List[str]. each item is a document to be encoded
            - List[List[str]] each item is a document consisted of many sentences
        - sentence_connector:
            - will be used when the type of texts is List[List[str]]
        - skip_repr_token_cnt:
            - will only be used in the OneGen
        """
        if isinstance(texts, str):
            texts = [texts]
        if isinstance(texts[0], list):
            _temp = []
            for doc_list in texts:
                _temp.append(sentence_connector.join(doc_list)+sentence_connector)
            texts = _temp
        assert isinstance(texts, list) and isinstance(texts[0], str)

        idx = 0
        batch = []
        embedding_list:List = []
        pbar = tqdm(total=len(texts))
        while idx < len(texts):
            batch.append(texts[idx])
            if len(batch) == batch_size:
                embedding = self.encode_backend(batch, skip_repr_token_cnt)
                embedding_list.append(embedding)
                batch.clear()
            idx += 1
            pbar.update(1)
        pbar.close()
        if len(batch) > 0:
            embedding = self.encode_backend(batch, skip_repr_token_cnt)
            embedding_list.append(embedding)
            batch.clear()
        embedding_cache = torch.cat(embedding_list, dim=0)  # [n, dim]
        return embedding_cache

    @torch.no_grad()
    def retrieve(
        self,
        query:str,
        candidate_list:Union[List[str], List[List[str]]],
        meta_candidate_list:Union[List[str], List[List[str]]]=None,
        batch_size:int=1,
        doc_embedding:torch.Tensor=None,
        sentence_connector:str="",
        past_key_values=None,
        mapping_func=None,
        skip_repr_token_cnt:int=0,
        use_faiss:bool=False,
    ) -> Tuple[Union[str, List[str]], torch.Tensor]:
        # we just modify the query embedding
        # 1. embedding query
        check_endswith = False
        for repr_token in self.generator_config.repr_token_list:
            if query.endswith(repr_token):
                check_endswith = True
                break
        if not check_endswith:
            _print(f"Error! The query for OneGen is `{query}`.")
            return None, None
        input_ids = torch.as_tensor(
            [self.generator_tokenizer(query, return_tensors=None)["input_ids"]],
            device="cuda"
        )
        query_output = self.generator(
            input_ids=input_ids,
            past_key_values=past_key_values,
            output_hidden_states=True,
            return_dict=True
        )
        query_embedding = query_output.hidden_states[-1][0,-1].unsqueeze(dim=0)  # [1, dim]

        # 2. embedding document
        if doc_embedding == None:
            doc_embedding:torch.Tensor = self.encode(
                candidate_list, batch_size, sentence_connector=sentence_connector,
                skip_repr_token_cnt=skip_repr_token_cnt
            )  # [n, dim]
        
        # 3. calculating similarity
        if use_faiss:
            # use faiss
            nearest_idx = faiss_sim_matrix(query_embedding, doc_embedding)
        else:
            # without faiss
            scores = sim_matrix(query_embedding, doc_embedding) # [1,n]
            nearest_idx = int(torch.argmax(scores))

        if len(doc_embedding) != len(candidate_list):
            # just used for multi hop qa
            assert mapping_func != None and len(doc_embedding) > len(candidate_list)
            return mapping_func(query, candidate_list, meta_candidate_list, doc_embedding, nearest_idx)
        else:
            return meta_candidate_list[nearest_idx], doc_embedding

class Evaluator:
    """
    model and tokenizer are in the Backend.
    The Evaluator is only responsible for data processing and file saving.
    """
    def __init__(self, **kwargs):
        raise NotImplementedError()

    def run_single(self, **kwargs):
        raise NotImplementedError()

    def run(self, **kwargs):
        raise NotImplementedError()

class RAGEvaluator(Evaluator):

    @classmethod
    def mapping_func(
        cls,
        query:str,
        # candidate_list:Union[List[str], List[List[str]]],
        candidate_list:List[List[str]],
        meta_candidate_list:List[List[str]],
        doc_embedding: torch.Tensor,
        index: int,
    ) -> Tuple[Union[str, List[str]], torch.Tensor]:
        assert len(doc_embedding) != len(candidate_list)
        assert isinstance(candidate_list[0], list)
        doc_id_list = []
        sent_id_list = []
        for doc_idx, doc in enumerate(candidate_list):
            for sent_idx, sent in enumerate(doc):
                doc_id_list.append(doc_idx)
                sent_id_list.append(sent_idx)
        assert len(doc_id_list) == len(doc_embedding)
        assert index<len(doc_id_list)
        return meta_candidate_list[doc_id_list[index]], doc_embedding

    def extract_query(self, output:dict) -> str:
        if self.mapping_func != None:
            # this is used for onegen
            return output['output']
        else:
            # this is used for baseline
            # We use rule to extract query from increment string.
            increment = output['increment']
            tags = ["First, ", "Second, ", "Third, ", "Fourth, ", "Fifth"]
            query = increment
            for tag in reversed(tags):
                if tag in increment:
                    query = tag.join(increment.split(tag)[1:])
                    break
            return query

    def extract_final_answer(self, output:dict) -> str:
        increment = output['increment']
        pattern = re.compile(r'<FINAL-ANSWER>(.*?)</FINAL-ANSWER>')
        result:list = pattern.findall(increment)
        if len(result) == 1:
            return result[0]
        else:
            return None

    def __init__(
        self,
        generator_config: ComponentConfig,
        retriever_config: ComponentConfig=None,
        **kwargs
    ):
        if retriever_config == None:
            self.backend = OneGen(generator_config)
            self.mapping_func = RAGEvaluator.mapping_func
        else:
            # This is baseline
            self.backend = Backend(generator_config, retriever_config)
            self.mapping_func = None
        
    def run_single(
        self,
        prompt:str,
        candidate_list:Union[List[str], List[List[str]]],
        meta_candidate_list:Union[List[str], List[List[str]]]=None,
        max_new_tokens:int=1024,
        input_ids=None,
        generation_config: GenerationConfig=DEFAULT_GENERATION_CONFIG,
        embed_batch_size:int=16,
        doc_embedding:torch.Tensor=None,
        doc_embedding_label:List[str]=None,
        sentence_connector:str="",
        max_retrieval_cnt:int=MAX_RETRIEVAL_CNT,
        skip_repr_token_cnt:int=0,
        use_faiss:bool=False,
    ) -> Tuple[str, str]:
        if generation_config == None:
            generation_config: GenerationConfig = DEFAULT_GENERATION_CONFIG
        output:dict = self.backend.generate(
            generation_config=generation_config,
            max_new_tokens=max_new_tokens,
            prompt=prompt,
            input_ids=None,
            past_key_values=None
        )
        retrieval_cnt = 0
        while not output["finish"]:
            # 1. retrieve
            query:str = self.extract_query(output)
            document, embedding_cache = self.backend.retrieve(
                query=query,
                candidate_list=candidate_list,
                meta_candidate_list=meta_candidate_list,
                batch_size=embed_batch_size,
                doc_embedding=doc_embedding,
                sentence_connector=sentence_connector,
                past_key_values=output['past_key_values'],
                mapping_func=self.mapping_func,
                skip_repr_token_cnt=skip_repr_token_cnt,
                use_faiss=use_faiss,
            )
            if isinstance(document, list):
                document = "".join(document)
            
            if document == None and embedding_cache == None:
                break

            # 2. concatenate
            prompt = self.backend.generator_config.concatenate_template.format(
                history=output['output'], document=document
            )
            
            # 3. generate
            output:Dict = self.backend.generate(
                generation_config=generation_config,
                max_new_tokens=max_new_tokens,
                prompt=prompt,
                input_ids=None,
                past_key_values=output['past_key_values']
            )

            retrieval_cnt += 1
            if retrieval_cnt >= max_retrieval_cnt:
                break
        return output['output'], self.extract_final_answer(output)

    def run(self, adapter, output_file_path:str):
        # data process
        pbar = tqdm(total=len(adapter))
        results = []
        with jsonlines.open(output_file_path, 'w') as writer:
            for i in range(len(adapter)):
                item, meta_item = adapter[i]
                output, answer = self.run_single(**item)
                result = dict(
                    model_output=output,
                    model_answer=answer
                )
                print(result)
                result.update(meta_item)
                results.append(result)
                pbar.update(1)
        # save file
        FileWriter.write_jsonl(results, output_file_path)
             
class EntityLinkingEvaluator(Evaluator):
    
    def __init__(
        self, 
        generator_config: ComponentConfig,
        **kwargs
    ):
        self.backend = OneGen(generator_config)
        assert "rules" in kwargs
        prefix_list: List[List[int]] = list()
        next_list: List[int] = list()
        for rule in kwargs['rules']:
            start_text, end_text = rule
            prefix_list.append(
                self.backend.generator_tokenizer(start_text)['input_ids']
            )
            next_list.append(
                self.backend.generator_tokenizer.convert_tokens_to_ids(end_text)
            )
        self.logits_processor = LogitsProcessorList()
        self.logits_processor.append(
            AutoAddLogitsProcessor(rules=prefix_list, nexts=next_list)
        )

    def run_single(
        self,
        prompt:str,
        candidate_list:Union[List[str], List[List[str]]]=None, 
        meta_candidate_list:Union[List[str], List[List[str]]]=None,
        max_new_tokens:int=1024,
        input_ids=None,
        generation_config: GenerationConfig=DEFAULT_GENERATION_CONFIG,
        embed_batch_size:int=16,
        doc_embedding:torch.Tensor=None,
        doc_embedding_label:List[str]=None,
        sentence_connector:str="",
        max_retrieval_cnt:int=MAX_RETRIEVAL_CNT,
        skip_repr_token_cnt:int=0,
        use_faiss:bool=False,
    ):
        # 1. generating when encounter eos token
        output:dict = self.backend.generate(
            generation_config=generation_config,
            max_new_tokens=max_new_tokens,
            prompt=prompt,
            input_ids=None,
            past_key_values=None,
            logits_processor=self.logits_processor
        )

        # 2. get the hidden state corresponding to the special token
        hidden_state_in_last_layer:list = [] 
        for token_hidden_state_in_layer in output['hidden_states']:
            last_hidden_state = token_hidden_state_in_layer[-1]
            hidden_state_in_last_layer.append(last_hidden_state)
        # [n_beams, n_token, dim] ~~~ [1, n_tokens, dim]
        hidden_state_in_last_layer:torch.Tensor = torch.cat(hidden_state_in_last_layer, dim=1)         
        assert len(hidden_state_in_last_layer) == 1, "We now only support the situation of n_beams=1."

        # 3. locate the special token
        repr_token_id_list:List[int] = self.backend.generator_config.repr_token_id_list
        token_seq = output['token_seq']
        row_index = []
        column_index = []
        for n_beam in range(token_seq.shape[0]):
            for n_token in range(token_seq.shape[1]):
                token_id = token_seq[n_beam, n_token]
                if token_id in repr_token_id_list:
                    row_index.append(n_beam)
                    column_index.append(n_token)
        row_index = torch.as_tensor(row_index)
        column_index = torch.as_tensor(column_index)
        
        # 4. get the embedding
        embedding:torch.Tensor = None
        if len(row_index) != 0:
            embedding = hidden_state_in_last_layer[row_index, column_index, :]
            if len(embedding.shape) == 3:
                embedding = embedding.squeeze(dim=0)        # [n_repr, dim]

        # 5. generate doc embedding
        if doc_embedding == None:
            assert candidate_list != None
            # [n_doc, dim]
            doc_embedding = self.backend.encode(
                texts=candidate_list,
                batch_size=embed_batch_size,
                sentence_connector=sentence_connector,
                skip_repr_token_cnt=skip_repr_token_cnt
            )

        # 6. calculate similarity
        if use_faiss:
            # use faiss when retrieve
            arg_max_index = faiss_sim_matrix(embedding, doc_embedding)
        else:
            # without faiss
            scores = sim_matrix(embedding, doc_embedding)    # [n_repr_, n_doc]
            arg_max_index = torch.argmax(scores, dim=1)    # [n_repr]

        return {
            "output": output['increment'],
            "output_qid": [doc_embedding_label[index] for index in arg_max_index]
        }

    def run(self, adapter, output_file_path:str):
        # self.pre_cache?
        # if yes, we will first execute the embedding for db
        if adapter.file_config.pre_cache:
            _print("Execute embedding ...")
            db_embedding = self.backend.encode(
                texts=adapter.db_data,
                batch_size=adapter.inference_config.embed_batch_size,
                skip_repr_token_cnt=adapter.inference_config.skip_repr_token_cnt
            )
            _print("Embedding done!")
            FileWriter.write_pickle((db_embedding, adapter.db_label), adapter.file_config.db_cache_embedding_path)
            
            # then we need notify the adapter that we have got the embedding data
            adapter.file_config.db_embedding = db_embedding
            adapter.file_config.db_label = adapter.db_label
            adapter.convert()
        
        # finally, we can do inference for test dataset
        pbar = tqdm(total=len(adapter))
        results:List[Dict] = list()
        for i in range(len(adapter)):   
            item, meta_item = adapter[i]
            pbar.update(1)
            result = self.run_single(**item)
            result.update(meta_item)
            print(result)
            results.append(result)
        pbar.close()

        # save file
        FileWriter.write_jsonl(results, output_file_path)

class MentionDetectionEvaluator(EntityLinkingEvaluator):
    
    def __init__(
        self, 
        generator_config: ComponentConfig,
        **kwargs
    ):
        super().__init__(generator_config, **kwargs)
    
    def run_single(
        self,
        prompt:str,
        candidate_list:Union[List[str], List[List[str]]]=None, 
        meta_candidate_list:Union[List[str], List[List[str]]]=None,
        max_new_tokens:int=1024,
        input_ids=None,
        generation_config: GenerationConfig=DEFAULT_GENERATION_CONFIG,
        embed_batch_size:int=16,
        doc_embedding:torch.Tensor=None,
        doc_embedding_label:List[str]=None,
        sentence_connector:str="",
        max_retrieval_cnt:int=MAX_RETRIEVAL_CNT,
        skip_repr_token_cnt:int=0
    ):
        # we don't need do retrieval for mention detection task.
        # 1. generating when encounter eos token
        output:dict = self.backend.generate(
            generation_config=generation_config,
            max_new_tokens=max_new_tokens,
            prompt=prompt,
            input_ids=None,
            past_key_values=None,
            logits_processor=self.logits_processor
        )

        return {
            "output": output['increment'],
            "output_qid": [doc_embedding_label[index] for index in arg_max_index]
        }
    
class EntityDisambiguationEvaluator(EntityLinkingEvaluator):
    def __init__(
        self, 
        generator_config: ComponentConfig,
        **kwargs
    ):
        super().__init__(generator_config, **kwargs)

    def run_single(
        self,
        prompt:str,
        candidate_list:Union[List[str], List[List[str]]]=None, 
        meta_candidate_list:Union[List[str], List[List[str]]]=None,
        max_new_tokens:int=1024,
        input_ids=None,
        generation_config: GenerationConfig=DEFAULT_GENERATION_CONFIG,
        embed_batch_size:int=16,
        doc_embedding:torch.Tensor=None,
        doc_embedding_label:List[str]=None,
        sentence_connector:str="",
        max_retrieval_cnt:int=MAX_RETRIEVAL_CNT,
        skip_repr_token_cnt:int=0,
        use_faiss:bool=False,
    ):
        # We don't need to do generate for entity disambiguation task.
        # 1. prefill the prompt to get the hidden_state
        output = dict(
            token_seq=torch.as_tensor([self.backend.generator_tokenizer(prompt)['input_ids']]), 
            hidden_states=None, 
            increment=""
        )
        _output = self.backend.generator.model(
            input_ids=output['token_seq'],
            output_hidden_states=True,
        )
        output['last_hidden_states'] = _output[0]

        # 2. get the hidden state corresponding to the special token
        # [1, n_tokens, dim] 
        hidden_state_in_last_layer:torch.Tensor = output['last_hidden_states']
        assert len(hidden_state_in_last_layer) == 1, "We now only support the situation of n_beams=1."

        # 3. locate the special token
        repr_token_id_list:List[int] = self.backend.generator_config.repr_token_id_list
        token_seq = output['token_seq']
        row_index = []
        column_index = []
        for n_beam in range(token_seq.shape[0]):
            for n_token in range(token_seq.shape[1]):
                token_id = token_seq[n_beam, n_token]
                if token_id in repr_token_id_list:
                    row_index.append(n_beam)
                    column_index.append(n_token)
        row_index = torch.as_tensor(row_index)
        column_index = torch.as_tensor(column_index)
        
        # 4. get the embedding
        embedding:torch.Tensor = None
        if len(row_index) != 0:
            embedding = hidden_state_in_last_layer[row_index, column_index, :]
            if len(embedding.shape) == 3:
                embedding = embedding.squeeze(dim=0)        # [n_repr, dim]

        # 5. generate doc embedding
        if doc_embedding == None:
            assert candidate_list != None
            # [n_doc, dim]
            doc_embedding = self.backend.encode(
                texts=candidate_list,
                batch_size=embed_batch_size,
                sentence_connector=sentence_connector,
                skip_repr_token_cnt=skip_repr_token_cnt
            )

        # 6. calculate similarity
        if use_faiss:
            # use faiss when retrieve
            arg_max_index = faiss_sim_matrix(embedding, doc_embedding)
        else: 
            # without faiss
            scores = sim_matrix(embedding, doc_embedding)    # [n_repr_, n_doc]
            arg_max_index = torch.argmax(scores, dim=1)    # [n_repr]

        return {
            "output": output['increment'],
            "output_qid": [doc_embedding_label[index] for index in arg_max_index]
        }
    

if __name__ == '__main__':
    pass