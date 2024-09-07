import sys
sys.path.append('../')
from onegen.templator import *
from onegen.util import FileReader, _print, sim_matrix, FileWriter
from onegen.util import DEFAULT_GENERATION_CONFIG, MAX_NEW_TOKENS, MAX_RETRIEVAL_CNT
from onegen.dataset import padding_input_ids
from onegen.evaluator import FileConfig, InferenceConfig, ComponentConfig
from onegen.evaluator import RAGEvaluator, EntityLinkingEvaluator
from onegen.evaluator import EntityLinkingAdapter, MultiHopQAAdapter

from typing import Dict, Tuple, Any, Union, List
from transformers import GenerationConfig
from transformers import AutoModel, AutoModelForCausalLM
import torch



def parse_eval_config(file_name:str) -> Tuple[ComponentConfig, ComponentConfig, FileConfig, InferenceConfig, Dict, str, Any, Any]:
    def parse_torch_dtype(dtype:str):
        if dtype.lower() in ["bfloat16", "bf16"]:
            return torch.bfloat16
        elif dtype.lower() in ["fp32", "float32"]:
            return torch.float32
        elif dtype.lower() in ["fp16", "float16"]:
            return torch.float16
        else:
            raise ValueError(f"Invalid torch_dtype `{torch_dtype}`.")
    
    meta_data:dict = FileReader.read_json(file_name)
    evaluator_class = eval(meta_data['evaluator']['evaluator_class'])
    adapter_class = eval(meta_data['adapter']['adapter_class'])

    generator_config = ComponentConfig(
        model_class=eval(meta_data["generator"]['model_class']),
        model_path=meta_data["generator"]['model_path'],
        tokenizer_path=meta_data["generator"]['tokenizer_path'],
        torch_dtype=parse_torch_dtype(meta_data["generator"]['torch_dtype']),
        special_token_list=meta_data["generator"]['special_token_list'],
        add_prefix_space=meta_data["generator"]['add_prefix_space'],
        add_eos_token=meta_data["generator"]['add_eos_token'],
        add_bos_token=meta_data["generator"]['add_bos_token'],
        padding_side=meta_data["generator"]['padding_side'],
        padding_token=meta_data["generator"]['padding_token'],
        concatenate_template=meta_data["generator"]['concatenate_template'],
        stop_token_list=meta_data["generator"]['stop_token_list'],
        repr_token_list=meta_data["generator"]['repr_token_list']
    )

    if meta_data['retriever'] == None:
        retriever_config = None
    else:
        retriever_config = ComponentConfig(
            model_class=eval(meta_data["retriever"]['model_class']),
            model_path=meta_data["retriever"]['model_path'],
            tokenizer_path=meta_data["retriever"]['tokenizer_path'],
            torch_dtype=meta_data["retriever"],
            special_token_list=parse_torch_dtype(meta_data["generator"]['torch_dtype']),
            add_prefix_space=meta_data["retriever"]['add_prefix_space'],
            add_eos_token=meta_data["retriever"]['add_eos_token'],
            add_bos_token=meta_data["retriever"]['add_bos_token'],
            padding_side=meta_data["retriever"]['padding_side'],
            padding_token=meta_data["retriever"]['padding_token'],
            concatenate_template=meta_data["retriever"]['concatenate_template'],
            stop_token_list=meta_data["retriever"]['stop_token_list'],
            repr_token_list=meta_data["retriever"]['repr_token_list']
        )

    file_config = FileConfig(
        test_file_path = meta_data['file']['test']['file_path'],
        db_file_path = meta_data['file']['db']['file_path'],
        db_cache_embedding_path = meta_data['file']['db']['cache_file_path'],
        test_templator = None if meta_data['file']['test']['templator'] in [None, ""] else eval(meta_data['file']['test']['templator']),
        db_templator = None if meta_data['file']['db']['templator'] in [None, ""] else eval(meta_data['file']['db']['templator'])
    )

    inference_config = InferenceConfig(
        max_new_tokens=meta_data["inference"]['max_new_tokens'],
        generation_config=GenerationConfig(
            **meta_data['inference']['generation_config']
        ),
        embed_batch_size=meta_data["inference"]['embed_batch_size'],
        sentence_connector=meta_data["inference"]['sentence_connector'],
        max_retrieval_cnt=meta_data["inference"]['max_retrieval_cnt'],
        skip_repr_token_cnt=meta_data["inference"]['skip_repr_token_cnt']
    )

    if "other" in meta_data and isinstance(meta_data['other'], dict):
        kwargs = meta_data['other']
    else:
        kwargs = dict()

    return generator_config, retriever_config, file_config, inference_config, kwargs, meta_data['output_file_path'], evaluator_class, adapter_class
    