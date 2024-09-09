# Here, we assume that expanded tokenizer is saved in the output_dir
import sys
sys.path.append('../../src')
import jsonlines
import json
from tqdm import tqdm 
import torch
import time
import fire
from transformers import MistralForCausalLM, LlamaForCausalLM
from transformers import AutoTokenizer
from onegen import OneGenModel
from onegen.util import _print
from typing import *
import os

class ConstantConfig:

    document_token_for_repr:str = '[Document Fragment Representation]'
    question_token_for_repr:str = '[Question Representation]'
    db_template:str = "Represent the following document fragment and output it after the '{special_token}' token. \nHere are document fragment:\n{input}{special_token}{output}"

    @classmethod
    def get_document_token_for_repr(cls) -> str:
        return cls.document_token_for_repr

    @classmethod
    def get_question_token_for_repr(cls) -> str:
        return cls.question_token_for_repr

    @classmethod
    def get_db_template(cls, tokenizer) -> :
        return cls.db_template.format(
            sepecial_token=cls.get_document_token_for_repr(),
            input="{input}",
            output="{output}"
        )

    @classmethod
    def get_question_token_id_for_repr(cls, tokenizer) -> int:
        token_id = tokenizer.convert_tokens_to_ids(
            cls.get_question_token_for_repr()
        )
        assert isinstance(token_id, int)
        return token_id

    @classmethod
    def get_document_token_id_for_repr(cls, tokenizer) -> int:
        token_id = tokenizer.convert_tokens_to_ids(
            cls.get_document_token_for_repr()
        )
        assert isinstance(token_id, int)
        return token_id



TASK_INST = {
    "wow": "Given a chat history separated by new lines, generates an informative, knowledgeable and engaging response. ",
    "fever": "Is the following statement correct or not? Say true if it's correct; otherwise say false.",
    "eli5": "Provide a paragraph-length response using simple words to answer the following question.",
    "obqa": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
    "arc_easy": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
    "arc_c": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
    "trex": "Given the input format 'Subject Entity [SEP] Relationship Type,' predict the target entity.",
    "asqa": "Answer the following question. The question may be ambiguous and have multiple correct answers, and in that case, you have to provide a long-form answer including all correct answers."
}


def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

def preprocess_input_data(item:str, task=None, prompt_template:str="### Instruction:\n{instruction}\n\n### Response:\n[Retrieval]"+ConstantConfig.get_question_token_for_repr()):
    if task in TASK_INST:
        instruction = TASK_INST[task]
    else:
        instruction = None
    if task == "arc_c":
        choices = item["choices"]
        answer_labels = {}
        for i in range(len(choices["label"])):
            answer_key = choices["label"][i]
            text = choices["text"][i]
            if answer_key == "1":
                answer_labels["A"] = text
            if answer_key == "2":
                answer_labels["B"] = text
            if answer_key == "3":
                answer_labels["C"] = text
            if answer_key == "4":
                answer_labels["D"] = text
            if answer_key in ["A", "B", "C", "D"]:
                answer_labels[answer_key] = text

        if "D" not in answer_labels:
            answer_labels["D"] = ""
        choices = "\nA: {0}\nB: {1}\nC: {2}\nD: {3}".format(
            answer_labels["A"], answer_labels["B"], answer_labels["C"], answer_labels["D"])
        if "E" in answer_labels:
            choices += "\nE: {}".format(answer_labels["E"])
        item["instruction"] = instruction + \
            "\n\n### Input:\n" + item["question"] + choices
        item["answers"] = [item["answerKey"]]
    else:
        prompt = instruction + "\n\n## Input:\n\n" + \
            item["question"] if instruction is not None else item["question"]
        item["instruction"] = prompt
    
    # print(prompt, item)
    try:
        return prompt_template.format_map(item)
    except:
        print(prompt)
        # print(item)
        assert False

def padding_item(item:dict, padding_side:str, input_padding_id:int, max_length:int):
    assert padding_side in ['left', 'right']
    copy_item = item
    remainder_for_input = [input_padding_id] * (max_length - len(copy_item['input_ids'])) 
    if padding_side == 'left':
        copy_item['input_ids'] = remainder_for_input + copy_item['input_ids']
    else:
        copy_item['input_ids'] = copy_item['input_ids'] + remainder_for_input
    return copy_item

def tokenize_and_collate(tokenizer, batch:List[str], embedding_token_id:int, last_token=False) -> dict:
    tokenized_items = []
    max_length = 0
    for item in batch:
        tokenized_item:dict = tokenizer(
            item, return_tensors=None, padding=False,
            truncation=False,
        )
        max_length = max(max_length, len(tokenized_item['input_ids']))
        assert embedding_token_id in tokenized_item['input_ids']
        embedding_index = []
        for position, token_id in enumerate(tokenized_item['input_ids']):
            if token_id == embedding_token_id:
                embedding_index.append(position)
        assert len(embedding_index) > 0
        if last_token == True:
            tokenized_item['embedding_index'] = embedding_index[-1]
        else:
            tokenized_item['embedding_index'] = embedding_index[0]
        # tokenized_item['embedding_index'] = tokenized_item['input_ids'].index(
        #     embedding_token_id
        #     # ConstantConfig.get_document_token_id_for_repr(tokenizer)
        # )
        tokenized_items.append(tokenized_item)
    input_ids = []
    row_index = []
    column_index = []
    for row_id, item in enumerate(tokenized_items):
        row_index.append(row_id)
        column_index.append(item['embedding_index'])
        input_ids.append(
            padding_item(
                item, padding_side='right', input_padding_id=tokenizer.pad_token_id,
                max_length=max_length
            )['input_ids']
        )
    try:
        return dict(
            input_ids=torch.as_tensor(input_ids).cuda(),
            embedding_index=[torch.as_tensor(row_index), torch.as_tensor(column_index)]
        )
    except:
        print(input_ids)
        assert False

def load_model_and_tokenizer(model_path:str, tokenizer_path:str=None):
    if tokenizer_path == None:
        tokenizer_path = model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.padding_side = 'right'
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_eos_token = False
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = OneGenModel.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    return model, tokenizer

def load_jsonl(dataset:str, tokenizer:AutoTokenizer, task=None)->dict:
    # uid: content db
    db_uid2content = {}

    # meta information
    meta_info = []

    with jsonlines.open(dataset, 'r') as reader:
        uid = 0
        for item in reader:
            item['prompt']:str = preprocess_input_data(
                item, task
            )
            meta_info.append(item)
            if "ctxs" in item:
                for evidence_item in item['ctxs']:
                    evidence_item['id'] = uid
                    uid += 1
                    if evidence_item['id'] in db_uid2content:
                        now_prompt = ConstantConfig.get_db_template(tokenizer).format(
                            input=f"{evidence_item['title']}\n{evidence_item['text']}",
                            output=""
                        )
                        assert now_prompt == db_uid2content[evidence_item['id']]
                    else:
                        db_uid2content[evidence_item['id']] = \
                            ConstantConfig.get_db_template(tokenizer).format(
                                input=f"{evidence_item['title']}\n{evidence_item['text']}",
                                output=""
                            )
                        
    return db_uid2content, meta_info

def main(
    model_path:str,
    output_folder:str,
    batch_size:int=64,
    datasets_folder:str='/gruntdata/event_graph/shanghua.zjt/rag/self-rag-main/retrieval_lm/eval_data',
    datasets:list=[
        'pop_sampled_10.jsonl',
        'popqa_longtail_w_gs.jsonl', 
        'triviaqa_test_w_gs.jsonl', 
        'health_claims_processed.jsonl', 
        'arc_challenge_processed.jsonl'
    ],
    tokenizer_path:str=None
):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    assert os.path.exists(output_folder)

    # tokenizer needs to add a special token
    model, tokenizer = load_model_and_tokenizer(
        model_path=model_path, tokenizer_path=tokenizer_path
    )  

    # db data
    database = {}
    # meta data & prompt
    meta_data = {}
    for dataset in datasets:
        if dataset == 'arc_challenge_processed.jsonl':
            task = 'arc_c'
        elif dataset == 'fever':
            task = 'health_claims_processed.jsonl'
        else:
            task = None
        dataset_path = f"{datasets_folder}/{dataset}"
        database[dataset], meta_data[dataset] = load_jsonl(dataset_path, tokenizer, task)


    for dataset in database:
        _print(f"embedding for {dataset} ...")
        batch = []
        final_index = []
        final_embedding = []
        output_file_name = f"{output_folder}/{dataset}"
        # 1. make the embedding of db
        pbar = tqdm(total=len(database[dataset]))
        for doc_id in database[dataset]:
            pbar.update(1)
            batch.append(database[dataset][doc_id])
            final_index.append(doc_id)
            if len(batch) == batch_size:
                input_for_model:dict = tokenize_and_collate(
                    tokenizer, batch, embedding_token_id=ConstantConfig.get_document_token_id_for_repr(tokenizer),
                    last_token=True
                )
                final_embedding.append(
                    model.get_embedding(**input_for_model)
                )
                batch.clear()
        pbar.close()
        if len(batch) != 0:
            input_for_model:dict = tokenize_and_collate(
                tokenizer, batch, embedding_token_id=ConstantConfig.get_document_token_id_for_repr(tokenizer),
                last_token=True
            )
            final_embedding.append(
                model.get_embedding(**input_for_model)
            )
            batch.clear()
        final_embedding = torch.cat(final_embedding, dim=0)
        assert len(final_embedding) == len(final_index)
        assert len(final_embedding) == len(database[dataset])

        # 2. make query
        pbar = tqdm(total=len(meta_data[dataset]))
        for i, item in enumerate(meta_data[dataset]):
            # get the embedding of query
            input_for_model:dict = tokenize_and_collate(
                tokenizer, batch=[item['prompt']], embedding_token_id=ConstantConfig.get_question_token_id_for_repr(tokenizer)
            )
            query_emb = \
                model.get_embedding(**input_for_model).cuda()
            # get the embedding of the corresponding candidate
            candidate_emb = []
            for evidence_item in item['ctxs']:
                doc_id = final_index.index(evidence_item['id'])
                candidate_emb.append(final_embedding[doc_id].unsqueeze(dim=0))
            candidate_emb = torch.cat(candidate_emb, dim=0)
            # print(candidate_emb)
            score = sim_matrix(query_emb, candidate_emb).squeeze().cpu()
            del query_emb
            assert len(score) == len(item['ctxs'])
            for idx, evidence_item in enumerate(item['ctxs']):
                # meta_data[dataset][i]['ctxs'][idx]
                evidence_item['my_score'] = score[idx].item()
            pbar.update(1)
        pbar.close()
        # 3. saving final data
        with jsonlines.open(output_file_name, 'w') as writer:
            for item in meta_data[dataset]:
                writer.write(item)
        del final_embedding
        torch.cuda.empty_cache()

if __name__ == '__main__':
    with torch.no_grad():
        fire.Fire(main)
