from .dataset import AutoDataset
from config import OneGenConfig
from .dataset_utils import padding_item
from typing import Tuple, Dict, List
import random
import torch

class BaseDataCollator:
    def __init__(self):
        raise NotImplementedError()

    def __call__(self):
        raise NotImplementedError()

    def random_select(self):
        pass

class AutoDataCollator:
    def __init__(
        self,
        dataset:AutoDataset,
        onegen_config: OneGenConfig,
        padding_config: PaddingConfig,
        random_n: int=100,
    ):
        """
        args:
            - random_n: if there are not sufficient negative examples, 
                we will random select negative examples for making the number reach the onegen_config.n_neg_per_pos
        """
        self.dataset: AutoDataset = dataset
        self.onegen_config: OneGenConfig = onegen_config
        self.random_n: int = random_n
        self.padding_config: PaddingConfig = padding_config

    def random_select(
        self, 
        positive:List[List],        # [[uid,uid], [uid]]
        negative:List[List],        # [[uid,uid], [uid]]
        embedding_index:List,       # [pos1, pos2]
    ) -> Tuple:
        # selection is only happened for an instance

        # select positive for each special token
        selected_positive_uid = []
        for pos_candidate in positive:
            assert isinstance(pos_candidate, list)
            if len(pos_candidate) == 0:
                selected_positive_uid.append(None)
            else:
                selected_positive_uid.append(
                    random.sample(pos_candidate, 1)[0]
                )

        # control the range
        all_idx = list(range(len(embedding_index)))
        valid_idx = []
        for idx in all_idx:
            if negative[idx] != None and  \
            len(negative[idx]) > 0 and \
            positive[idx] != None and \
            self.dataset.get_doc_id_for(selected_positive_uid[idx]) in self.dataset.db_data:
                valid_idx.append(idx)

        if len(valid_idx) == 0:
            # this instance will be classified as generation case.
            return None, None
        else:
            # this instance will be served as anchor/hybrid case.
            if len(valid_idx) > self.onegen_config.n_pos_per_sent:
                selected_positive_index: List[int] = sorted(
                    random.sample(valid_idx, self.onegen_config.n_pos_per_sent)
                )
            else:
                selected_positive_index:List[int] = valid_idx
            selected_positive_uid: List[str] = [
                selected_positive_uid[idx] for idx in selected_positive_index
            ]
            # the selected_positive_index is the index of selected positive examples in the `positive`
            return selected_positive_uid, selected_positive_index

    
    def __call__(
        self, 
        instances: List[Tuple],
    ):
        """
        for each instance in instances:
            instance: Tuple[int, List[List], List[List]]
            instance = [
                (
                    idx,                                    # index in the AutoDataset
                    positive_id: [[uid, uid], [uid, uid], ...],  # positive index
                    corresponding_neg_qid: [
                        [uid, uid, uid, uid, uid], 
                        [uid, uid, uid, uid, uid],
                        ...
                    ]
                ),
                ...
            ]
        
        Packing Sequence:
            Generation Only (GEN+CTX)
            Hybrid/Anchor (RET+GEN+CTX)
            Positive (RET+CTX)
            Negative (RET+CTX)
        if there are no positive examples for a case from Hybrid, this case will be converted to Generation-Only

        Our goal is:
            Step 1. Make sure there are no overlap for negative in a batch and there are no overlap between the positive and negative.
            Step 2. Classify the each instance in instances. Generation Only or Hybrid
            Step 3. Get index and flag
            Step 4. padding
            Step 5. Return
        """

        generation_info = {
            "id_in_dataset": [],        # instance index in the all dataset
            "input_ids": [],
            "labels": [],
            "meta_info": []
        }

        positive_info = {
            "input_ids": [],
            "labels": [],
            "row_index": [],
            "column_index": [],
            "meta_info": [],
            "loc_id": [],
            "index_meta_doc_id": []
        }

        negative_info = {
            "input_ids": [],
            "labels": [],
            "row_index": [],
            "column_index": [],
            "meta_info": [],
            "loc_id": [],
            "index_meta_doc_id": []
        }

        anchor_info = {
            "id_in_dataset": [],
            "selected_pos_uid": [],
            "selected_pos_index_in_pos_set": [],    # [bs, n_pos_per_sent]
            "corresponding_neg_uid_list": [],
            "meta_info": [],
            "input_ids": [],
            "labels": [],
            "row_index": [],
            "column_index": []
        }

        # the start offset of positive example
        # len(generation) + len(hybrid/anchor)
        positive_start_row_index:int = len(instances)

        # the start offset of negative example is unknown.
        # because once some anchor part is converted to the generation
        # the quantity of positive examples will reduce.
        negative_start_row_index:int = None

        # ====================================================
        # Step 1. Sample positive and negative examples;
        #         Classify instances to generation and anchor;
        # ====================================================
        for item in instances:
            idx, pos_id_list, corresponding_neg_id_list = item
            corresponding_embedding_index:List[int] = \
                self.dataset.get_tokenized_info_for_train_data(idx)['embedding_index']
            # selected_positive_uid, selected_positive_index
            selected_pos_uid_list, selected_pos_index_in_pos_set_list = self.random_select(
                pos_id_list, corresponding_neg_id_list, corresponding_embedding_index
            )
            if selected_pos_uid_list == None:
                # Generation-only case
                generation_info['id_in_dataset'].append(idx)
            else:
                # Anchor case
                anchor_info['id_in_dataset'].append(idx)
                # [uid, uid]
                anchor_info['selected_pos_uid'].append(
                    selected_pos_uid_list
                )
                # selected positive example in the positive list
                anchor_info['selected_pos_index_in_pos_set'].append(
                    selected_pos_index_in_pos_set_list
                )
                # [[uid, uid, uid], [uid, uid, uid]]
                anchor_info['corresponding_neg_uid_list'].append(
                    [corresponding_neg_id_list[_] for _ in selected_pos_index_in_pos_set_list]
                )

        # ====================================================
        # Step 2. Select the final negative example;
        #         There are no overlap between P&N, N&N.
        # ====================================================
        flatten_pos_uid:List[str] = []
        for pos_uid_list in anchor['selected_pos_uid']:
            # batch -> instance
            for pos_uid in pos_uid_list:
                # instance -> positive_list
                flatten_pos_uid.append(pos_uid)
        # cache the negative uid for filtering the repeated negative example
        flatten_neg_uid = []
        # [n_hybrid_examples, n_pos_per_sent, n_neg_per_pos]
        final_neg_uid:List = []
        for bs in range(len(anchor_info['id_in_dataset'])):
            final_neg_uid.append(list())
            for neg_uid_list in anchor_info['corresponding_neg_uid_list'][bs]:
                # neg_uid_list is the negative list corresponding to a positive example
                final_neg_uid[-1].append(list())
                # counter. if the counter's value reach to the n_neg_per_pos, then stop.
                cnt = 0
                for uid in neg_uid_list:
                    if uid not in flatten_neg_uid and uid in self.dataset.db_data and uid not in flatten_pos_uid:
                        # unique negative in the flatten_neg_uid
                        # this example has a description
                        # this example is not a positive
                        cnt += 1
                        final_neg_uid[-1][-1].append(uid)
                        flatten_neg_uid.append(uid)
                    if cnt >= self.onegen_config.n_neg_per_pos:
                        break
                # if the number of the negative examples doesn't reach the n_neg_per_pos
                # (current instance and current positve example). then we do random selection.
                while len(final_neg_uid[-1][-1]) < self.onegen_config.n_neg_per_pos:
                    # {doc_id}-{sent_id}
                    # TODO: the element in _random_uid is the form of `{doc_id}-{sent_id}`.
                    # we must make sure that all the uid is the form of `{doc_id}-{sent_id}`.
                    _random_uid:List[str] = self.dataset.get_random_uid_list(n=self.random_n)
                    for uid in _random_uid:
                        if uid not in flatten_neg_uid and uid not in flatten_pos_uid and self.data.get_doc_id_for(uid) in self.data.db_data:
                            cnt += 1
                            final_neg_uid[-1][-1].append(uid)
                            flatten_neg_uid.append(uid)
                        if cnt >= self.onegen_config.n_neg_per_pos:
                            break
                assert len(final_neg_uid[-1][-1]) == self.onegen_config.n_neg_per_pos
            assert len(final_neg_uid[-1]) == len(anchor_info['corresponding_neg_uid_list'][bs])
        assert len(final_neg_uid) == len(anchor_info['corresponding_neg_uid_list'])

        # ====================================================
        # Step 3. Find the tokenized item for each examples
        # ====================================================     
        # Handle the hybrid case  
        # 0-1, 0-2, so we just add 0.
        for i in range(len(anchor['id_in_dataset'])):
            idx_in_dataset = anchor_info['id_in_dataset'][i]
            anchor_info['meta_info'].append(
                self.dataset.get_tokenized_input(idx_in_dataset)
            )
            cur_pos_index_list = [] # [n_pos_per_sent]
            for uid in anchor_info['selected_pos_uid'][i]:
                doc_id:str = self.dataset.get_doc_id_for(uid)
                sent_id:int = self.dataset.get_sent_id_for(uid)
                cur_pos_index_list.append([doc_id, sent_id])
                if doc_id not in positive_info['index_meta_doc_id']:
                    positive_info['index_meta_doc_id'].append(doc_id)
                    positive_info['meta_info'].append(
                        self.dataset.get_tokenized_db(doc_id)
                    )
            positive_info['loc_id'].append(cur_pos_index_list)
            cur_neg_index_list = [] # [n_pos_per_sent, n_neg_per_pos]
            for uid_list in final_neg_uid[i]:
                cur_neg_index_list.append(list())
                for uid in uid_list:
                    doc_id:str = self.dataset.get_doc_id_for(uid)
                    sent_id:int = self.dataset.get_sent_id_for(uid)
                    cur_neg_index_list[-1].append([doc_id, sent_id])
                    if doc_id not in negative_info['index_meta_doc_id']:
                        negative_info['index_meta_doc_id'].append(doc_id)
                        negative_info['meta_info'].append(
                            self.dataset.get_tokenized_db(doc_id)
                        )
            negative_info['loc_id'].append(cur_neg_index_list)
        assert len(anchor_info['meta_info']) == len(anchor_info['selected_pos_index_in_pos_set'])
        
        # Handle the generation-only case
        for idx in generation_info['id_in_dataset']:
            generation_info['meta_info'].append(
                self.dataset.get_tokenized_input(idx)
            )
        assert len(anchor_info['meta_info']) + len(generation_info['meta_info']) == len(instances)

        # ====================================================
        # Step 4. Padding
        # ====================================================  
        # Determine the positive_start_row_index and negative_start_row_index
        positive_start_row_index = len(instances)
        pos_doc_id_set = set()
        for item in anchor_info['selected_pos_uid']:
            for pos_id in item:
                pos_doc_id_set.add(self.dataset.get_doc_id_for(pos_id))
        negative_start_row_index = len(instances) + len(pos_doc_id_set)
        assert len(pos_doc_id_set) == len(positive_info['info_meta_doc_id'])

        # Generation
        for idx, tokenized_item in enumerate(generation_info['meta_info']):
            new_item:dict = padding_item(
                item=tokenized_item,
                padding_side=self.padding_config.padding_side,
                label_padding_id=self.padding_config.padding_label_id,
                input_padding_id=self.padding_config.padding_input_id,
                max_length=self.padding_config.padding_max_length
            )
            generation_info['input_ids'].append(new_item['input_ids'])
            generation_info['labels'].append(new_item['labels'])
        
        # Hybrid
        # 1. Anchor
        for idx, tokenized_item in enumerate(anchor_info['meta_info']):
            new_item:dict = padding_item(
                item=tokenized_item,
                padding_side=self.padding_config.padding_side,
                label_padding_id=self.padding_config.padding_label_id,
                input_padding_id=self.padding_config.padding_input_id,
                max_length=self.padding_config.padding_max_length
            )
            anchor_info['input_ids'].append(new_item['input_ids'])
            anchor_info['labels'].append(new_item['labels'])
            anchor_info['row_index'].extend(
                [
                    idx+len(generation_info['meta_info']) \
                    for _ in range(len(anchor_info['selected_pos_index_in_pos_set'][idx]))
                ]
            )
            anchor_info['column_index'].extend(
                [
                    tokenized_item['embedding_index'][_] \
                    for _ in anchor_info['selected_pos_index_in_pos_set'][idx]
                ]
            )
        # 2. Positive
        # 2.1 Padding
        for idx in range(len(positive_info['meta_info'])):
            pos_uid_item = positive_info['meta_info'][idx]
            new_pos_uid_item:dict = padding_item(
                item=pos_uid_item,
                padding_side=self.padding_config.padding_side,
                label_padding_id=self.padding_config.padding_label_id,
                input_padding_id=self.padding_config.padding_input_id,
                max_length=self.padding_config.padding_max_length
            )
            positive_info['input_ids'].append(
                new_pos_uid_item['input_ids']
            )
            positive_info['labels'].append(
                new_pos_uid_item['labels']
            )
        # 2.2 handle index
        for idx_anchor in range(len(positive_info['loc_id'])):
            for idx_pos in range(len(positive_info['loc_id'][idx_anchor])):
                doc_id, sent_id = positive_info['loc_id'][idx_anchor][idx_pos]
                positive_info['row_index'].append(
                    positive_info['index_meta_doc_id'].index(doc_id) + positive_start_row_index
                )
                positive_info['column_index'].append(
                    positive_info['meta_info'][positive_info['index_meta_doc_id'].index(doc_id)]['embedding_index'][sent_id]
                )
        # 3. Negative
        # 3.1 Padding
        for idx in range(len(negative_info['meta_info'])):
            neg_uid_item = negative_info['meta_info'][idx]
            new_neg_uid_item:dict = padding_item(
                item=neg_uid_item,
                padding_side=self.padding_config.padding_side,
                label_padding_id=self.padding_config.padding_label_id,
                input_padding_id=self.padding_config.padding_input_id,
                max_length=self.padding_config.padding_max_length
            )
            negative_info['input_ids'].append(
                new_neg_uid_item['input_ids']
            )
            negative_info['labels'].append(
                new_neg_uid_item['labels']
            )
        # 3.2 handle index
        for idx_anchor in range(len(negative_info['loc_id'])):
            for idx_pos in range(len(negative_info['loc_id'][idx_anchor])):
                for idx_neg in range(len(negative_info['loc_id'][idx_anchor][idx_pos])):
                    doc_id, sent_id = negative_info['loc_id'][idx_anchor][idx_neg]
                    negative_info['row_index'].append(
                        negative_info['index_meta_doc_id'].index(doc_id) + negative_start_row_index
                    )
                    negative_info['column_index'].append(
                        negative_info['meta_info'][negative_info['index_meta_doc_id'].index(doc_id)]['embedding_index'][sent_id]
                    )
    
        # ====================================================
        # Step 5. Packing and Return
        # ====================================================  
        assert len(anchor_info['input_ids']) + len(generation_info['input_ids']) == len(instances)
        return dict(
            input_ids=torch.as_tensor(
                generation_info['input_ids'] + anchor_info['input_ids'] + positive_info['input_ids'] + negative_info['input_ids']
            ),
            labels=torch.as_tensor(
                generation_info['labels'] + anchor_info['labels'] + positive_info['labels'] + negative_info['labels']
            ),
            embedding_index=(
                torch.as_tensor(anchor_info['row_index'] + positive_info['row_index'] + negative_info['row_index']),
                torch.as_tensor(anchor_info['column_index'] + positive_info['column_index'], negative_info['column_index'])
            ),
            embedding_index_split_flag=[
                len(anchor_info['row_index']),
                len(positive_info['row_index']),
                len(negative_info['row_index']),
            ]
        )