#!/bin/bash

# ground truth file path
hotpot_gold_path="../data/eval_data/multi_hop_qa/hotpotqa/hotpot_dev_distractor_v1.json"
wiki_gold_path="../data/eval_data/multi_hop_qa/2wiki/dev.json"
wiki_id_alias_path="../data/eval_data/multi_hop_qa/2wiki/id_aliases.json"

if [ "$#" -ne 2 ]; then
    echo "Usage: \$0 file_path dataset_name"
    exit 1
fi

FILE_PATH="$1"
DATASET="$2"

if [ "$DATASET" != "hotpotqa" ] && [ "$DATASET" != "2wiki" ]; then
    echo "The dataset_name must be 'hotpotqa' or '2wiki'"
    exit 1
fi

if [ "$DATASET" == "hotpotqa" ]; then
    python multi_hop_qa/eval_hotpotqa.py $FILE $hotpot_gold_path
elif [ "$DATASET" == "2wiki" ]; then
    python run_2wiki.py  $FILE $wiki_gold_path $wiki_id_alias_path
fi
