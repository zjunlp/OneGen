#!/bin/bash

MODE="$1"
FILE_PATH="$2"

if [ "$MODE" != "el" ] && [ "$MODE" != "ed" ] && [ "$MODE" != "md" ]; then
    echo "The MODE must be 'el' (Entity Linking), 'ed' (Entity Disambiguation), or 'md' (Mention Detection)"
    exit 1
fi

if [ "$MODE" == "el" ]; then
    python entity_linking/eval_entity_linking.py  --file_path $FILE_PATH
elif [ "$MODE" == "ed" ]; then
    python entity_linking/eval_entity_disambiguation.py  --file_path $FILE_PATH
elif [ "$MODE" == "md" ]; then
    python entity_linking/eval_mention_detection.py --file_path $FILE_PATH
fi  