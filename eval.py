import sys
sys.path.append('src/')

import torch
import argparse
from transformers import TrainingArguments

from onegen.config import parse_eval_config, TrainingConfig, DataConfig, PaddingConfig, SpecialTokenConfig, OneGenConfig
from onegen.util import _print

def get_parser():
    parser = argparse.ArgumentParser(description="OneGen")
    parser.add_argument('--config', type=str, help="evaluation config")
    args = parser.parse_args()
    return args

def main():
    import deepspeed
    deepspeed.init_distributed()
    args = get_parser()

    # Step 1. Load config
    generator_config, retriever_config, file_config, inference_config, kwargs, output_file_path, \
        evaluator_class, adapter_class = parse_eval_config(args.config)
    
    # Step 2. Load evaluator and adapter
    evaluator = evaluator_class(generator_config=generator_config, retriever_config=retriever_config, **kwargs)
    adapter = adapter_class(
        file_config=file_config,
        inference_config=inference_config,
        **kwargs
    )

    # Step 3. Run
    evaluator.run(adapter, output_file_path=output_file_path)

    # Step 4. Metric

if __name__ == '__main__':
    main()