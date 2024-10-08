import sys
sys.path.append('src/')

import torch
import argparse
from transformers import TrainingArguments
from transformers import LlamaForCausalLM, Qwen2ForCausalLM, MistralForCausalLM

# from onegen import OneGenModel
from onegen import Tokenizer, OneGenTrainer
from onegen import create_onegen_model_class
from onegen.trainer import OneGenTensorBoardCallback
from onegen.config import parse_workflow, TrainingConfig, DataConfig, PaddingConfig, SpecialTokenConfig, OneGenConfig
from onegen.dataset import AutoDataCollator, AutoDataset
from onegen.util import FileReader, _print

def get_parser():
    parser = argparse.ArgumentParser(description="OneGen")
    parser.add_argument('--local_rank', type=int, help="just used for deepspeed.")
    parser.add_argument('--workflow', type=str, help="workflow file path")
    args = parser.parse_args()
    return args

def main():
    import deepspeed
    deepspeed.init_distributed()
    args = get_parser()
    training_config:TrainingConfig

    # Step 1. Load config
    training_config, data_train_config, data_db_config, \
        padding_config, special_token_config, onegen_config, resume_checkpoint_path = \
            parse_workflow(args.workflow)
    
    # Step 2. Load tokenizer
    tokenizer = Tokenizer(
        tokenizer_path=onegen_config.tokenizer_path,
        special_token_list=special_token_config.get_all_tokens(),
        add_prefix_space=False
    )
    tokenizer.save(f"{training_config.output_dir}")
    special_token_config.update_tokenizer(tokenizer)

    # Step 3. Load model
    # model = OneGenModel.from_pretrained(
    #     onegen_config.model_path, torch_dtype=torch.bfloat16
    # )
    model = create_onegen_model_class(eval(onegen_config.model_class)).from_pretrained(
        onegen_config.model_path, torch_dtype=torch.bfloat16
    )
    model.load_train_config(onegen_config=onegen_config)
    model.resize_and_initialize(tokenizer=tokenizer, special_token_config=special_token_config)
    
    # Step 4. Load dataset and data_collator
    train_dataset = AutoDataset(
        db_file_config=data_db_config,
        train_file_config=data_train_config,
        tokenizer=tokenizer
    )
    data_collator = AutoDataCollator(
        dataset=train_dataset, onegen_config=onegen_config,
        padding_config=padding_config
    )

    # Step 5. Load trainer
    trainer = OneGenTrainer(
        model=model,
        train_dataset=train_dataset,
        args=TrainingArguments(
            local_rank=args.local_rank,
            **training_config.to_dict()
        ),
        data_collator=data_collator,
    )
    trainer.add_callback(OneGenTensorBoardCallback())

    if isinstance(resume_checkpoint_path, str):
        if not FileReader.is_existed(resume_checkpoint_path):
            _print(f"The file `{resume_checkpoint_path}` for resume_checkpoint_path is not existed.")
            resume_checkpoint_path = None
        else:
            _print(f"resume from checkpoint `{resume_checkpoint_path}`")
    trainer.train(resume_from_checkpoint=resume_checkpoint_path) 
    trainer.save_model(training_config.output_dir)

if __name__ == '__main__':
    main()