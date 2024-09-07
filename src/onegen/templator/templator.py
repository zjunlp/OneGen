# model template

import sys
sys.path.append('../')
from typing import List, Tuple, Any, Dict
from dataclasses import dataclass
import warnings

class Templator:

    @classmethod
    def wrap(self, messages:List, **kwargs) -> str:
        raise NotImplementedError("Please implement the function `wrap` for the class Templator.")
    
    @classmethod
    def generate_structured_input(
        cls, 
        messages: List[Dict],
        system_template:str,
        user_template:str,
        assistant_template:str,
        assistant_template_left:str,
        assistant_template_right:str,
        splitter:str,
        splitter_after_user:str=None,
        final_input:str = "",
        structured_final_input: List[str] = [""],
        # just for the inference
        add_special_token_when_last_role_is_user:bool = True,
    ) -> List[str]:
        for message in messages:
            # Our goal is to get the some part segament that can be concatenated directly and to mask some part segament
            if message['role'] == 'system':
                if system_template == None:
                    warnings.warn("The current templator is not supported the role `system`. Now we ignore the content in the system role.")
                else:
                    final_input = f"{final_input}{system_template.format(prompt=message['content'])}{splitter}"
                    structured_final_input[-1] = f"{structured_final_input[-1]}{system_template.format(prompt=message['content'])}{splitter}"
            elif message['role'] == 'user':
                if splitter_after_user == None:
                    final_input = f"{final_input}{user_template.format(prompt=message['content'])}{splitter}"
                    structured_final_input[-1] = f"{structured_final_input[-1]}{user_template.format(prompt=message['content'])}{splitter}"
                else:
                    final_input = f"{final_input}{user_template.format(prompt=message['content'])}{splitter_after_user}"
                    structured_final_input[-1] = f"{structured_final_input[-1]}{user_template.format(prompt=message['content'])}{splitter_after_user}"
            elif message['role'] == 'assistant':
                final_input = f"{final_input}{assistant_template.format(prompt=message['content'])}{splitter}"
                structured_final_input[-1] = f"{structured_final_input[-1]}{assistant_template_left}"
                structured_final_input.append(message['content']+assistant_template_right)
                structured_final_input.append(f"{splitter}")
            else:
                raise ValueError(f"the role `{message['role']}` is not supported. Our supported role list is `[system, user, assistant]`.")
        if len(splitter) > 0:
            # remove the last splitter
            assert final_input.endswith(splitter)
            final_input = final_input[:-len(splitter)]
            assert structured_final_input[-1].endswith(splitter)
            structured_final_input[-1] = structured_final_input[-1][:-len(splitter)]
        if add_special_token_when_last_role_is_user and messages[-1]['role'] == 'user':
            structured_final_input[-1] = f"{structured_final_input[-1]}{splitter}{assistant_template_left}"
            final_input = f"{final_input}{splitter}{assistant_template_left}"
        assert final_input == "".join(structured_final_input)
        return structured_final_input

class Qwen2Templator(Templator):
    # no explicit system prompt
    """<|im_start|>system
You are a helpful assistant<|im_end|>
<|im_start|>user
user input 1<|im_end|>
<|im_start|>assistant
model output 1<|im_end|>"""
    # explicit system prompt
    """<|im_start|>system
system prompt 1<|im_end|>
<|im_start|>user
user input 1<|im_end|>
<|im_start|>assistant
model output 1<|im_end|>"""
    @classmethod
    def wrap(cls, messages:List[Dict], add_special_token_when_last_role_is_user:bool=False, force_system_prompt:bool=False) -> List[str]:
        # no bos and no eos
        default_system_prompt = "You are a helpful assistant"
        system_template = "<|im_start|>system\n{prompt}<|im_end|>"
        user_template = "<|im_start|>user\n{prompt}<|im_end|>"
        assistant_template = "<|im_start|>assistant\n{prompt}<|im_end|>"
        assistant_template_left = "<|im_start|>assistant\n"
        assistant_template_right = "<|im_end|>"
        splitter = "\n"

        if force_system_prompt:
            is_existed = False
            for message in messages:
                if message['role'] == 'system':
                    is_existed = True
                    break
            if is_existed == False:
                messages = [{"role": "system", "content": default_system_prompt}] + messages

        return cls.generate_structured_input(
            messages=messages,
            system_template=system_template,
            user_template=user_template,
            assistant_template=assistant_template,
            assistant_template_left=assistant_template_left,
            assistant_template_right=assistant_template_right,
            splitter=splitter,
            add_special_token_when_last_role_is_user=add_special_token_when_last_role_is_user
        )
        
class Llama2Templator(Templator):
    # no explicit system prompt
    """<s>[INST] user input 1 [/INST] model output 1 </s>"""
    # explicit system prompt
    """<s>[INST] <<SYS>>
system prompt 1
<</SYS>>

user input 1 [/INST] model output 1 </s><s>[INST] user input 2 [/INST] model output 2 </s>"""
    
    @classmethod
    def wrap(cls, messages:List, add_special_token_when_last_role_is_user:bool=False) -> List[str]:
        # multi-round is not supported by official implementation
        default_system_prompt = None
        # system_template = "<<SYS>>\n{prompt}\n<</SYS>>\n\n"
        # user_template = " <s>[INST] {prompt} [/INST]"
        # assistant_template = "{prompt} </s>"   # ` {prompt} </s>`
        # assistant_template_left = ""           # ` `
        # assistant_template_right = " </s>"
        # splitter = ""
        system_template = "<<SYS>>\n{prompt}\n<</SYS>>\n\n"
        user_template = "<s>[INST] {prompt}"
        assistant_template = "[/INST]{prompt} </s>"    # `[/INST] {prompt} </s>` -> `[/INST]{prompt}</s>`
        assistant_template_left = "[/INST]"           # `[/INST] ` -> `[/INST]`
        assistant_template_right = " </s>"             # ` </s>` -> `</s>`
        splitter = "\n"
        splitter_after_user = ""
        # if system role is shown in messages, then we first make the system_template; we finally concatenate the system_template and user_prompt to input the user_template
        final_input:str = ""
        structured_final_input: List = [""]

        # check if the system role in the messages
        if messages[0]['role'] == 'system':
            assert messages[1]['role'] == 'user'
            final_input = user_template.format(
                prompt=system_template.format(prompt=messages[0]['content']) + messages[1]['content']
            )
            structured_final_input[-1] = final_input
            messages = messages[2:]
        
        structured_input = cls.generate_structured_input(
            messages=messages,
            system_template=system_template,
            user_template=user_template,
            assistant_template=assistant_template,
            assistant_template_left=assistant_template_left,
            assistant_template_right=assistant_template_right,
            splitter=splitter,
            splitter_after_user=splitter_after_user,
            final_input=final_input,
            structured_final_input=structured_final_input,
            add_special_token_when_last_role_is_user=add_special_token_when_last_role_is_user
        )
        return structured_input

class Llama3Templator(Templator):
    # no explicit system prompt
    """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

user input 1<|eot_id|><|start_header_id|>assistant<|end_header_id|>

model output 1<|eot_id|>"""
    # explicit system prompt
    """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

system prompt 1<|eot_id|><|start_header_id|>user<|end_header_id|>

user input 1<|eot_id|><|start_header_id|>assistant<|end_header_id|>

model output 1<|eot_id|>"""
    @classmethod
    def wrap(cls, messages:List, add_special_token_when_last_role_is_user:bool=False) -> List[str]:
        default_system_prompt = None
        system_template = "<|start_header_id|>system<|end_header_id|>\n\n{prompt}<|eot_id|>"
        user_template = "<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>"
        assistant_template = "<|start_header_id|>assistant<|end_header_id|>\n\n{prompt}<|eot_id|>"
        assistant_template_left = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        assistant_template_right = "<|eot_id|>"
        splitter = ""

        return cls.generate_structured_input(
            messages=messages,
            system_template=system_template,
            user_template=user_template,
            assistant_template=assistant_template,
            assistant_template_left=assistant_template_left,
            assistant_template_right=assistant_template_right,
            splitter=splitter,
            final_input = "<|begin_of_text|>",
            structured_final_input = ["<|begin_of_text|>"],
            add_special_token_when_last_role_is_user=add_special_token_when_last_role_is_user
        )


class MistralTemplator(Templator):
    # no explicit system prompt
    """<s>[INST] user input 1[/INST] model output 1</s>"""
    # explicit system prompt
    """<s>[INST] user input 1[/INST] model output 1</s>"""
    @classmethod
    def wrap(cls, messages:List, add_special_token_when_last_role_is_user:bool=False) -> List[str]:
        default_system_prompt = None
        system_template = None
        user_template = "[INST] {prompt}"
        assistant_template = "[/INST]{prompt}</s>"  # `[/INST] {prompt}</s>`
        assistant_template_left = "[/INST]"         # `[/INST] `
        assistant_template_right = "</s>"
        splitter = ""
        return cls.generate_structured_input(
            messages=messages,
            system_template=system_template,
            user_template=user_template,
            assistant_template=assistant_template,
            assistant_template_left=assistant_template_left,
            assistant_template_right=assistant_template_right,
            splitter=splitter,
            final_input = "<s>",
            structured_final_input = ["<s>"],
            add_special_token_when_last_role_is_user=add_special_token_when_last_role_is_user
        )

class GemmaTemplator(Templator):
    # no explicit system prompt
    """<bos><start_of_turn>user
user input 1<end_of_turn>
<start_of_turn>model
model output 1<end_of_turn>
<start_of_turn>user
user input 2<end_of_turn>
<start_of_turn>model
model output 2<end_of_turn>"""
    # explicit system prompt
    # system role not supported
    @classmethod
    def wrap(cls, messages:List) -> List[str]:
        # system role is not supported by official
        default_system_prompt = None
        system_template = "<start_of_turn>system\n{prompt}<end_of_turn>"
        user_template = "<start_of_turn>user\n{prompt}<end_of_turn>"
        assistant_template = "<start_of_turn>model\n{prompt}<end_of_turn>"
        assistant_template_left = "<start_of_turn>model\n"
        assistant_template_right = "<end_of_turn>"
        splitter = "\n"
        return cls.generate_structured_input(
            messages=messages,
            system_template=system_template,
            user_template=user_template,
            assistant_template=assistant_template,
            assistant_template_left=assistant_template_left,
            assistant_template_right=assistant_template_right,
            splitter=splitter,
            final_input = "<bos>",
            structured_final_input = ["<bos>"],
        )

class DocumentTemplator(Templator):

    @classmethod
    def wrap(cls, messages:List, **kwargs) -> List[str]:
        # [user, model, user, model]
        # Step 1. Check whether the user and assistant interweave
        for idx, message in enumerate(messages):
            if idx % 2 == 0:
                assert message['role'] == 'user'
            else:
                assert message['role'] == 'assistant'
        
        # Step 2. Construct output for tokenize
        results = []
        for idx, message in enumerate(messages):
            if idx % 2 == 0:
                results.append(message['content'])
            else:
                results.append(message['content'])
        return results

class SelfRAG_LLama2Templator(Templator):

    @classmethod
    def wrap(cls, messages:List) -> List[str]:
        assert len(messages) == 2
        assert messages[0]['role'] == 'assistant'
        assert messages[1]['role'] == 'user'
        system_template = None
        user_template = "### Instruction:\n{prompt}"
        assistant_template = "\n\n### Response:\n{prompt}</s>"  
        assistant_template_left = "\n\n### Response:\n"        
        assistant_template_right = "</s>"
        splitter = ""
        return cls.generate_structured_input(
            messages=messages,
            system_template=system_template,
            user_template=user_template,
            assistant_template=assistant_template,
            assistant_template_left=assistant_template_left,
            assistant_template_right=assistant_template_right,
            splitter=splitter,
            final_input = "",
            structured_final_input = [""],
        )

class SelfRAG_Llama2_DocTemplator(Templator):
    # TODO: check!
    @classmethod
    def wrap(cls, messages:List) -> List[str]:
        assert len(messages) == 2
        assert messages[0]['role'] == 'assistant'
        assert messages[1]['role'] == 'user'
        system_template = None
        user_template = "### Instruction:\n{prompt}"
        assistant_template = "\n\n### Response:\n{prompt}</s>"  
        assistant_template_left = "\n\n### Response:\n"        
        assistant_template_right = "</s>"
        splitter = ""
        return cls.generate_structured_input(
            messages=messages,
            system_template=system_template,
            user_template=user_template,
            assistant_template=assistant_template,
            assistant_template_left=assistant_template_left,
            assistant_template_right=assistant_template_right,
            splitter=splitter,
            final_input = "",
            structured_final_input = [""],
        )

class EntityLinking_Llama2Templator(Templator):
    @classmethod
    def wrap(cls, messages:List) -> List[str]:
        assert len(messages) == 2
        assert messages[0]['role'] == 'user'
        assert messages[1]['role'] == 'assistant'
        return [messages[0]['content'], " "+messages[1]['content']]




if __name__ == '__main__':
    
    # TODO: check len(messages) == len(output)

    from transformers import AutoTokenizer
    data = [
        {'role': 'system', 'content': 'system prompt 1'},
        {'role': 'user', 'content': 'user input 1'},
        {'role': 'assistant', 'content': 'model output 1'},
        {'role': 'user', 'content': 'user input 2'},
        {'role': 'assistant', 'content': 'model output 2'},
    ]


    tokenizer = AutoTokenizer.from_pretrained("/disk/disk_20T/share/Llama-3-8B-Instruct", add_prefix_space=False)
    print(Llama3Templator.wrap(data))
    exit()

    tokenizer_check_list = [
        ["/disk/disk_20T/share/Llama-3-8B-Instruct", Llama3Templator, {"messages": data}],
        # ["/disk/disk_20T/share/Qwen2-7B", Qwen2Templator, {"messages": data, "force_system_prompt": True}],
        # ["/disk/disk_20T/qiaoshuofei/PLMs/llama-2-7b-chat", Llama2Templator, {"messages": data}],
        # ["mistralai/Mistral-7B-Instruct-v0.3", MistralTemplator, {"messages": data}],
        # ["google/gemma-2-9b-it", GemmaTemplator, {"messages": data}]
    ]

    for tokenizer_path, templator, args in tokenizer_check_list:
        print(f"checking {str(templator)} ...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, add_prefix_space=False)
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False
        output_from_tokenizer = tokenizer.apply_chat_template(data, tokenize=False).strip()
        output_list_from_templator = templator.wrap(**args)
        output_from_templator = "".join(output_list_from_templator)
        tokenized_from_tokenizer = tokenizer(output_from_templator, return_tensors=None, padding=False, truncation=True, max_length=1024)['input_ids']
        tokenized_from_templator = []
        # print(output_from_templator)
        # print("origin:\n",tokenizer.tokenize(output_from_templator))
        for segament in output_list_from_templator:
            if len(segament) > 0:
                tokenized_from_templator.extend(
                    tokenizer(segament, return_tensors=None, padding=False, truncation=True, max_length=1024)['input_ids']
                )
                # print(tokenizer.tokenize(segament))
        # input("wait:")
        # print(output_list_from_templator)
        # print(tokenized_from_tokenizer)
        # print(tokenized_from_templator)
        # input()
        if len(tokenized_from_templator) != len(tokenized_from_tokenizer):
            print("not equal!")
        else:
            print("equal!")
        # if output_from_templator != output_from_tokenizer:
        #     print(f"tokenizer:\n#{output_from_tokenizer}#\n\ntemplator:\n#{output_from_templator}#")
        # print(f"#{output_from_templator}#")
