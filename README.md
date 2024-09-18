<div align="center">
<h1 align="center"> 👉 OneGen 👈 </h1>
<b>OneGen: Efficient One-Pass Unified Generation and Retrieval for LLMs</b>

[![Awesome](https://awesome.re/badge.svg)](https://github.com/zjunlp/OneGen) 
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
![](https://img.shields.io/github/last-commit/zjunlp/OneGen?color=green) 

<p align="center">
  <a href="https://drive.google.com/drive/folders/1ByufnAyvsfnrIVJzMwOHql3lYFVy6IJx?usp=drive_link">☁️ Google Drive (Data)</a>
  <br>
  <a href="https://arxiv.org/abs/2409.05152">📄arXiv</a> •
  <a href="https://x.com/zxlzr/status/1833433788036354523">𝕏 Blog</a> •
  <a>🌐 Web</a>
  <br>
  <br>
  <a>🤗 HF (Model)👇</a> •
  <a>🔭 Model Scope (Model)👇</a> •
  <a>🧊 Wise Model (Model)👇</a> 
</p>

| 🎯 Task Name      | 🤗 HuggingFace                              | 🔭 ModelScope                               | 🧊 WiseModel                                |
| -------------- | ---------------------------------------- | ---------------------------------------- | ---------------------------------------- |
| Entity Linking | [Llama2-7B](https://huggingface.co/zjunlp/OneGen-EntityLinking-Llama2-7B) | [Llama2-7B](https://www.modelscope.cn/models/ZJUNLP/OneGen-EntityLinking-Llama2-7B) | [Llama2-7B](https://www.wisemodel.cn/models/zjunlp/OneGen-EntityLinking-Llama2-7B) |
| Single-hop QA  | [Llama2-7B](https://huggingface.co/zjunlp/OneGen-SelfRAG-Llama2-7B) | [Llama2-7B](https://www.modelscope.cn/models/ZJUNLP/OneGen-SelfRAG-Llama2-7B) | [Llama2-7B](https://www.wisemodel.cn/models/zjunlp/OneGen-SelfRAG-Llama2-7B) |
| Multi-hop QA   | [Llama2-7B](https://huggingface.co/zjunlp/OneGen-MultiHop-Llama2-7B) | [Llama2-7B](https://www.modelscope.cn/models/ZJUNLP/OneGen-MultiHop-Llama2-7B) | [Llama2-7B](https://www.wisemodel.cn/models/zjunlp/OneGen-MultiHop-Llama2-7B) |
</div>




## Table of Contents

- 📋[TODO](#todo)
- 👀[Overview](#overview)
- 🔧[Installation](#installation)
- 🏃[Quick Start](#quick-start)
- 🚩[Citation](#citation)


## 📋TODO

- [ ] Support LoRA train
- [ ] Code documentation
- [ ] Support vLLM inference
- [ ] Support distributed embedding
- [ ] Gradio

## 👀Overview

We introduce a **One**-pass **Gen**eration and retrieval framework (**OneGen**) for fine-tuning LLMs on generation, retrieval, or hybrid tasks. Our core idea is to integrate generation and retrieval to the same context by allocating the retrieval task to *retirval tokens* generated in an autoregressive manner, thus enabling LLM to perform both tasks in a single forward pass.

The following figure illustrates the training process. We first introduce the concept named `roles of tokens in LLMs`. A token $x_i$ is the basic unit processed by an LLM. Token in the input of an LLM serves three different roles:
- *Generating next token*, noted as $role(x_i)=\texttt{GEN}$.
- *Providing context information*, noted as $role(x_i)=\texttt{CTX}$.
- *Representing a sentence*, noted as $role(x_i)=\texttt{RET}$.

Hence, we apply the *cross-entropy loss* for the token $x_i$ where $role(x_i)=\texttt{GEN}$ and apply the *contrastive loss* for the token $x_i$ where $role(x_i)=\texttt{RET}$. This is the training overview.

![](./assets/train.jpg)

The following figure illustrates the inference process of different methods for RAG task. First, we can see both GritLM and OneGen only need to deploy a single model, which can lower the deployment cost. However, GritLM achieves generation and retrieval within a single model by switching back and forth between causal attention and bidirectional attention. Additionally, both GritLM and the Pipeline method require explicit queries, which leads to the need for two forward passes for the queries. In contrast, OneGen can perform retrieval during the generation process, thus **avoiding the two forward pass calculations** for the queries and **allowing for the direct use of kv-cache**, significantly reducing inference costs.

![](./assets/comparison.jpg)

## 🔧Installation

```bash
git clone https://github.com/zjunlp/OneGen
cd OneGen
conda create -n onegen python=3.9 -y
conda activate onegen
pip install -r requirements.txt
```

## 🏃Quick Start

> The inference section focuses on running model predictions to get output results (Single-hop QA is an exception). The evaluation of these results is discussed in the Evaluation section. 

### Download the data

Download `train_data.tar.gz` and `eval_data.tar.gz` from [Google Drive](https://drive.google.com/drive/folders/1ByufnAyvsfnrIVJzMwOHql3lYFVy6IJx?usp=drive_link). After extracting, you will get two folders: `train_data` and `eval_data`. Move these two folders into the `data` directory. Use the following commands to extract the files:
```bash
tar -xzvf train_data.tar.gz
tar -xzvf eval_data.tar.gz
```

### Download the trained model (Optional)

<details> 
<summary><b>Download the trained model (Optional)</b></summary> 
  
The model weights trained on three tasks have been made public and are available for download on three platforms: `🤗Huggingface`, `🔭ModelScope`, and `🧊WiseModel`. For detailed information, please refer to the table below:
| 🎯 Task Name      | 🤗 HuggingFace                              | 🔭 ModelScope                               | 🧊 WiseModel                                |
| -------------- | ---------------------------------------- | ---------------------------------------- | ---------------------------------------- |
| Entity Linking | [Llama2-7B](https://huggingface.co/zjunlp/OneGen-EntityLinking-Llama2-7B) | [Llama2-7B](https://www.modelscope.cn/models/ZJUNLP/OneGen-EntityLinking-Llama2-7B) | [Llama2-7B](https://www.wisemodel.cn/models/zjunlp/OneGen-EntityLinking-Llama2-7B) |
| Single-hop QA  | [Llama2-7B](https://huggingface.co/zjunlp/OneGen-SelfRAG-Llama2-7B) | [Llama2-7B](https://www.modelscope.cn/models/ZJUNLP/OneGen-SelfRAG-Llama2-7B) | [Llama2-7B](https://www.wisemodel.cn/models/zjunlp/OneGen-SelfRAG-Llama2-7B) |
| Multi-hop QA   | [Llama2-7B](https://huggingface.co/zjunlp/OneGen-MultiHop-Llama2-7B) | [Llama2-7B](https://www.modelscope.cn/models/ZJUNLP/OneGen-MultiHop-Llama2-7B) | [Llama2-7B](https://www.wisemodel.cn/models/zjunlp/OneGen-MultiHop-Llama2-7B) |

</details>


> [!NOTE]
> It is worth noting that for the Entity Linking task, we have pre-stored the entity embeddings. Click [here](https://huggingface.co/zjunlp/OneGenEmbedding/blob/main/OneGen-EntityLinking-Llama2-7B-Embedding.pkl) to download them.

### Training model from scratch (Optional)

<details> 
<summary><b>Training model from scratch (Optional)</b></summary>

We provide the training scripts for three tasks. If you are using a locally downloaded model, you can modify the `info-model` field in the `workflow/{task}/{model}.json` file. Update the `model_path` and `tokenizer_path` with the local paths. Note that the hyperparameters in the configuration files are set for 8xA800 GPUs. If you encounter OOM (Out of Memory) issues, please reduce the `per_device_train_batch_size`, `n_pos_per_sent`, `n_neg_per_pos`, and `max_length`.

```bash
# Entity Linking
deepspeed train.py --workflow workflow/entity_linking/llama2.json
# Single-Hop QA
deepspeed train.py --workflow workflow/self_rag/llama2.json
# Multi-hop QA
deepspeed train.py --workflow workflow/multi_hop_qa/llama2.json
```
</details>

### Inference

Here are the inference scripts for the Entity Linking and Multi-hop QA tasks. The inference script for Single-Hop QA is introduced in the next section. You can modify the values of fields such as `model_path`, `tokenizer_path`, `file`, and `output_file_path` in `{config}/{eval_config}/{task}/{config}.json` as needed.

```bash
# Entity Linking (Need GPU)
python eval.py --config config/eval_config/entity_linking/llama2_wo_pkl.json
# Multi-hop QA (Need GPU)
python eval.py --config config/eval_config/multi_hop_qa/llama2.json
```


### Evaluation

Below are the evaluation scripts for the Entity Linking and Multi-hop QA tasks. `/your/path/to/result.jsonl` is the file saved during the inference stage.

```bash
# Entity Linking (CPU)
bash scripts/eval_el.sh el /your/path/to/result.jsonl

# Multi-hop QA for HotpotQA dataset (CPU)
bash scripts/eval_multi_hop_qa.sh /your/path/to/result.jsonl hotpotqa

# Multi-hop QA for 2WIKI dataset (CPU)
bash scripts/eval_multi_hop_qa.sh /your/path/to/result.jsonl 2wiki
```

Here is the evaluation for the Single-Hop QA task, mainly based on [Self-RAG](https://github.com/AkariAsai/self-rag):
```bash
# Single-hop QA using Self-RAG (Need GPU)
# [CUDA_VISIBLE_DEVICES] [MODE] [MODEL_PATH] [SAVE_TAG] [SAVED_DATASET_PATH] [N_DOC] [ENV] [SCORE]
bash scripts/eval_self_rag.sh 0 always_retrieve /your/path/to/model model_tag saved_rank_path 5 true true
```

## 🚩Citation

If this work is helpful, please kindly cite as:

```bibtex
@misc{zhang2024onegen,
      title={OneGen: Efficient One-Pass Unified Generation and Retrieval for LLMs}, 
      author={Jintian Zhang and Cheng Peng and Mengshu Sun and Xiang Chen and Lei Liang and Zhiqiang Zhang and Jun Zhou and Huajun Chen and Ningyu Zhang},
      year={2024},
      eprint={2409.05152},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2409.05152}, 
}
```
