<div align="center">
<h1 align="center"> 👉 OneGen 👈 </h1>
<b>OneGen: Efficient One-Pass Unified Generation and Retrieval for LLMs</b>
  
<p align="center">
  <a href="https://arxiv.org/">📄arXiv</a> •
  <a href="https://x.com/">𝕏 Blog</a> •
  <a href="https://huggingface.co/">🤗 HF</a>
</p>

[![Awesome](https://awesome.re/badge.svg)](https://github.com/zjunlp/OneGen) 
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
![](https://img.shields.io/github/last-commit/zjunlp/OneGen?color=green) 
</div>

## Table of Contents

- 📋[TODO](#todo)
- 👀[Overview](#overview)
- 🔧[Installation](#installation)
- 🏃[Quick Start](#quick-start)
- 🚩[Citation](#citation)


## 📋TODO

- [ ] Support LoRA train
- [ ] Refactor evaluation code
- [ ] Upload model
- [ ] Code documentation
- [ ] Support vLLM inference

## 👀Overview



## 🔧Installation

```bash
git clone https://github.com/zjunlp/OneGen
cd OneGen
conda create -n onegen python=3.9 -y
conda activate onegen
pip install -r requirements.txt
```

## 🏃Quick Start

```bash
# Entity Linking
deepspeed train.py --workflow workflow/entity_linking/llama2.json
# Single-Hop QA
deepspeed train.py --workflow workflow/self_rag/llama2.json
# Multi-hop QA
deepspeed train.py --workflow workflow/multi_hop_qa/llama2.json
```

```bash
bash eval_scripts/eval_self_rag.sh
```

```bash
bash eval_scripts/eval_multi_hop.sh
```

## 🚩Citation

If this work is helpful, please kindly cite as:

```bibtex

```

