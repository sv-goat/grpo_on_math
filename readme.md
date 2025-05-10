# Group Relative Policy Optimization for Mathematical Reasoning

## Course Project - EECS 6892: Reinforcement Learning

This repository contains the implementation of Group Relative Policy Optimization (GRPO) for mathematical reasoning using the GSM8k dataset.

### Model Weights

The trained model weights are available on [Hugging Face](https://huggingface.co/rocketpenguin25603/outputs)

### Directory Tree

```tree
.
├── grpo-scratch.py          # Contains the from-scratch implementation of GRPO
├── readme.md
├── scripts
│   ├── evaluator.py
│   ├── grpo.py
│   ├── __init__.py
│   └── reward.py
├── train_grpo_0_5B_TRL.ipynb    # TRL implementation of GRPO on Qwen-2.5-0.5B-Instruct model
└── vllm_verify_gsm8k.ipynb      # Evaluation results for the from-scratch implementation
```