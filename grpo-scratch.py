import datetime
import os
import re
from typing import List, Dict, Callable, Iterable
import copy
import wandb

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import datasets
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.grpo import GRPODataset, RewardGrader, CompletionProcessor, logp_per_token
from scripts.reward import correctness_reward, format_reward

from dotenv import load_dotenv
load_dotenv()

# model_id = "Qwen/Qwen2.5-1.5B-Instruct"
model_id = "meta-llama/Llama-3.2-1B-Instruct"

Q_FIELD = "question"
A_FIELD = "answer"

SYSTEM_PROMPT = f"""
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""
MAX_GEN_LEN = 400
TOP_K = 50
TOP_P = 0.9
TEMPERATURE = 1.0

GROUP_SIZE = 14
GRPO_ITER = 1
EPS = 0.2
BETA = 1e-2

EARLY_STOP = True
EARLY_STOP_STEP = 400
PRECISION = torch.bfloat16
EPOCHS = 1
LR = 5e-6
BATCH_SIZE = 7

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

def optimize_model_memory(model):
    """
    Optimizes the model to use less memory during training.

    Args:
        model: The language model to optimize.

    Returns:
        The optimized model.

    Explanation:
        1. Sets the model to training mode.
        2. Disables KV caching to save memory.
        3. Enables gradient checkpointing to trade computation for memory.
        4. Ensures that input embeddings require gradients:
           - Either uses the built-in method if available.
           - Or adds a forward hook to the input embeddings layer.
        5. Returns the optimized model ready for memory-efficient training.
    """
    model.train()
    model.config.use_cache = False

    # First ensure inputs will require gradients
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # Then enable gradient checkpointing
    model.gradient_checkpointing_enable()

    return model


gsm8k_train = load_dataset("openai/gsm8k", "main", split="train")

gsm8k_train_ds = GRPODataset(
    dataset=gsm8k_train,
    prompt_field=Q_FIELD,
    answer_field=A_FIELD,
    sys_prompt_template=SYSTEM_PROMPT,
)

train_dataloader = DataLoader(
    gsm8k_train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

reward_fn_registry = {
    "correctness": correctness_reward,
    "format": format_reward,
}
reward_grader = RewardGrader(reward_fn_registry=reward_fn_registry)


timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# main training loop
# torchrun --nproc_per_node=2 grpo-scratch.py

# 1. Setup distributed environment
dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

# init wandb
if dist.get_rank() == 0:
    wandb.init(
        project="RFT",
        name=f"{model_id.split('/')[-1]}-GRPO-GSM8K-{timestamp}",
        config={
            "model_id": model_id,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "group_size": GROUP_SIZE,
            "group_iter": GRPO_ITER,
            "eps": EPS,
            "beta": BETA,
            "max_gen_len": MAX_GEN_LEN,
            "top_k": TOP_K,
            "top_p": TOP_P,
            "temperature": TEMPERATURE,
        },
        reinit=True,
    )


# 2. Initialize model on correct device
policy_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=PRECISION).to(device)
policy_model.config.pad_token_id = tokenizer.pad_token_id
policy_model.config.eos_token_id = tokenizer.eos_token_id
policy_model.config.use_cache = False
policy_model = optimize_model_memory(policy_model)

# 3. Wrap with DistributedDataParallel
policy_model = DDP(policy_model, device_ids=[local_rank])

# 4. Use DistributedSampler for dataloader
train_sampler = DistributedSampler(gsm8k_train_ds)
train_dataloader = torch.utils.data.DataLoader(gsm8k_train_ds, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=4)


for epoch in range(EPOCHS):
    train_sampler.set_epoch(epoch)  # ensures good shuffling across workers

    # Update reference model (no DDP needed for inference-only models)
    ref_model = copy.deepcopy(policy_model.module).to(device)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # move optimizer into loop
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=LR)

    for i, batch in enumerate(train_dataloader):
        if EARLY_STOP and i >= EARLY_STOP_STEP:
            break
        old_model = copy.deepcopy(policy_model.module if GRPO_ITER > 0 else ref_model).to(device)
        old_model.eval()
        for param in old_model.parameters():
            param.requires_grad = False

        completion_processor = CompletionProcessor(policy_model.module, tokenizer, max_length=MAX_GEN_LEN)

        questions = batch[Q_FIELD]
        answers = batch[A_FIELD]

        torch.cuda.empty_cache()
        with torch.no_grad():
            rollout_data = completion_processor.generate_rollout_data(
                prompts=questions,
                num_sequences=GROUP_SIZE,
                temperature=TEMPERATURE
            )

            ref_logp_per_token = logp_per_token(rollout_data, ref_model)
            old_logp_per_token = logp_per_token(rollout_data, old_model)

        completions_mask = rollout_data.completions_mask
        repeated_answers = [answer for answer in answers for _ in range(GROUP_SIZE)]
        rewards = reward_grader(
            completions=rollout_data.completions,
            answers=repeated_answers
        ).to(PRECISION).to(device).view(-1, GROUP_SIZE)

        advantages = (rewards - rewards.mean(dim=-1, keepdim=True)) / (rewards.std(dim=-1, keepdim=True) + 1e-8)
        completion_len_dim = rollout_data.completions_ids.size(1)
        advantages = advantages.view(-1).unsqueeze(1).expand(-1, completion_len_dim)

        policy_model.train()
        for mu in range(GRPO_ITER):
            new_logp_per_token = logp_per_token(rollout_data, policy_model.module)

            ref_over_new_logp = ref_logp_per_token - new_logp_per_token
            kl_div = torch.exp(ref_over_new_logp) - ref_over_new_logp - 1

            new_over_old_prob = torch.exp(new_logp_per_token - old_logp_per_token)
            surrogated_advantage = torch.min(
                new_over_old_prob * advantages,
                torch.clip(new_over_old_prob, 1 - EPS, 1 + EPS) * advantages
            )

            per_token_loss = - (surrogated_advantage - BETA * kl_div) * completions_mask
            loss = (per_token_loss.sum(dim=-1) / completions_mask.sum(dim=-1)).mean()

            # Only log on main process
            if dist.get_rank() == 0:
                print(f"\nEpoch {epoch}, Step {i}, Iteration {mu}, Loss: {loss.item()}, Avg reward: {rewards.mean().item()}")
                wandb.log({
                    "epoch": epoch,
                    "step": i,
                    "loss": loss.item(),
                    "avg_reward": rewards.mean().item(),
                })

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=0.1)
            optimizer.step()

wandb.finish()

save_dir = f"/network/rit/lab/wang_lab_cs/ptian/output/rft/{model_id.split("/")[-1].lower()}-grpo-{timestamp}"
tokenizer.save_pretrained(save_dir)
policy_model.module.save_pretrained(save_dir)
