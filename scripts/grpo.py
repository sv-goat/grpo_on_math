from typing import List, Dict, Callable, Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import datasets



class RolloutData:
    """
    Rollout completions for a list of prompts
    """
    def __init__(
        self,
        prompts_ids: torch.Tensor,
        prompts_mask: torch.Tensor,
        completions_ids: torch.Tensor,
        completions_mask: torch.Tensor,
        prompts: List[str],
        completions: List[str],
    ): 
        self.prompts_ids = prompts_ids
        self.prompts_mask = prompts_mask
        self.completions_ids = completions_ids
        self.completions_mask = completions_mask
        self.prompts = prompts
        self.completions = completions

    def __getitem__(self, key):
        if key == "prompts_ids":
            return self.prompts_ids
        elif key == "prompts_mask":
            return self.prompts_mask
        elif key == "completions_ids":
            return self.completions_ids
        elif key == "completions_mask":
            return self.completions_mask
        elif key == "prompts":
            return self.prompts
        elif key == "completions":
            return self.completions
        else:
            raise KeyError(f"Key {key} not found in RolloutData.")


class CompletionProcessor:
    """
    Processor for generating rollout sequences.
    """
    def __init__(self, model, tokenizer, max_length=512, top_k=50, top_p=0.95):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.top_k = top_k
        self.top_p = top_p

    def generate_rollout_data(
        self, 
        prompts: List[str], 
        num_sequences: int = 4, 
        temperature: int = 0.7
    ):
        """
        Generate rollout sequences and completion masks for 
        """
        # self.model.eval()
        input_prompts = []
        for prompt in prompts:
            for _ in range(num_sequences):
                input_prompts.append(prompt)
        # generate token ids with left padding
        inputs = self.tokenizer(
            input_prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            padding_side="left", 
            # max_length=self.max_length
        ).to(self.model.device)
        prompts_ids = inputs["input_ids"]
        prompts_mask = inputs["attention_mask"]
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_length,
            do_sample=True,
            temperature=temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            pad_token_id=self.tokenizer.eos_token_id
        )

        # get completion masks
        prompt_len_dim = inputs["input_ids"].size(1)
        completions_ids = outputs[:, prompt_len_dim:]
        is_eos = completions_ids == self.tokenizer.eos_token_id
        eos_exist = is_eos.any(dim=1)
        eos_idx = torch.full((completions_ids.size(0), ), completions_ids.size(1)).to(torch.long).to(self.model.device)
        eos_idx[eos_exist] = is_eos[eos_exist].int().argmax(dim=1)
        completions_mask = torch.zeros_like(completions_ids)
        completions_mask = (completions_mask <= eos_idx.unsqueeze(1)).int()
        return RolloutData(
            prompts_ids=prompts_ids,
            prompts_mask=prompts_mask,
            completions_ids=completions_ids,
            completions_mask=completions_mask,
            prompts=input_prompts,
            completions=self.tokenizer.batch_decode(completions_ids, skip_special_tokens=True),
        )

class GRPODataset(Dataset):
    def __init__(
            self, 
            dataset: datasets.Dataset, 
            prompt_field: str, 
            answer_field: str, 
            sys_prompt_template: str,
        ):
        self.dataset = dataset
        self.prompt_field = prompt_field
        self.answer_field = answer_field
        self.sys_prompt_template = sys_prompt_template
    
    def build_prompt(self, example):
        return "\n".join([
            self.sys_prompt_template,
            example[self.prompt_field],
        ])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        example = self.dataset[idx]
        prompt = self.build_prompt(example)
        answer = example[self.answer_field]
        return {
            self.prompt_field: prompt,
            self.answer_field: answer,
        }
    
class RewardGrader:
    def __init__(self, reward_fn_registry: Dict[str, Callable]):
        self.reward_fn_registry = reward_fn_registry

    def batch_reward(
        self, 
        completions: Iterable[str],
        reward_fn_name: str,
        **kwargs: Dict[str, str]
    ) -> Iterable[float]:
        """
        Compute rewards for a batch of prompts, completions, and answers using the specified reward function.
        
        """
        # check if reward function is registered
        if reward_fn_name not in self.reward_fn_registry:
            raise ValueError(f"Reward function {reward_fn_name} not found in registry.")
        # get reward function
        reward_fn = self.reward_fn_registry[reward_fn_name]
        # check answers input
        if "answers" not in kwargs.keys():
            answers = [None] * len(completions)
        else:
            answers = kwargs["answers"]
        # compute rewards
        rewards = []
        for a, c in zip(answers, completions):
            rewards.append(reward_fn(completion=c, answer=a, **kwargs))
        return rewards
    
    def compute_reward(self, completions, **kwargs):
        all_rewards = []
        for reward_fn_name in self.reward_fn_registry.keys():
            all_rewards.append(
                self.batch_reward(completions=completions, reward_fn_name=reward_fn_name, **kwargs)
            )
        all_rewards = torch.tensor(all_rewards)
        return all_rewards.sum(dim=0)
        

    def __call__(self, **kwargs):
        return self.compute_reward(**kwargs)


        

def logp_per_token(rollout_data: RolloutData, model) -> torch.Tensor:
    """
    Compute the log probability of each token in the completions.
    """
    inputs_ids = torch.cat([rollout_data.prompts_ids, rollout_data.completions_ids], dim=1)
    attention_mask = torch.cat([rollout_data.prompts_mask, rollout_data.completions_mask], dim=1)
    outputs = model(inputs_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]  # omit last token's logits (autoregressive model property)
    # completion logits: (batch_size * group_size, completion_len, vocab_size)
    completion_len_dim = rollout_data.completions_ids.size(1)
    completion_logits = logits[:, -completion_len_dim:, :]
    # completion token log probabilities: (batch_size * group_size, completion_len)
    completion_log_probs = F.log_softmax(completion_logits, dim=-1)  # (batch_size, completion_len, vocab_size)
    log_probs_per_token = torch.gather(completion_log_probs, dim=-1, index=rollout_data.completions_ids.unsqueeze(-1)).squeeze(-1)
    return log_probs_per_token