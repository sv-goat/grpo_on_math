import re
from typing import Optional, Union
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader

from datasets import Dataset

from vllm import LLM, SamplingParams

class FewShotEvaluator:
    """
    Few-shot evaluator for math reasoning tasks.
    """
    def __init__(self, dataset: Dataset, n_shots: int = 3, device: str = "cuda", batch_size: int = 16) -> None:
        self.dataset = dataset
        self.n_shots = n_shots
        self.device = device
        self.batch_size = batch_size
        self.fewshot_prompt = self.get_fewshot_prompt()

    def get_fewshot_prompt(self) -> str:
        prompt = "Solve these math problems:\n\n"
        for i in range(self.n_shots):
            example = self.dataset[i]
            prompt += f"Question: {example['question']}\nAnswer: {example['answer']}" + "\n\n"
        return prompt

    def preprocess_eval(self, examples: dict) -> dict:
        # Preprocess the example to include the few-shot prompt
        return {
            "prompt": [self.fewshot_prompt + f"Question: {question}\nAnswer:\n" for question in examples["question"]]
        }

    def parse_answer(self, answer: str) -> Optional[str]:
        # Extract the answer from the generated text
        try:
            predicted_answer = re.search(r"#### (-?\d+\.?\d*)", answer).group(1)
        except:
            predicted_answer = None
        return predicted_answer

    # def eval(self, model_path: str, tokenizer: AutoTokenizer, device: str = "cuda", temperature: float = 0.7, top_p: float = 0.95, max_tokens: int = 256) -> float:
    def eval(self, model_path: str, dtype: str = "auto", device: str = "cuda", temperature: float = 0.7, top_p: float = 0.95, max_tokens: int = 256) -> float:
        """
        Evaluate exact match accuracy
        """
        # Load dataset
        eval_dataset = self.dataset.select(range(self.n_shots, len(self.dataset)))
        eval_dataset = eval_dataset.map(self.preprocess_eval, batched=True)
        eval_dataloader = DataLoader(eval_dataset, batch_size=self.batch_size, shuffle=False)

        # Load model
        llm = LLM(model=model_path, dtype=dtype)
        # Shared or individual sampling settings
        sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens)

        correct = 0
        num_questions = 0

        answers = []

        # batch inference
        for _, batch in tqdm(enumerate(eval_dataloader), desc="Eval Inference: ", total=len(eval_dataloader)):
            # inputs = tokenizer(batch["prompt"], return_tensors="pt", max_length=256, padding="max_length", truncation=True).to(device)
            # outputs = model.generate(**inputs, max_new_tokens=256)
            # batch_answers = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            # answers.extend(batch_answers)
            prompts = batch["prompt"]
            outputs = llm.generate(prompts, sampling_params)
            batch_answers = [output.outputs[0].text.strip() for output in outputs]
            answers.extend(batch_answers)
            torch.cuda.empty_cache()

        # text parse for exact match
        for i, (correct_answer, generated_answer) in tqdm(enumerate(zip(eval_dataset['answer'], answers)), desc="Evaluating Exact Match Accuracy: ", total=len(eval_dataset)):
            # # Remove the input tokens from the output for transformers inference
            # generated_answer = generated_answer[len(eval_dataset['prompt'][i]):]

            # Extract final answer
            predicted_answer = self.parse_answer(generated_answer)
            ground_truth = self.parse_answer(correct_answer)

            # Check if the predicted answer matches the ground truth
            if ground_truth:
                num_questions += 1
                if predicted_answer and predicted_answer == ground_truth:
                    correct += 1

        return correct / num_questions if num_questions > 0 else 0