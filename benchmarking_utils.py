def format_question(example):
    return f"Question: {example['question']}\nAnswer: {example['answer']}"

''' Different math datasets might have differnet identifiers for the answer.
For GSM8k, the format is: #### <answer>. 
'''
def evaluate_gsm8k(model, tokenizer, dataset, n_shots=3):
    correct = 0
    num_questions = 0
    shots = []
    for i in range(n_shots):
        shot = dataset[i]
        shots.append(shot)
    non_shot_data = dataset.select(range(n_shots, len(dataset)))
    for i, example in enumerate(non_shot_data):  # Iterate over the dataset:
        # Build few-shot prompt
        prompt = "Solve these math problems:\n\n"
        # Making sure that the example is not in the shots
        for shot in shots:
            prompt += format_question(shot) + "\n\n"
        # Get it to gen the required answer
        prompt += f"Question: {example['question']}\nAnswer:"

        # Generate answer
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=200)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove the input tokens from the output
        answer = answer[len(prompt):]

        # Extract final answer
        try:
            predicted_answer = re.search(r"#### (\d+\.?\d*)", answer)
            ground_truth = re.search(r"#### (\d+\.?\d*)", example["answer"]).group(1)
            num_questions += 1
        except:
            predicted_answer = None
            ground_truth = None
            continue

        if predicted_answer and predicted_answer.group(1) == ground_truth:
            # We want the first instnace of the ### to be the required answer
            correct += 1
        # If things are working, increase to 25 and verfiy. then we can let it run to completion. # also get  the length of the dataset.
        if i % 25 == 0:
            print("Completed", i)
            print("Accuracy", correct / (num_questions))
            print("Sanity check")
            print("Prompt", prompt)
            print("Answer", answer)
            print("was correct", predicted_answer and predicted_answer.group(1) == ground_truth)

        # Free memory
        inputs = None
        outputs = None
        torch.cuda.empty_cache()

    return correct / (num_questions)