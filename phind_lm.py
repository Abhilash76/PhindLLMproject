from transformers import AutoTokenizer, LlamaForCausalLM
from human_eval.data import write_jsonl, read_problems
from tqdm import tqdm

# initialize the model

model_path = "Phind/Phind-CodeLlama-34B-v2"
model = LlamaForCausalLM.from_pretrained(model_path, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)


# HumanEval helper

def generate_one_completion(prompt: str):
    # Set up tokenizer and model (assuming you have them instantiated somewhere in your code)
    tokenizer.pad_token = tokenizer.eos_token

    # Tokenize the notebook content
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)

    # Generate
    generate_ids = model.generate(inputs.input_ids.to("cuda"), max_new_tokens=384, do_sample=True, top_p=0.75, top_k=40,
                                  temperature=0.1)
    completion = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    completion = completion.replace(prompt, "").split("\n\n\n")[0]

    return completion


def extract_summary(generated_text: str):
    # Split the generated text into paragraphs
    paragraphs = generated_text.split("\n\n")

    # Extract the first paragraph as the summary
    if paragraphs:
        summary = paragraphs[0]
    else:
        summary = "No summary available."

    return summary


def generate_notebook_summary(notebook_content: str):
    # Tokenize the notebook content
    inputs = tokenizer(notebook_content, return_tensors="pt", truncation=True, max_length=8192)

    # Generate
    generate_ids = model.generate(inputs.input_ids.to("cuda"), max_new_tokens=384, do_sample=True, top_p=0.75, top_k=40,
                                  temperature=0.1)

    # Decode generated tokens
    completion = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    # Extract summary from the generated completion (modify as needed)
    summary = extract_summary(completion)

    return summary


# perform HumanEval
problems = read_problems()

num_samples_per_task = 1
samples = [
    dict(task_id=task_id, completion=generate_one_completion(problems[task_id]["prompt"]))
    for task_id in tqdm(problems)
    for _ in range(num_samples_per_task)
]
write_jsonl("samples.jsonl", samples)

# run `evaluate_functional_correctness samples.jsonl` in your HumanEval code sandbox
