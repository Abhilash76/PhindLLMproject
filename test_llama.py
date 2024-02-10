from transformers import AutoTokenizer
import transformers
import torch

model = "codellama/CodeLlama-7b-hf"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "question-answering",
    model=model,
    torch_dtype=torch.float16,
    device_map="cpu",
)

# Specify the path to the file
file_path = "results.txt"

# Read the content of the file
with open(file_path, "r", encoding="utf-8") as file:
    text_content = file.read()


sequences = pipeline(
    'Generate summary for the following code:\n'+text_content,
    do_sample=True,
    top_k=10,
    temperature=0.5,
    top_p=1,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=2000,
    truncation=True
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")
