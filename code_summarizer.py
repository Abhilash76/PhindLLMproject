import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration, pipeline

model_name = "Salesforce/codet5-large"
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-large")
model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-large")


config = "model"

model.eval()

pipe = pipeline("summarization",
                model=model_name,
                config=config,
                tokenizer=tokenizer,
                max_length=1024,
                torch_dtype=torch.float32,
                device_map="auto")

# Specify the path to the file
file_path = "results.txt"

# Read the content of the file
with open(file_path, "r", encoding="utf-8") as file:
    text_content = file.read()

if len(text_content) > 512:
    text_content = text_content[:1024]

raw_code = text_content

input_ids = tokenizer(raw_code, return_tensors="pt").input_ids

# simply generate a single sequence
generated_ids = model.generate(input_ids, max_length=1024)
print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
