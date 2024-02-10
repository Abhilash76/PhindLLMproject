from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoConfig,
    pipeline,
)

model_name = "sagard21/python-code-explainer"

tokenizer = AutoTokenizer.from_pretrained(model_name, padding=True)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

config = AutoConfig.from_pretrained(model_name)

model.eval()

pipe = pipeline("summarization", model=model_name, config=config, tokenizer=tokenizer, max_length=1024)

# Specify the path to the file
file_path = "results.txt"

# Read the content of the file
with open(file_path, "r", encoding="utf-8") as file:
    text_content = file.read()

if len(text_content) > 512:
    text_content = text_content[:1024]

raw_code = text_content

print(pipe(raw_code)[0]["summary_text"])
