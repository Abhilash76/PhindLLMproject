import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from rouge_score import rouge_scorer
import pandas as pd
from datasets import Dataset

ds = Dataset.from_file("dataset.arrow")

updated_data = [{'Code': item['text'], 'Description': item['target']} for item in ds]

df = pd.DataFrame(updated_data)

# Load the fine-tuned model
model = AutoModelForSeq2SeqLM.from_pretrained('model')
tokenizer = AutoTokenizer.from_pretrained('ashwinR/CodeExplainer')

# Evaluate the fine-tuned model
fine_tuned_rouge_scores = []
for index, row in df.iterrows():
    input_ids = tokenizer.encode(row['Code'], return_tensors='pt')
    generated_summary = model.generate(input_ids, max_length=70, do_sample=True, temperature=0.5)
    generated_summary_text = tokenizer.decode(generated_summary[0], skip_special_tokens=True)
    print("Printing the target summary here: \n", row['Description'], "\n and the generated summary here: \n", generated_summary_text)
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(row['Description'], generated_summary_text)
    fine_tuned_rouge_scores.append(scores)

# Load the original pre-trained model
original_model = AutoModelForSeq2SeqLM.from_pretrained('ashwinR/CodeExplainer')
original_tokenizer = tokenizer

# Evaluate the original pre-trained model
original_rouge_scores = []
for index, row in df.iterrows():
    input_ids = tokenizer.encode(row['Code'], return_tensors='pt')
    generated_summary = model.generate(input_ids, max_length=70, do_sample=True, temperature=0.5)
    generated_summary_text = tokenizer.decode(generated_summary[0], skip_special_tokens=True)
    print("Printing the target summary here: \n", row['Description'], "\n and the generated summary here: \n", generated_summary_text)
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(row['Description'], generated_summary_text)
    original_rouge_scores.append(scores)

# Compare ROUGE scores
print("Fine-tuned model ROUGE scores:", fine_tuned_rouge_scores)
print("Original pre-trained model ROUGE scores:", original_rouge_scores)
