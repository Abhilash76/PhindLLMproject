from transformers import pipeline

# classifier = pipeline("sentiment-analysis")

# res = classifier("I am overjoyed that the program executed.")

# generator = pipeline("text-generation", model="distilgpt2")

# res = generator(
#     "Twinkle twinkle little stars, how I",
#     max_length=70,
#     num_return_sequences=2
# )

# print(res)

# pipe = pipeline("summarization", model="Phind/Phind-CodeLlama-34B-v2")

# pipe = pipeline("text2text-generation", model="stmnk/codet5-small-code-summarization-python")

pipe = pipeline("text-generation", model="codellama/CodeLlama-7b-Instruct-hf")

# Specify the path to the file
file_path = "results.txt"

# Read the content of the file
with open(file_path, "r", encoding="utf-8") as file:
    text_content = file.read()

# Generate text using the pipeline
generated_text = pipe(text_content, num_return_sequences=1)

# Print or use the generated text as needed
print(generated_text[0]['generated_text'])

file_path = "output.txt"
# Open the file in write mode
with open(file_path, 'w') as file:
    # Write each result to a new line
    file.write(generated_text)
