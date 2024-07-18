import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class Inference:
    def generate_output(self, input_data):
        # load the fine-tuned model
        tokenizer = AutoTokenizer.from_pretrained("ashwinR/CodeExplainer")
        model = AutoModelForSeq2SeqLM.from_pretrained("model")

        # Move the model to GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Set the model to evaluation model
        model.eval()

        # Define the input data
        input_data = input_data

        # Tokenize the input data
        tokenized_input = tokenizer.encode(input_data, return_tensors='pt')

        # Move tokenized input to device
        tokenized_input = tokenized_input.to(device)

        # Generate output using the model
        with torch.no_grad():
            output = model.generate(tokenized_input,
                                    max_length=1024,
                                    num_beams=5,
                                    early_stopping=True,
                                    num_return_sequences=1,
                                    do_sample=True,
                                    top_k=8,
                                    top_p=0.95,
                                    temperature=0.8,
                                    repetition_penalty=1.2)

        # Decode the output
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

        print(generated_text)

        return generated_text
