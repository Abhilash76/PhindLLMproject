import torch
from transformers import AutoTokenizer, LlamaForCausalLM, pipeline


def get_content():
    with open("C:\\Users\\Abhilash\\PhindLLMproject\\results.txt", "r") as file:
        code_content = file.read()
    return code_content


class nb_summary:
    # initialize the model
    model_path = "TheBloke/CodeLlama-13B-Instruct-fp16"

    pipe = pipeline("text-generation", model="TheBloke/CodeLlama-13B-Instruct-fp16")

    model = LlamaForCausalLM.from_pretrained(model_path, device_map="auto", offload_folder="offload",
                                             offload_state_dict=True, torch_dtype=torch.float32)
    # Device Placement (Optional)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Set up tokenizer and model
    tokenizer.pad_token = tokenizer.eos_token

    # Get the input string
    code = get_content()

    # Tokenize input text
    tokens = tokenizer(code, return_tensors="pt")

    # Ensure attention mask is set
    attention_mask = tokens["attention_mask"]

    # Pass the input string to the tokenizer
    inputs = tokenizer(code, return_tensors="pt", truncation=True, max_length=8192)

    # Use the model to generate code and docstring
    generated_ids = model.generate(inputs.input_ids.to("cuda"), max_length=8192, num_beams=5, length_penalty=0.8)
    generated_code = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    print("Generated Docstring:", generated_code)
