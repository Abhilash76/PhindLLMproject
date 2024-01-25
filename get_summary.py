import nbformat


def extract_text_from_notebook(notebook_path, flag):
    with open(notebook_path, "r", encoding="utf-8") as notebook_file:
        notebook_content = nbformat.read(notebook_file, as_version=4)

    # Extract text from code and markdown cells
    markdown_text_content = ""
    code_text_content = ""
    for cell in notebook_content['cells']:
        if cell['cell_type'] == 'code':
            code_text_content += cell['source'] + '\n'
        elif cell['cell_type'] == 'markdown':
            markdown_text_content += cell['source'] + '\n\n'

    if flag == 'code':
        return code_text_content
    elif flag == 'markdown':
        return markdown_text_content


# Example usage
notebook_path = "C:\\Users\\Abhilash\\Downloads\\Assignment4_Lannisters\\CA4\\Evaluation.ipynb"
notebook_text = extract_text_from_notebook(notebook_path, 'markdown')
# Specify the file path
file_path = "results.txt"
# Open the file in write mode
with open(file_path, 'w') as file:
    # Write each result to a new line
    for result in notebook_text:
        file.write(result)
print(notebook_text)
