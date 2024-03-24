import torch
from flask import Flask, render_template, request
from tokenizer.tokenizer import Tokenizer
from animeGPT.config import config
from animeGPT.transformer import GPTLanguageModel

app = Flask(__name__)

device = 'cpu'

model = GPTLanguageModel(config['vocab_size'], config['n_embd'], config['max_seq_length'], config['n_head'], config['n_layer'], config['dropout'])
model.load_state_dict(torch.load('model',  map_location=torch.device('cpu')))


animeTokenizer = Tokenizer()
animeTokenizer.load('./tokenizer/animeTokenizer.model')

def extract_text(text):
  """
  Extracts text from a string with specific tags.

  Args:
      text: The string containing the text to be extracted.

  Returns:
      A dictionary containing the extracted text for each tag ('t', 'g', 's').
  """
  tags = {'t': None, 'g': None, 's': None}
  for tag, value in tags.items():
    start_tag = f"<{tag}>"
    end_tag = f"</{tag}>"
    start_index = text.find(start_tag)
    if start_index != -1:
      end_index = text.find(end_tag, start_index)
      if end_index != -1:
        value = text[start_index + len(start_tag):end_index]
        tags[tag] = value
  return tags

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the web form (you'll create this in index.html)
    # Preprocess the input data to the format your model expects 
    # ...

    # Perform inference with your model
    context = torch.tensor(animeTokenizer.encode('<t>', 'all'), dtype=torch.long, device=device).reshape(-1, 1)
    text_output = (animeTokenizer.decode(model.generate(context, 100, 512)[0].tolist()))
    output = extract_text(text_output)
    title = output['t']
    genres = output['g']
    
    display = f'Title : {title} \n Genres : {genres}'

    # Process the model output into a user-friendly format
    # ...

    return render_template('result.html', prediction=display) 




if __name__ == '__main__':
    app.run() 