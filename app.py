from flask import Flask , render_template, request 
from transformers import MarianMTModel, MarianTokenizer
app = Flask(__name__)

# to select the model 
model_name = 'Helsinki-NLP/opus-mt-en-hi'  # English to Hindi translation model
# to load the model 
model = MarianMTModel.from_pretrained(model_name)
# to load the  tonkenizer
tokenizer = MarianTokenizer.from_pretrained(model_name)

def translation(data):
    # convert the text into tokens
    inputs = tokenizer(data, return_tensors="pt", padding=True)
    # translate the data using the model 
    translated = model.generate(**inputs)
    # decode the token into the readable text
    result = tokenizer.decode(translated[0], skip_special_tokens=True)
    return result

@app.route('/', methods=['GET', 'POST'])
def index():
    translated_text = ""
    if request.method == 'POST':
        data = request.form['data']
        translated_text = translation(data)
    return render_template('index.html', translated_text=translated_text) 

if __name__ == '__main__':
    app.run(debug=True)