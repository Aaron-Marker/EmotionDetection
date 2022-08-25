from flask import Flask,render_template,request
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, pipeline
import tensorflow as tf

application = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = TFAutoModelForSequenceClassification.from_pretrained("./emotionModel/", num_labels=6)
#model.load_weights("model")
classifier = pipeline("text-classification", model = model, tokenizer = tokenizer)
f = open("demofile.txt", "r")
test = f.read()

#@application.before_first_request
#def load_model():
      

@application.route('/')
def home():
    return render_template('site.html')

@application.route('/results/', methods = ['POST', 'GET'])
def results():
    if request.method == 'GET':
        return f"The URL /results should not be accessed directly. Try navigating to the root directory."
    if request.method == 'POST':
        sentence = request.form["sentence"]
        emotions = {'LABEL_0':'sadness', 'LABEL_1':'joy', 'LABEL_2':'love', 'LABEL_3':'anger', 'LABEL_4':'fear', 'LABEL_5':'surprise'}
        prediction = emotions[classifier(sentence)[0]["label"]]
        form_data = [sentence, prediction, test]
        return render_template('data.html',form_data = form_data)
