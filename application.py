from flask import Flask,render_template,request,redirect, url_for
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, pipeline
import tensorflow as tf

application = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
#from transformers import AutoModel

#model = AutoModel.from_pretrained("AaronMarker/my-awesome-model")
model = TFAutoModelForSequenceClassification.from_pretrained("AaronMarker/my-awesome-model", num_labels=6)
#model.load_weights("model")
classifier = pipeline("text-classification", model = model, tokenizer = tokenizer)
f = open("demofile.txt", "r")
test = f.read()

#@application.before_first_request
#def load_model():
      

@application.route('/')
def home():
    #if request.method == 'POST':
    #    sentence = request.form["home"]
    return render_template('site.html')

@application.route('/results/', methods = ['POST', 'GET'])
def results():
    if request.method == 'GET':
        return redirect(url_for('home'))
    if request.method == 'POST':
        sentence = request.form["sentence"]
        emotions = {'LABEL_0':'Sadness', 'LABEL_1':'Joy', 'LABEL_2':'Love', 'LABEL_3':'Anger', 'LABEL_4':'Fear', 'LABEL_5':'Surprise'}
        prediction = emotions[classifier(sentence)[0]["label"]]
        form_data = [sentence, prediction, test]
        return render_template('data.html',form_data = form_data)
