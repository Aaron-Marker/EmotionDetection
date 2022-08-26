from flask import Flask,render_template,request,redirect, url_for
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, pipeline
import tensorflow as tf
#Author: Aaron Marker

application = Flask(__name__)

#Load Model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = TFAutoModelForSequenceClassification.from_pretrained("AaronMarker/emotionClassifier", num_labels=9)
classifier = pipeline("text-classification", model = model, tokenizer = tokenizer)

      
@application.route('/')
def home():
    return render_template('site.html')

@application.route('/results/', methods = ['POST', 'GET'])
def results():
    if request.method == 'GET':
        return redirect(url_for('home'))
    if request.method == 'POST':
        sentence = request.form["sentence"]
        emotions = {'LABEL_0':'Joy', 
                    'LABEL_1':'Desire', 
                    'LABEL_2':'Admiration', 
                    'LABEL_3':'Approval', 
                    'LABEL_4':'Curiosity', 
                    'LABEL_5':'Fear', 
                    'LABEL_6':'Sadness', 
                    'LABEL_7':'Anger', 
                    'LABEL_8':'Neutral'}
        prediction = emotions[classifier(sentence)[0]["label"]]
        form_data = [sentence, prediction]
        return render_template('data.html',form_data = form_data)
