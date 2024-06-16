from flask import Flask, request, render_template
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

#Loading the model
with open ('nlp_model.pkl', 'rb') as file:
    model = pickle.load(file)

#Loading the count Vectorizer
with open('count_vectorizer.pkl', 'rb') as file:
    cv = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        review = request.form['review']
        data = [review]
        vect = cv.transform(data).toarray()
        prediction = model.predict(vect)
        sentiment = 'Positive' if prediction[0] == 1 else 'Negative'
        return render_template('index.html', prediction=sentiment)

if __name__ == '__main__':
    app.run(debug=True)