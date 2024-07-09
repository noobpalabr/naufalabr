from flask import Flask, request, render_template
import pickle
import os

app = Flask(__name__)

# Load the clustering model, vectorizer, and sentiment model
with open('perum.pkl', 'rb') as model_file:
    clustering_model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

with open('sentimen.pkl', 'rb') as sentiment_file:
    sentiment_model = pickle.load(sentiment_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/clustering')
def clustering():
    return render_template('clusterperum.html')

@app.route('/cluster', methods=['POST'])
def cluster():
    input_data = [
        request.form['input1'], 
        request.form['input2'], 
        request.form['input3'], 
        request.form['input4']
    ]
    transformed_data = vectorizer.transform(input_data)
    cluster_result = clustering_model.predict(transformed_data)
    return render_template('clusterperum.html', result=cluster_result)

@app.route('/sentimen')
def sentimen():
    return render_template('sentimen.html')

@app.route('/analyze_sentimen', methods=['POST'])
def analyze_sentimen():
    text = request.form['text']
    transformed_text = vectorizer.transform([text])
    sentiment_result = sentiment_model.predict(transformed_text)
    return render_template('sentimen.html', result=sentiment_result)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
