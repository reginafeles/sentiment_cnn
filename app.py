from flask import Flask, render_template, request
from model.load import init, predict_sentiment

app = Flask(__name__)

model, tokenizer = init()

labels = {0: "Negative", 1: "Neutral", 2: "Positive"}


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    if request.method == 'POST':
        text = request.form['text']
        if text:
            class_idx, conf = predict_sentiment(text)
            prediction = labels.get(class_idx, "Unknown")
            confidence = round(conf * 100, 2)
    return render_template('index.html', prediction=prediction, confidence=confidence)


if __name__ == '__main__':
    app.run(debug=True)
