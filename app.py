import pickle
import pandas as pd
import torch
from flask import render_template, request, Flask
from pattern.text import Sentence
from pattern.text.en import sentiment, parse, modality
from transformers import AutoTokenizer, AutoModel

MODEL_PATH = 'data/finalized_model.pkl'
file = open(MODEL_PATH, 'rb')
model_clf = pickle.load(file)


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def preprocess(sentences):
    # Tokenize sentences
    tokenizer = AutoTokenizer.from_pretrained("mrm8488/bert-tiny-finetuned-squadv2")
    model = AutoModel.from_pretrained("mrm8488/bert-tiny-finetuned-squadv2")

    encoded_input = tokenizer(sentences.to_list(), padding=True, truncation=True, max_length=128, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
        # Perform pooling. In this case, mean pooling
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    sentiment_train = sentences.apply(lambda x: sentiment(x))
    sentiment_train = pd.DataFrame(sentiment_train.values.tolist(),
                                   columns=['polarity', 'subjectivity'],
                                   index=sentences.index)
    parse_s = sentences.apply(lambda x: parse(x, lemmata=True))
    sent = parse_s.apply(lambda x: Sentence(x))
    modality_s = pd.DataFrame(sent.apply(lambda x: modality(x)))

    meta_df = sentiment_train.merge(modality_s, left_index=True, right_index=True)
    input_matrix = pd.concat([meta_df.reset_index(drop=True), pd.DataFrame(sentence_embeddings)], axis=1)

    return input_matrix


app = Flask(__name__)


@app.route('/')
def home():
    """ This is the homepage of our API.
    It can be accessed by http://127.0.0.1:5000/
    """
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    To make a prediction on one sample of the text
    satire or fake news
    :return: a result of prediction in HTML page
    """

    res = ''
    if request.method == 'POST':
        message = request.form['message']
        data = pd.Series(message)
        vect = preprocess(data)
        prediction = model_clf.predict(vect)
        output = prediction[0]
        if output == 0:
            my_prediction = "Fake News"
        else:
            my_prediction = "Satire"
        res = render_template('result.html', prediction=my_prediction)
        return res


if __name__ == '__main__':
    app.run(debug=True)

