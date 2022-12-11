from flask import Flask, request, render_template
# from flask_debugtoolbar import DebugToolbarExtension
import os
import string
import pandas as pd
import pickle

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import wordnet as wn

app = Flask(__name__)  # creation application
app.debug = True
# toolbar = DebugToolbarExtension(app)
app.static_folder = 'static'

# LOAD MODEL
MODEL_VERSION = 'lstm_model_first.h5'  # modèle
MODEL_PATH = os.path.join(os.getcwd(), 'models',
                          MODEL_VERSION)   # path vers le modèle
MODEL = load_model(MODEL_PATH)  # chargement du modèle
# LOAD DATASET
BDD_VERSION = 'df_cleaned.csv'
BDD_PATH = os.path.join(os.getcwd(), 'data',
                        BDD_VERSION)  # path vers la bdd
BDD_CLIENTS = pd.read_csv(
    BDD_PATH)  # import bdd
# PREPROCESS TEXT

stop_words = stopwords.words('english')


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('V'):
        return wn.VERB
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    else:
        return wn.NOUN


def cleaning(data):

    # 1. Tokenize
    text_tokens = word_tokenize(data.replace("'", "").lower())

    # 2. Remove Puncs
    tokens_without_punc = [w for w in text_tokens if w.isalpha()]

    # 3. Removing Stopwords
    tokens_without_sw = [t for t in tokens_without_punc if t not in stop_words]

    # 4. Lemmatize
    POS_tagging = pos_tag(tokens_without_sw)
    wordnet_pos_tag = []
    wordnet_pos_tag = [(word, get_wordnet_pos(pos_tag))
                       for (word, pos_tag) in POS_tagging]
    wnl = WordNetLemmatizer()
    lemma = [wnl.lemmatize(word, tag) for word, tag in wordnet_pos_tag]

    return " ".join(lemma)


X_train_padseq = pad_sequences(
    tokenizer.texts_to_sequences(X_train), maxlen=300)
#pred_train_lstm = lstm_model.predict(X_train_padseq)

# PREDICT


def my_predict(text):
    # Tokenize text
    x_test = pad_sequences(tokenizer.texts_to_sequences(
        [text]), maxlen=300)
    # Predict
    score = lstm_model.predict([x_test])

    return {"Texte": text, "Recommendation": "Recommandé" if score > 0.5 else "Non Recommandé", "Score": float(score)}


@app.route('/', methods=['GET', 'POST'])  # route homepage par GET et POST
def predict():

    if request.method == 'POST':
        if request.form['customer_feedback']:
            customer_feedback = str(request.form['customer_feedback'])

            bdd_client = pd.DataFrame(BDD_CLIENTS.loc[id_client])
            # score_test = bdd_client.drop(bdd_client.columns[:2], axis=1)
            score_test = bdd_client.T

            predictions = model.predict_proba(score_test)

            score = round((predictions[0][0]), 2)*100

            prediction_cat = model.predict(score_test)

            prediction = 'Client avec des difficultés de payement' if prediction_cat[
                0] == 1 else 'Bon client'

            # Definition du df_radar avec les 3 scores client
            radar_ID = score_test[['EXT_SOURCE_3',
                                   'EXT_SOURCE_2', 'EXT_SOURCE_1']]
            # Récupération des scores du client X
            data = radar_ID.values.tolist()

            # Initialisation d'une liste vide
            flat_list = []
            # ittération dans data
            for item in data:
                # appending elements to the flat_list
                flat_list += item
            df = pd.DataFrame(dict(
                r=flat_list,
                theta=['EXT_SOURCE_3', 'EXT_SOURCE_2', 'EXT_SOURCE_1']))
            fig = px.line_polar(df, r="r", theta="theta", line_close=True)

            graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

            return render_template('index.html', graphJSON=graphJSON, text=f"Prédiction : {prediction}", score=f"Score : {score}/100", submission=f"Client n°{id_client}")
        # elif request.form['NAME_EDUCATION_TYPE'] and (request.form['male'] or request.form['female']) and request.form['EXT_SOURCE_1'] and request.form['EXT_SOURCE_2'] and request.form['EXT_SOURCE_3'] and request.form['AMT_ANNUITY'] and request.form['AMT_CREDIT'] and request.form['DAYS_EMPLOYED'] and request.form['AMT_GOODS_PRICE'] and request.form['DAYS_REGISTRATION'] and request.form['AMT_INCOME_TOTAL']:
            # else:
            # if request.form['NAME_EDUCATION_TYPE'] and (request.form['male'] or request.form['female']) and request.form['EXT_SOURCE_1'] and request.form['EXT_SOURCE_2'] and request.form['EXT_SOURCE_3'] and request.form['AMT_ANNUITY'] and request.form['AMT_CREDIT'] and request.form['DAYS_EMPLOYED'] and request.form['AMT_GOODS_PRICE'] and request.form['DAYS_REGISTRATION'] and request.form['AMT_INCOME_TOTAL']:
            # print("ok")
            # NAME_EDUCATION_TYPE = request.form['NAME_EDUCATION_TYPE']
            # GENDER = request.form['male'] if request.form['male'] else request.form['female']
            # EXT_SOURCE_1 = request.form['EXT_SOURCE_1']
            # EXT_SOURCE_2 = request.form['EXT_SOURCE_2']
            # EXT_SOURCE_3 = request.form['EXT_SOURCE_3']
            # AMT_ANNUITY = request.form['AMT_ANNUITY']
            # AMT_CREDIT = request.form['AMT_CREDIT']
            # DAYS_EMPLOYED = request.form['DAYS_EMPLOYED']
            # AMT_GOODS_PRICE = request.form['AMT_GOODS_PRICE']
            # DAYS_REGISTRATION = request.form['DAYS_REGISTRATION']
            # AMT_INCOME_TOTAL = request.form['AMT_INCOME_TOTAL']
            return render_template('index.html', text=f"OK", score=f"OK")

    if request.method == 'GET':
        return render_template('index.html')


if __name__ == '__main__':  # faire run l'application
    app.run(debug=True, use_debugger=True)
