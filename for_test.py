import pickle
import os

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import wordnet as wn


# LOAD MODEL
MODEL_VERSION = 'lstm_model_rus.h5'  # modèle
MODEL_PATH = os.path.join(os.getcwd(), 'models',
                          MODEL_VERSION)  # path vers le modèle
# model = load_model(MODEL_PATH, custom_objects={'MyOptimizer': Adam})
model = load_model(MODEL_PATH)  # chargement du modèle

# LOAD TOKENIZER
TOKENIZER_VERSION = 'tokenizer_rus.pickle'
TOKENIZER_PATH = os.path.join(os.getcwd(), 'models',
                              TOKENIZER_VERSION)  # path vers le tokenizer
with open(TOKENIZER_PATH, 'rb') as handle:
    tokenizer = pickle.load(handle)

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

# PREDICT


def my_predict(text):
    # Tokenize text
    text_pad_sequences = pad_sequences(tokenizer.texts_to_sequences(
        [text]), maxlen=300)
    # Predict
    predict_val = float(model.predict([text_pad_sequences]))
    recommandation = "Recommandé" if predict_val > 0.5 else "Non Recommandé"
    score = int(predict_val*100)
    return score, recommandation

# route recommandation par GET et POST


def predict():
    customer_feedback = "Mon texte de test"
    clean_comment = cleaning(customer_feedback)

    score, recommandation = my_predict(clean_comment)
    score = f"Note estimée : {score}/100"

    return customer_feedback, recommandation, score
