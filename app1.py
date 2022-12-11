from flask import Flask, request, render_template
# from flask_debugtoolbar import DebugToolbarExtension
import plotly
import plotly.express as px
import json
import pickle
import os
import pandas as pd

app = Flask(__name__)  # creation application
app.debug = True
# toolbar = DebugToolbarExtension(app)


MODEL_VERSION = 'model_LR'  # modèle
BDD_CLIENTS_VERSION = 'clean_df.csv'  # bdd

# Load dataset
bbd_clients_path = os.path.join(os.getcwd(), 'data',
                                BDD_CLIENTS_VERSION)  # path vers la bdd
BDD_CLIENTS = pd.read_csv(
    bbd_clients_path, index_col='SK_ID_CURR')  # import bdd

# load model
model_path = os.path.join(os.getcwd(), 'modele',
                          MODEL_VERSION)  # path vers le modèle
model = pickle.load(open(model_path, 'rb'))  # chargement du modèle


@app.route('/', methods=['GET', 'POST'])  # route homepage par GET et POST
def predict():

    if request.method == 'POST':
        if request.form['id_client']:
            id_client = int(request.form['id_client'])

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
