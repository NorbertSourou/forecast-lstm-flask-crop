import os

from flask import Flask, render_template, send_from_directory, url_for, jsonify, request, redirect, flash
import tensorflow as tf
from joblib import dump, load
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# from keras.models import Sequential
# from keras.layers import LSTM, Dense, Dropout
# import keras
# from keras.models import load_model
# model = tf.keras.models.load_model('E:/Projects/python/flask_project/flaskr/static/model/lstm_best.keras',
#                                    compile=False)
# scaler = load('E:/Projects/python/flask_project/flaskr/static/model/scaler.joblib')
# df = pd.read_csv('E:/Projects/python/flask_project/flaskr/static/model/data.csv', index_col='date',
#                  parse_dates=True)
# sequence_length = 12
# scaled_data = scaler.transform(df[['prix', 'precipitation', 'tmax', 'tmin']])

# Charger les données
df1 = pd.read_csv('E:/Projects/python/flask_project/flaskr/static/model/data_train-Copie.csv', delimiter=';')
df2 = pd.read_csv('E:/Projects/python/flask_project/flaskr/static/model/data_test-Copie.csv', delimiter=';')
df = pd.concat([df1, df2])

# Convertir 'year' et 'month' en un seul champ datetime
df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
df.set_index('date', inplace=True)
df.drop(columns=['year', 'month'], inplace=True)

df3 = pd.read_csv('E:/Projects/python/flask_project/flaskr/static/model/precipitation.csv', delimiter=',')
# change columns name
# df3=df3.rename(columns={"mois":"month","valeur":"precipitation","zone":"commune"})
# for replacing
# df3['commune']=df3['commune'].str.replace('_AEROPORT','')

df3['commune'] = df3['commune'].str.replace('KANDI_AEROPORT', 'Banikoara')
df3['commune'] = df3['commune'].str.capitalize()
df3['date'] = pd.to_datetime(df3[['year', 'month']].assign(day=1))
df3.set_index('date', inplace=True)
df3.drop(columns=['year', 'month'], inplace=True)

df4 = pd.read_csv('E:/Projects/python/flask_project/flaskr/static/model/tmax.csv', delimiter=',')
# change columns name
# df3=df3.rename(columns={"mois":"month","valeur":"precipitation","zone":"commune"})
# for replacing
# df3['commune']=df3['commune'].str.replace('_AEROPORT','')

df4['commune'] = df4['commune'].str.replace('KANDI_AEROPORT', 'Banikoara')
df4['commune'] = df4['commune'].str.capitalize()
df4['date'] = pd.to_datetime(df4[['year', 'month']].assign(day=1))
df4.set_index('date', inplace=True)
df4.drop(columns=['year', 'month'], inplace=True)

df5 = pd.read_csv('E:/Projects/python/flask_project/flaskr/static/model/tmin.csv', delimiter=',')
# change columns name
# df3=df3.rename(columns={"mois":"month","valeur":"precipitation","zone":"commune"})
# for replacing
# df3['commune']=df3['commune'].str.replace('_AEROPORT','')

df5['commune'] = df5['commune'].str.replace('KANDI_AEROPORT', 'Banikoara')
df5['commune'] = df5['commune'].str.capitalize()
df5['date'] = pd.to_datetime(df5[['year', 'month']].assign(day=1))
df5.set_index('date', inplace=True)
df5.drop(columns=['year', 'month'], inplace=True)
#

df = (pd.merge(df, df3, on=['date', 'commune']))
df = (pd.merge(df, df4, on=['date', 'commune']))
df = (pd.merge(df, df5, on=['date', 'commune']))

df = df[df['commune'] == 'Banikoara']
print(df)

# Supprimer les colonnes inutiles
df = df.drop(['departement'], axis=1)
df = df.drop(['commune'], axis=1)
df = df.drop(['marche'], axis=1)

# df.drop(columns=['year', 'month'], inplace=True)

# Vérifier les données
print(df.head())

# Normalisation des données
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# Convertir les données en DataFrame pour plus de clarté
scaled_data = pd.DataFrame(scaled_data, columns=df.columns, index=df.index)


# Définir la fonction pour créer des séquences
def create_sequences(data, sequence_length):
    sequences = []
    labels = []
    for i in range(len(data) - sequence_length):
        seq = data[i:i + sequence_length]
        label = data[i + sequence_length, 2]  # Prix du maïs
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)


# Longueur de la séquence
sequence_length = 12  # Utiliser les données des 12 derniers mois

# Créer les séquences
sequences, labels = create_sequences(scaled_data.values, sequence_length)

# Séparer en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, shuffle=False)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Construire le modèle LSTM
model = Sequential()
model.add(LSTM(units=10, return_sequences=True, input_shape=(sequence_length, X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=10, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mae')

# Entraîner le modèle
model.fit(X_train, y_train, epochs=100, batch_size=72, validation_data=(X_test, y_test))


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)

    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    @app.route("/preline.js")
    def serve_preline_js():
        return send_from_directory("../node_modules/preline/dist", "preline.js")

    @app.route("/apexcharts.js")
    def serve_apexcharts_js():
        return send_from_directory("../node_modules/apexcharts/dist", "apexcharts.js")

    @app.route("/apexcharts.css")
    def serve_apexcharts_css():
        return send_from_directory("../node_modules/apexcharts/dist", "apexcharts.css")

    @app.route("/lodash.js")
    def serve_lodash_js():
        return send_from_directory("../node_modules/lodash", "lodash.js")

    # a simple page that says hello
    @app.route('/login', methods=['GET', 'POST'])
    def login():
        data = request.form
        if request.method == 'POST':
            if data['email'] == 'souwouinnorbert@gmail.com' and data['password'] == 'password':
                # flash('Connexion réussie !', 'success')
                return redirect(url_for('home'))
            else:
                print('Échec de la connexion. Veuillez vérifier vos informations.')
                flash('Échec de la connexion. Veuillez vérifier vos informations.', 'danger')
        return render_template('login.html')

    @app.route('/register')
    def register():
        return render_template('register.html')

    @app.route('/home')
    def home():
        return render_template('dashboard.html', active_page='home')

    @app.route('/forecast')
    def forecast():
        try:
            # Obtenir la dernière séquence de données
            last_sequence = scaled_data.values[-sequence_length:]

            future_prices = []

            last_known_values = scaled_data.values[-1, 1:]

            for _ in range(24):  # Prévoir les 24 prochains mois (2 ans)
                input_seq = last_sequence[-sequence_length:]
                input_seq = np.expand_dims(input_seq, axis=0)
                predicted_price = model.predict(input_seq)
                future_prices.append(predicted_price[0, 0])
                # Ajouter la prévision aux futures données
                next_month_data = np.append(last_sequence[-1, :-1], predicted_price[0, 0])
                last_sequence = np.vstack([last_sequence, next_month_data])

            # Faire les prédictions
            # future_prices_scaled = predict_next_12_months(last_sequence)
            #
            # # Préparer les données pour l'inversion de la normalisation
            future_data_for_inverse = np.zeros((24, 4))
            future_data_for_inverse[:, 0] = future_prices
            future_data_for_inverse[:, 1:] = last_known_values

            # Inverser la normalisation
            future_prices = scaler.inverse_transform(future_data_for_inverse)[:, 0]

            # Créer les dates futures
            last_date = df.index[-1]
            future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=24, freq='ME')

            # Créer un dictionnaire avec les résultats
            predictions = {date.strftime('%d %B %Y'): price for date, price in zip(future_dates, future_prices)}
            print(predictions)
            date_list = [date.strftime('%d %B %Y') for date in future_dates]
            price_list = [float(price) for price in future_prices]
            return render_template('forecast.html', active_page='forecast', dates=date_list, prices=price_list)

        except Exception as e:
            return jsonify({'error': str(e)}), 400

    @app.route('/about')
    def about():
        return render_template('about.html', active_page='about')

    @app.route('/contact')
    def contact():
        return render_template('contact.html', active_page='contact')

    @app.route('/documentation')
    def documentation():
        return render_template('documentation.html', active_page='documentation')

    return app
