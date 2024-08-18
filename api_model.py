from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from flask_cors import CORS  # Importa CORS

app = Flask(__name__)

# Configura CORS para permitir solicitudes desde localhost:4200
CORS(app, resources={r"/predict": {"origins": "*"}})

# Cargar el modelo entrenado
model = joblib.load('random_forest_model.pkl')

# Valores originales para los encoders
genero_values = ['Masculino', 'Femenino']
tipo_caso_values = ['Civil', 'Penal', 'Laboral', 'Mercantil', 'Administrativo','Ambiental']  # Añade todos los tipos de caso posibles

# Inicializar y ajustar los label encoders
label_encoder_genero = LabelEncoder()
label_encoder_genero.fit(genero_values)

label_encoder_tipo_caso = LabelEncoder()
label_encoder_tipo_caso.fit(tipo_caso_values)

def preprocess_input_data(data):
    data['fecha_inicio'] = pd.to_datetime(data['fecha_inicio'])
    data['fecha_cierre'] = pd.to_datetime(data['fecha_cierre'])
    data['ultima_actividad'] = pd.to_datetime(data['ultima_actividad'])

    data['dias_inicio'] = (pd.Timestamp('now') - data['fecha_inicio']).dt.days
    data['dias_cierre'] = (pd.Timestamp('now') - data['fecha_cierre']).dt.days
    data['dias_ultima_actividad'] = (pd.Timestamp('now') - data['ultima_actividad']).dt.days

    data['genero'] = label_encoder_genero.transform(data['genero'])
    data['tipo_caso'] = label_encoder_tipo_caso.transform(data['tipo_caso'])

    expected_features = ['id_cliente', 'edad', 'genero', 'tipo_caso', 'dias_inicio', 'dias_cierre', 'dias_ultima_actividad']
    data = data[expected_features]

    return data

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.json
    input_df = pd.DataFrame(input_data)
    preprocessed_data = preprocess_input_data(input_df)
    prediction = model.predict(preprocessed_data)
    return jsonify({"prediction": int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
