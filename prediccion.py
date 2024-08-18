import joblib

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar el archivo CSV
data = pd.read_csv('clientes2.csv')

# Preprocesamiento de datos
label_encoder = LabelEncoder()
data['genero'] = label_encoder.fit_transform(data['genero'])
data['tipo_caso'] = label_encoder.fit_transform(data['tipo_caso'])

data['fecha_inicio'] = pd.to_datetime(data['fecha_inicio'])
data['fecha_cierre'] = pd.to_datetime(data['fecha_cierre'])
data['ultima_actividad'] = pd.to_datetime(data['ultima_actividad'])

data['dias_inicio'] = (pd.Timestamp('now') - data['fecha_inicio']).dt.days
data['dias_cierre'] = (pd.Timestamp('now') - data['fecha_cierre']).dt.days
data['dias_ultima_actividad'] = (pd.Timestamp('now') - data['ultima_actividad']).dt.days

data = data.drop(columns=['nombre', 'apellido', 'telefono', 'correo_electronico',
                          'fecha_inicio', 'fecha_cierre', 'ultima_actividad'])

X = data.drop(columns=['deserto'])
y = data['deserto']

# Balanceo de clases con SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Dividir datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Entrenar el modelo con RandomForest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predecir en el conjunto de prueba
y_pred = model.predict(X_test)

# Evaluar el modelo
print(classification_report(y_test, y_pred))
roc_auc = roc_auc_score(y_test, y_pred)
print(f'ROC-AUC Score: {roc_auc}')

# Generar la matriz de confusión
cm = confusion_matrix(y_test, y_pred)

# Visualizar la matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
joblib.dump(model, 'random_forest_model.pkl')
# Cargar el modelo entrenado
model = joblib.load('random_forest_model.pkl')

# Obtener los nombres de las características del modelo entrenado
feature_names = model.feature_names_in_
print("Feature names used during training:", feature_names)

