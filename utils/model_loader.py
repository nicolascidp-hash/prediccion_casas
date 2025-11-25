import joblib
import json
import pandas as pd  # <--- AHORA NECESITAMOS PANDAS AQUÍ
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

def cargar_pipeline():
    ruta = os.path.join(BASE_DIR, 'models', 'pipeline_gb.pkl')
    return joblib.load(ruta)

def cargar_feature_names():
    ruta = os.path.join(BASE_DIR, 'models', 'feature_names.json')
    with open(ruta, 'r') as f:
        return json.load(f)

def predecir(pipeline, valores_en_orden, feature_names):
    """
    Recibe la lista de valores y la convierte a DataFrame
    para que el Pipeline encuentre los nombres de las columnas.
    """
    # CRÍTICO: Creamos un DataFrame con los nombres de las columnas
    input_df = pd.DataFrame([valores_en_orden], columns=feature_names)
    
    # Predecimos usando el DataFrame
    pred = pipeline.predict(input_df)[0]
    return float(pred)
