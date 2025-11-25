import joblib
import json
import numpy as np
import os
import re

# Definir la ruta base del proyecto
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

def cargar_pipeline():
    """Carga el modelo usando joblib (compatible con scikit-learn)"""
    ruta = os.path.join(BASE_DIR, 'models', 'pipeline_gb.pkl')
    # Usamos joblib en lugar de pickle para evitar errores de compatibilidad
    pipeline = joblib.load(ruta)
    return pipeline

def cargar_feature_names():
    """Carga los nombres de las columnas desde el JSON"""
    ruta = os.path.join(BASE_DIR, 'models', 'feature_names.json')
    with open(ruta, 'r') as f:
        return json.load(f)

def predecir(pipeline, valores_en_orden, feature_names):
    """Realiza la predicción convirtiendo la lista a un array 2D"""
    # Convertir lista a array de numpy con forma (1, n_columnas)
    X_arr = np.array(valores_en_orden).reshape(1, -1)
    
    # Hacer la predicción
    pred = pipeline.predict(X_arr)[0]
    return float(pred)
