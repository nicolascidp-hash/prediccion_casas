from flask import Flask, render_template, request
from utils.model_loader import cargar_pipeline, cargar_feature_names, predecir
import os

app = Flask(__name__)

# Cargar modelo al iniciar
pipeline = cargar_pipeline()
feature_names = cargar_feature_names()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediccion', methods=['GET', 'POST'])
def prediccion():
    resultado = None
    error = None

    if request.method == 'POST':
        try:
            # 1. Obtener datos numéricos
            input_data = {
                'yearly_median_income': float(request.form.get('yearly_median_income')),
                'housing_median_age': float(request.form.get('housing_median_age')),
                'median_rooms': float(request.form.get('median_rooms')),
                'median_bedrooms': float(request.form.get('median_bedrooms')),
                'households': float(request.form.get('households'))
            }

            # 2. Manejar el One-Hot Encoding (CON ESPACIOS, COMO EN EL CSV)
            prox = request.form.get('ocean_proximity')
            
            # Inicializamos en 0 usando los nombres EXACTOS del CSV
            input_data['ocean_proximity_INLAND'] = 0
            input_data['ocean_proximity_ISLAND'] = 0
            input_data['ocean_proximity_NEAR BAY'] = 0  # <--- OJO: CON ESPACIO
            input_data['ocean_proximity_NEAR OCEAN'] = 0 # <--- OJO: CON ESPACIO
            
            # Activamos la selección
            if prox == 'INLAND': 
                input_data['ocean_proximity_INLAND'] = 1
            elif prox == 'ISLAND': 
                input_data['ocean_proximity_ISLAND'] = 1
            elif prox == 'NEAR_BAY': 
                # El HTML envía 'NEAR_BAY' (value del select), pero activamos la columna CON ESPACIO
                input_data['ocean_proximity_NEAR BAY'] = 1 
            elif prox == 'NEAR_OCEAN': 
                # El HTML envía 'NEAR_OCEAN', activamos la columna CON ESPACIO
                input_data['ocean_proximity_NEAR OCEAN'] = 1 
            
            # 3. Ordenar valores según lo que espera el modelo
            # (feature_names debe venir de tu JSON, que ojalá tenga los espacios también)
            valores = []
            for feature in feature_names:
                # Si el JSON tiene 'NEAR_BAY' (con guion), fallará aquí.
                # Asumimos que el JSON se generó bien desde el CSV (con espacios).
                valores.append(input_data[feature])

            # 4. Predecir
            resultado = predecir(pipeline, valores, feature_names)

        except Exception as e:
            error = f"Error procesando datos: {str(e)}"
            # Imprimir para depurar en los logs de Render si falla
            print(f"ERROR DATA: {input_data.keys()}")
            print(f"ERROR: {e}")

    return render_template('prediccion.html', resultado=resultado, error=error)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

