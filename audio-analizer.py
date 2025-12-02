# predict.py

import argparse
import numpy as np
import librosa
import tensorflow as tf
import pickle # Para cargar metadatos
import os

# --- PARÁMETROS GLOBALES (Deben coincidir con el entrenamiento) ---
SR = 16000
DURATION = 3.0
N_MFCC_BASE = 13
HOP_LENGTH = 160
N_TIMESTEPS = 297
NUM_MFCCS = N_MFCC_BASE * 3

# --- 1. FUNCIÓN DE EXTRACCIÓN (extract_mfccs) ---
def extract_mfccs(audio_path: str) -> np.ndarray:
    """Carga audio, estandariza duración y extrae MFCCs (39)."""
    longitud_en_muestras = int(DURATION * SR)
    y, sr = librosa.load(audio_path, sr=SR)
    y_estandarizado = librosa.util.fix_length(y, size=longitud_en_muestras)
    mfccs = librosa.feature.mfcc(y=y_estandarizado, sr=SR, n_mfcc=N_MFCC_BASE, hop_length=HOP_LENGTH)
    mfccs_delta = librosa.feature.delta(mfccs)
    mfccs_delta2 = librosa.feature.delta(mfccs, order=2)
    mfccs_completos = np.concatenate((mfccs, mfccs_delta, mfccs_delta2), axis=0)
    mfccs_fijo = librosa.util.fix_length(mfccs_completos, size=N_TIMESTEPS, axis=1)
    return mfccs_fijo.T 

# --- 2. FUNCIÓN DE CARGA DE METADATOS ---
def load_metadata():
    # Carga la media, desviación estándar y nombres de clase que guardaste.
    # Asegúrate de que estas rutas sean correctas.
    try:
        GLOBAL_MEAN = np.load('model_metadata/global_mean.npy')
        GLOBAL_STD = np.load('model_metadata/global_std.npy')
        with open('model_metadata/class_names.pkl', 'rb') as f:
            CLASS_NAMES = pickle.load(f)
        unique_emotions = sorted(list(CLASS_NAMES.values()))
        return GLOBAL_MEAN, GLOBAL_STD, unique_emotions
    except FileNotFoundError as e:
        print(f"Error: No se encontró el archivo de metadatos: {e}")
        exit()

# --- 3. FUNCIÓN PRINCIPAL DE CLASIFICACIÓN ---
def classify_audio_cli(audio_path, model, mean, std, class_names):
    
    # 3.1. Validación de Archivo
    if not os.path.exists(audio_path):
        print(f"Error: Archivo no encontrado en la ruta: {audio_path}")
        return

    print(f"\nProcesando audio: {audio_path}")

    # 3.2. Preprocesamiento
    # Obtener la matriz de MFCCs (297, 39)
    mfccs_matrix = extract_mfccs(audio_path)
    
    # 3.3. Normalización Z-Score
    # Aplicar la normalización con los parámetros de entrenamiento
    mfccs_norm = (mfccs_matrix - mean) / std
    
    # Keras espera un lote (batch): (1, 297, 39)
    input_tensor = np.expand_dims(mfccs_norm, axis=0) 
    
    # 3.4. Predicción
    predictions = model.predict(input_tensor, verbose=0)[0]
    
    # 3.5. Formato de Salida
    predicted_index = np.argmax(predictions)
    confidence = predictions[predicted_index]
    predicted_emotion = class_names[predicted_index]

    print("--- Resultado ---")
    print(f"Emoción Detectada: {predicted_emotion.upper()}")
    print(f"Confianza: {confidence:.4f}")

# --- 4. FUNCIÓN PRINCIPAL Y ARGS ---
if __name__ == "__main__":
    
    # Definir argumentos de línea de comandos
    parser = argparse.ArgumentParser(description="Clasificador de Emociones en Audio con Keras CLI.")
    parser.add_argument('audio_file', type=str, help='Ruta al archivo .wav que se desea clasificar.')
    args = parser.parse_args()
    
    # Cargar modelo y metadatos
    print("Cargando modelo y metadatos...")
    try:
        model = tf.keras.models.load_model('model-gru-augmented.keras') 
    except Exception as e:
        print(f"Error al cargar el modelo de Keras: {e}")
        exit()
        
    mean, std, class_names = load_metadata()
    
    # Ejecutar clasificación
    classify_audio_cli(args.audio_file, model, mean, std, class_names)