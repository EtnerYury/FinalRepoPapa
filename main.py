import streamlit as st
import cv2
import numpy as np
import onnxruntime as rt
from rembg import remove, new_session
from PIL import Image
import os
import tempfile

# Configuración de la página
st.set_page_config(
    page_title="Clasificador de Imágenes",
    page_icon="📸",
    layout="centered"
)

# CSS para mejorar la apariencia en móvil
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        margin-bottom: 10px;
    }
    .uploadedFile {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# Inicializar la sesión de rembg
if 'rembg_session' not in st.session_state:
    st.session_state.rembg_session = new_session()

# Función para remover el fondo
def remove_background(input_image):
    if isinstance(input_image, np.ndarray):
        input_pil = Image.fromarray(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
    else:
        input_pil = input_image
    
    # Remover el fondo
    output_image = remove(input_pil, session=st.session_state.rembg_session)
    return output_image

def process_image_with_white_background(input_array):
    input_pil = Image.fromarray(cv2.cvtColor(input_array, cv2.COLOR_BGR2RGB))
    output_image = remove(input_pil, session=st.session_state.rembg_session)
    background = Image.new("RGBA", output_image.size, (255, 255, 255, 255))
    combined_image = Image.alpha_composite(background, output_image)
    final_array = cv2.cvtColor(np.array(combined_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    return final_array

def extract_features(image_path, fixed_size=(350, 450)):
    try:
        image = cv2.imread(image_path)
        if image is None:
            st.error("No se pudo leer la imagen")
            return None
            
        image = cv2.resize(image, fixed_size, interpolation=cv2.INTER_AREA)
        image = process_image_with_white_background(image)
        
        # Extracción de características
        def fd_histogram(image, bins=8):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], 
                              [0, 256, 0, 256, 0, 256])
            cv2.normalize(hist, hist)
            return hist.flatten()

        def fd_haralick(image):
            import mahotas
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            haralick = mahotas.features.haralick(gray).mean(axis=0)
            return haralick

        def fd_hu_moments(image):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            feature = cv2.HuMoments(cv2.moments(image)).flatten()
            return feature

        hist_features = fd_histogram(image)
        haralick_features = fd_haralick(image)
        hu_features = fd_hu_moments(image)

        global_feature = np.hstack([hist_features, haralick_features, hu_features])
        return global_feature.reshape(1, -1).astype(np.float32)
        
    except Exception as e:
        st.error(f"Error en extracción de características: {str(e)}")
        return None

def predict_with_onnx(features, model_path):
    try:
        sess = rt.InferenceSession(model_path)
        input_name = sess.get_inputs()[0].name
        label_name = sess.get_outputs()[0].name
        
        prediction = sess.run([label_name], {input_name: features})[0]
        return int(prediction[0]), 1.0
    except Exception as e:
        st.error(f"Error en predicción ONNX: {str(e)}")
        return None, None

# Interfaz de usuario
st.title("Clasificador de Imágenes 📸")

# Selector de entrada de imagen
source = st.radio("Selecciona la fuente de la imagen:", 
                 ["Cámara 📷", "Galería 🖼️"],
                 horizontal=True)

if source == "Cámara 📷":
    img_file = st.camera_input("Toma una foto")
else:
    img_file = st.file_uploader("Selecciona una imagen", type=['jpg', 'jpeg', 'png'])

if img_file is not None:
    # Crear dos columnas para mostrar las imágenes
    col1, col2 = st.columns(2)
    
    # Mostrar imagen original
    with col1:
        st.subheader("Imagen Original")
        st.image(img_file, caption="Original", use_column_width=True)
    
    # Procesar y mostrar imagen sin fondo
    with col2:
        st.subheader("Imagen sin Fondo")
        # Convertir el archivo a imagen PIL
        input_image = Image.open(img_file)
        # Remover el fondo
        output_image = remove_background(input_image)
        # Mostrar la imagen sin fondo
        st.image(output_image, caption="Sin fondo", use_column_width=True)
    
    # Botón de predicción
    if st.button("Predecir 🔍"):
        with st.spinner("Procesando imagen..."):
            # Guardar la imagen temporalmente
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                tmp_file.write(img_file.getvalue())
                temp_path = tmp_file.name

            # Extraer características
            features = extract_features(temp_path)
            
            if features is not None:
                # Realizar predicción
                model_path = "random_forest_model.onnx"  # Ajusta esto a la ruta de tu modelo
                prediction, confidence = predict_with_onnx(features, model_path)
                
                if prediction is not None:
                    # Mostrar resultados
                    st.success(f"Predicción: Clase {prediction}")
                    st.progress(confidence)
                    
            # Limpiar archivo temporal
            os.unlink(temp_path)

# Información adicional
with st.expander("ℹ️ Acerca de"):
    st.write("""
    Esta aplicación permite:
    1. Seleccionar una imagen de la galería o tomar una foto
    2. Ver la imagen original y sin fondo
    3. Realizar una predicción de la clase de la imagen utilizando un modelo de machine learning
    """)