import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Hide deprecation warnings which directly don't affect the working of the application
import warnings
warnings.filterwarnings("ignore")

# Path to the icon image
icon_path = "LWC.png"

# Set some pre-defined configurations for the page
st.set_page_config(
    page_title="Reconocimiento de Productos de Góndola",
    page_icon=icon_path,
    initial_sidebar_state='auto'
)

# Custom CSS styling
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('./Gondola_model.h5')
    return model

with st.spinner('Modelo está cargando...'):
    model = load_model()

# Sidebar content
with st.sidebar:
    st.image('LWC.png')
    st.title("Bienvenido a UNAB IA: Reconocimiento de Objetos de Góndola")
    st.subheader("Reconocimiento de imágenes para productos de góndola")
    satisfaction = st.slider("¿Qué tan satisfecho estás con la aplicación?", 0, 100, 50)
    st.write(f"Nivel de satisfacción: {satisfaction}%")

st.markdown("<h1 style='text-align: center;'>UNAB IA</h1>", unsafe_allow_html=True)

# Set the background image
st.markdown("""
    <style>
    body {
        background-image: url("https://www.unired.edu.co/images/instituciones/UNAB/2017/junio/unab.jpg");
        background-size: cover;
        background-position: center;
    }
    </style>
""", unsafe_allow_html=True)

# Centered text with bold parts
html_text = """
<div style="text-align: center; color: white;">
    Nuestra aplicación utiliza tecnología avanzada de inteligencia artificial para <b>identificar y catalogar productos en góndolas</b> de manera rápida y precisa.
    Simplifica la gestión de inventarios y mejora la experiencia de compra con nuestra innovadora herramienta de reconocimiento visual.
    ¡Descubre una nueva forma de optimizar tu negocio con UNAB IA!
</div>
"""
st.markdown(html_text, unsafe_allow_html=True)

st.write("## Detección de productos")

def import_and_predict(image_data, model, class_names):
    image_data = image_data.resize((180, 180))
    image = tf.keras.utils.img_to_array(image_data)
    image = tf.expand_dims(image, 0)  # Create a batch

    prediction = model.predict(image)
    index = np.argmax(prediction)
    score = tf.nn.softmax(prediction[0])
    class_name = class_names[index]

    return class_name, score

class_names = open("./clases.txt", "r").readlines()

st.write("## Suba una foto o tome una nueva para identificar un producto")

# Options for image upload or capture
img_file_buffer = st.file_uploader("Subir una imagen", type=["jpg", "jpeg", "png"])
img_camera = st.camera_input("O tome una foto")

image = None  # Initialize the image variable

# Determine which input to use
if img_file_buffer is not None:
    image = Image.open(img_file_buffer)
elif img_camera is not None:
    image = Image.open(img_camera)

if image is not None:
    st.image(image, use_column_width=True)
    
    # Perform prediction
    class_name, score = import_and_predict(image, model, class_names)
    
    # Display result
    if np.max(score) > 0.5:
        st.subheader(f"Tipo de producto: {class_name}")
        st.text(f"Puntuación de confianza: {100 * np.max(score):.2f}%")
    else:
        st.text("No se pudo determinar el tipo de producto")
else:
    st.text("Por favor, suba una imagen o tome una foto.")

