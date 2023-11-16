# Para crear el requirements.txt ejecutamos 
# pipreqs --encoding=utf8 --force

# Primera Carga a Github
# git init
# git add .
# git remote add origin https://github.com/nicoig/compara-tabla-nutricional.git
# git commit -m "Initial commit"
# git push -u origin master

# Actualizar Repo de Github
# git add .
# git commit -m "Se actualizan las variables de entorno"
# git push origin master

# En Render
# agregar en variables de entorno
# PYTHON_VERSION = 3.9.12

################################################


import streamlit as st
import base64
from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage, AIMessage
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from PIL import Image
from io import BytesIO

# Cargar las variables de entorno para las claves API
load_dotenv(find_dotenv())

# Función para codificar imágenes en base64
def encode_image(image_file):
    return base64.b64encode(image_file.read()).decode('utf-8')

# Funciones auxiliares para obtener información nutricional y realizar comparativa
def obtener_info_nutricional(image_base64, model):
    chain = ChatOpenAI(model=model, max_tokens=1024)
    msg = chain.invoke(
        [AIMessage(content="Identifique los elementos de la tabla nutricional de esta imagen."),
         HumanMessage(content=[{"type": "text", "text": "Analizar tabla nutricional."},
                               {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}])
        ]
    )
    return msg.content

def realizar_comparativa(info1, info2):
    chain = ChatOpenAI(model="gpt-3.5-turbo", max_tokens=1024)
    prompt = PromptTemplate.from_template(
        """
        A continuación, se presentan las tablas nutricionales de dos productos diferentes. Realiza una comparación detallada entre ambos, considerando aspectos como calorías, carbohidratos, proteínas, grasas, fibra y sodio. Basándote en esta comparación, proporciona un veredicto sobre cuál producto es más saludable o adecuado para ciertos objetivos nutricionales.

        Tabla Nutricional Producto 1:
        {tabla1}

        Tabla Nutricional Producto 2:
        {tabla2}

        Realiza la comparativa y proporciona un veredicto detallado:
        """
    )
    runnable = prompt | chain | StrOutputParser()
    comparativa = runnable.invoke({"tabla1": info1, "tabla2": info2})
    return comparativa

# Configuración de la aplicación Streamlit
st.title("Comparador de Tabla Nutricional IA")

st.markdown("""
    <style>
    .small-font {
        font-size:18px !important;
    }
    </style>
    <p class="small-font">Realiza una comparación de tus productos favoritos y descrube cual es el más adecuado para ti</p>
    """, unsafe_allow_html=True)

st.image('img/robot.jpg', width=350)

# Carga de imágenes en dos columnas
col1, col2 = st.columns(2)
with col1:
    uploaded_file1 = st.file_uploader("Carga la primera imagen", type=["jpg", "png", "jpeg"])
with col2:
    uploaded_file2 = st.file_uploader("Carga la segunda imagen", type=["jpg", "png", "jpeg"])

if st.button("Comparar") and uploaded_file1 is not None and uploaded_file2 is not None:
    with st.spinner('Cargando...'):
        image1 = encode_image(uploaded_file1)
        info_nutricional1 = obtener_info_nutricional(image1, "gpt-4-vision-preview")
        st.markdown("**Tabla Nutricional Producto 1:**")
        st.write(info_nutricional1)

        image2 = encode_image(uploaded_file2)
        info_nutricional2 = obtener_info_nutricional(image2, "gpt-4-vision-preview")
        st.markdown("**Tabla Nutricional Producto 2:**")
        st.write(info_nutricional2)

        comparativa = realizar_comparativa(info_nutricional1, info_nutricional2)
        st.markdown("**Comparativa y Veredicto:**")
        st.write(comparativa)

        st.session_state['analisis'] = f"{info_nutricional1}\n\n{info_nutricional2}\n\n{comparativa}"

if 'analisis' in st.session_state:
    st.download_button("Descargar Análisis", st.session_state['analisis'], "analisis.txt", "text/plain")
