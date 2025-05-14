import os
import sys
import warnings
import pandas as pd
import streamlit as st
from transformers import pipeline
import torch

# 🔧 Correcciones para evitar errores con torch.classes y Streamlit
sys.modules["torch.classes"] = torch.classes
os.environ["STREAMLIT_DISABLE_WATCHDOG_WARNINGS"] = "true"
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ⚙️ Cargar modelo
table_qa = pipeline("table-question-answering", model="google/tapas-base-finetuned-wtq", device=-1)

# 📄 Cargar dataset limitado
df = pd.read_csv("Dataset.csv").head(250)
df = df.astype(str)
table = df.to_dict(orient="records")

# 🖥️ Interfaz Streamlit
st.title("Chat experto en restaurantes del mundo")

user_question = st.text_input("Haz una pregunta:")

if user_question:
    answer = table_qa(table=table, query=user_question)
    st.markdown(f"**Pregunta:** {user_question}")
    st.markdown(f"**Respuesta:** {answer['answer']}")
