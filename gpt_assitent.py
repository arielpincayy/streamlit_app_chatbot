import os
import openai
import pandas as pd
import streamlit as st

# 🧠 Título de la app
st.title("Asistente GPT sobre un Dataset")

# 🔐 Pedir clave de API
api_key = st.text_input("Introduce tu clave de OpenAI:", type="password")

# 📥 Pregunta del usuario
user_question = st.text_input("Haz una pregunta sobre el dataset:")

# 📄 Cargar y preparar el dataset
df = pd.read_csv("Dataset.csv").head(50)  # ⚠️ Reducido para evitar exceder tokens
context = df.to_string(index=False)

# ✅ Solo continuar si hay clave y pregunta
if api_key and user_question:
    openai.api_key = api_key

    prompt = f"""
    Tengo esta tabla de datos de restaurantes:

    {context}

    Responde la siguiente pregunta SOLO basándote en esta tabla. 
    Si no puedes responder, resume lo que hay en la base de datos, y di que no puedes resonder a la pregunta'.

    Pregunta: {user_question}
    Respuesta:"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300
        )
        answer = response["choices"][0]["message"]["content"]
        st.markdown(f"**Respuesta:** {answer}")

    except Exception as e:
        st.error(f"Error al consultar OpenAI: {e}")
