import streamlit as st
import ollama
import pandas as pd



df = pd.read_csv("bd_recetas_chile_2025_llm_recipes.csv")


# CONFIG PAGINA


st.set_page_config(
    page_title="Chatbot de Recetas",
    page_icon="🍽️",
    layout="centered"
)

st.title("🍽️ ChefBot AI")

st.markdown("""
###  Asistente Inteligente de Gastronomía

Puedes preguntar sobre:
- comidas chilenas
- recetas latinoamericanas
- platos tradicionales
- comidas vegetarianas
- recetas rápidas
- recomendaciones gastronómicas
""")

with st.sidebar:
    st.header("📚 Información")

    st.write("""
    Este chatbot utiliza:
    -  Ollama + Llama3
    -  Dataset gastronómico
    -  Memoria conversacional
    -  Búsqueda inteligente
    """)

    st.divider()

    st.write("Desarrollado por castaña.")


# MEMORIA DEL CHAT


if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

# Mostrar historial
for mensaje in st.session_state.mensajes:
    with st.chat_message(mensaje["role"]):
        st.markdown(mensaje["content"])


# INPUT USUARIO


pregunta = st.chat_input("Escribe tu pregunta...")

if pregunta:

    # Guardar mensaje usuario
    st.session_state.mensajes.append({
        "role": "user",
        "content": pregunta
    })

    # Mostrar mensaje usuario
    with st.chat_message("user"):
        st.markdown(pregunta)

    
    # BUSQUEDA INTELIGENTE
    

    recetas_encontradas = []

    pregunta_lower = pregunta.lower()

    for _, fila in df.iterrows():

        texto_completo = f"""
        {fila['title']}
        {fila['aliases']}
        {fila['category']}
        {fila['dish_type']}
        {fila['region']}
        {fila['diet_notes']}
        {fila['cultural_context']}
        """

        if any(palabra in texto_completo.lower() for palabra in pregunta_lower.split()):
            recetas_encontradas.append(texto_completo)

    contexto = "\n\n".join(recetas_encontradas[:5])

    
    # PROMPT IA
    

    prompt = f"""
Eres un chatbot experto en recetas y gastronomía.

Tu tarea es responder como un chef y asistente gastronómico profesional.

Responde:
- de forma amigable
- detallada
- clara
- ordenada
- usando emojis
- usando títulos
- separando información importante

Cuando hables de recetas intenta incluir:
- 🍽️ nombre
- 📍 origen o región
- ⏱️ tiempo
- ⭐ dificultad
- 🥘 tipo de plato
- 📖 descripción cultural
- 💡 recomendaciones o variaciones

IMPORTANTE:

Debes responder SOLO usando la información del contexto entregado.

NO inventes recetas.
NO uses conocimiento externo.
NO respondas preguntas fuera del dataset.

Si la información no existe en el contexto responde exactamente:

"No tengo información suficiente en mi base de recetas para responder esa pregunta."

Usa la información entregada.

Si no encuentras información exacta, responde educadamente.

CONTEXTO DE RECETAS:
{contexto}

CONVERSACIÓN PREVIA:
{st.session_state.mensajes}

PREGUNTA:
{pregunta}
"""

    
    # RESPUESTA OLLAMA
    

    respuesta = ollama.chat(
        model='llama3',
        messages=[
            {
                'role': 'user',
                'content': prompt
            }
        ]
    )

    texto_respuesta = respuesta['message']['content']

    # Mostrar respuesta
    with st.chat_message("assistant"):
        st.markdown(texto_respuesta)

    # Guardar respuesta
    st.session_state.mensajes.append({
        "role": "assistant",
        "content": texto_respuesta
    })