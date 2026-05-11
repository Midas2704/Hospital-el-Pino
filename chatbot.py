import ollama


with open("recetas.txt", "r", encoding="utf-8") as archivo:
    recetas = archivo.read()

print("=== CHATBOT DE RECETAS CHILENAS ===")
print("Escribe 'salir' para terminar.\n")

while True:
    pregunta = input("Tú: ")

    if pregunta.lower() == "salir":
        print("Bot: ¡Hasta luego!")
        break

    prompt = f"""
Eres un chatbot especializado en recetas chilenas.

Debes responder SOLO usando la información entregada.

RECETAS:
{recetas}

Pregunta del usuario:
{pregunta}
"""

    respuesta = ollama.chat(
        model='llama3',
        messages=[
            {
                'role': 'user',
                'content': prompt
            }
        ]
    )

    print("\nBot:", respuesta['message']['content'])
    print()