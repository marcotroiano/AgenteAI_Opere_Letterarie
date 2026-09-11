import os
from dotenv import load_dotenv
from google import genai

# Carichiamo la tua chiave gratuita dal file .env
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

print("📡 Connessione ai server Google in corso...")
print("📂 Modelli Generativi attualmente disponibili e attivi:\n")

# Chiediamo al server la lista di tutti i modelli che la tua chiave può usare
for modello in client.models.list():
    # Filtriamo solo i modelli che supportano la generazione di testo (generateContent)
    if 'generateContent' in modello.supported_actions:
        print(f"✅ {modello.name}")