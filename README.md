# 📚 Agente AI Asincrono per l'Analisi di Opere Letterarie (RAG Ibrido)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-0.10+-lightgrey.svg)](https://www.llamaindex.ai/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Hybrid_Search-red.svg)](https://qdrant.tech/)
[![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-black.svg)](https://ollama.com/)

Questo progetto implementa una pipeline **RAG (Retrieval-Augmented Generation) di livello enterprise**, progettata per l'interrogazione semantica avanzata di un vasto corpus di opere letterarie storiche. 

Il sistema si basa su un'architettura **completamente asincrona** ed esegue l'inferenza LLM on-premise appoggiandosi a un server aziendale tramite VPN, garantendo la massima riservatezza dei dati e il superamento dei limiti di contesto.

## 🏗️ Architettura e Pilastri Ingegneristici

Il progetto è stato ingegnerizzato seguendo rigorosi pattern di programmazione orientata agli oggetti (OOP) e principi di *Defensive Programming*:

1. **Ingegneria del Software (OOP):** 
   - Forte utilizzo dell'**Astrazione** e del **Duck Typing** per la validazione dinamica dei documenti in ingresso (Fail-Fast).
   - Utilizzo della **Composizione (Has-a)**: gli oggetti `LibroStorico` inglobano dinamicamente istanze di `Frammento`, ottimizzando la memoria in vista del processo di chunking.
   - Piena aderenza al **Single Responsibility Principle (SRP)** tramite una netta separazione dei moduli (Agent, Models, RAG, Utils).

2. **Intelligenza Artificiale (RAG Ibrido e Asincrono):**
   - **Hybrid Search Nativa:** Il retrieval vettoriale non si limita alla Similarità Cosenica (Dense Vectors), ma fonde nativamente i punteggi con i vettori sparsi (BM25) tramite algoritmo RRF (Reciprocal Rank Fusion) delegato interamente al database Qdrant su SSD.
   - **Elaborazione Asincrona (`async/await`):** Il "cervello" dell'agente gestisce le chiamate di rete verso l'LLM aziendale in modalità cooperativa, mantenendo l'Event Loop attivo e prevenendo colli di bottiglia durante le inferenze pesanti.

3. **Strutturazione dell'Output (Pydantic):**
   - Implementazione di una "gabbia semantica" nel Prompt Engineering combinata con la validazione rigorosa degli schemi di output tramite `Pydantic`.
   - Il modello è costretto a restituire JSON validati, valutando metriche di affidabilità e azzerando attivamente le allucinazioni.

## 📂 Struttura del Progetto

```text
AgenteAI_Opere_Letterarie/
├── dataset/                  # Dataset JSON originali (esclusi dal versioning)
├── src/                      # Codice sorgente principale
│   ├── agent/                # Logica asincrona dell'Agente LLM
│   ├── models/               # Classi base, schemi Pydantic e contratti OOP
│   ├── rag/                  # Ingestion, Parser, Connessione Qdrant e Retrieval
│   └── utils/                # Script di diagnostica e utility di rete
├── .env.example              # Template delle variabili d'ambiente
├── main.py                   # Entry point e orchestratore della pipeline
└── requirements.txt          # Dipendenze del progetto