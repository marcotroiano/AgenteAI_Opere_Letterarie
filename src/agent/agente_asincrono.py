# ==============================================================================
# File: src/agent/agente_asincrono.py
# (SPOSTATO DALLA ROOT ALLA CARTELLA 'src/agent/' PER ARCHITETTURA MODULARE)
# OBIETTIVO: Orchestrazione del Paradigma RAG con Server OLLAMA Aziendale (On-Premise)
# ==============================================================================

"""
================================================================================
DOCUMENTAZIONE ARCHITETTURALE E MODULI DI STUDIO
================================================================================
RIFERIMENTI AL MANUALE (Inizio formale del Modulo 6):
- Modulo 6: Pipeline RAG (Fase di Generazione), Prompt Engineering e Asincronia.

Questo file rappresenta il "Cervello" del sistema e mette in pratica i concetti
fondamentali del Modulo 6, unendo il database vettoriale all'LLM generativo.

1. RUOLO E COMPITI TECNICI (COSA FA)
- Retrieval: Invoca il 'motore_ricerca.py' per estrarre i frammenti storici.
- Prompt Engineering: Costruisce una "Gabbia Semantica" (System Prompt) per
  istruire l'LLM a rispondere SOLO in base al testo recuperato e in formato JSON.
- Asincronia (async/await): Interroga il modello AI sul server senza bloccare l'applicativo.
- Validazione Strutturata: Inietta l'output grezzo dell'IA in Pydantic per
  garantire che il JSON finale sia perfetto.

2. SCELTE INGEGNERISTICHE E PATTERN
- PROGRAMMAZIONE ASINCRONA: Le chiamate di rete (VPN) verso il server aziendale
  possono avere latenza. L'uso di 'async/await' evita che l'interfaccia utente si
  "congeli" in attesa della risposta.
- TRANSIZIONE A OLLAMA E RICERCA IBRIDA NATIVA:
  Il sistema interroga esclusivamente il server Ollama interno all'azienda. Inoltre,
  il sistema chiama la nuova funzione di Ricerca Ibrida asincrona, beneficiando
  del Retrieval su SSD.
- PROMPT PER RISPOSTE DETTAGLIATE E ANTI-ALLUCINAZIONE ASSOLUTO:
  Con il passaggio a modelli generativi, il prompt è stato potenziato
  non solo per blindare le allucinazioni, ma anche per forzare il modello a
  valutare criticamente la pertinenza dei metadati prima di lanciarsi in sintesi
  fuori contesto.
================================================================================
"""

# ==============================================================================
# IMPORTAZIONI DI SISTEMA E LIBRERIE
# ==============================================================================
import os
import json
import asyncio  # Modulo nativo Python per gestire il multitasking cooperativo (Coroutines)
from dotenv import load_dotenv

# Importiamo 'Settings' da LlamaIndex per richiamare il modello Ollama
# che abbiamo configurato precedentemente nel file archivio_rag.py
from llama_index.core import Settings

# ==============================================================================
# MODIFICA ARCHITETTURALE: AGGIORNAMENTO PATH DEGLI IMPORT
# Poiché abbiamo strutturato il progetto in modo professionale (cartella 'src'),
# i moduli non si trovano più tutti nella stessa cartella disordinata.
# Ora usiamo i percorsi assoluti 'src.rag' e 'src.models' per richiamare le funzioni.
# ==============================================================================
# Importiamo la nuova funzione Ibrida asincrona dal sottomodulo RAG
from src.rag.motore_ricerca import esegui_ricerca_ibrida
# Importiamo il modello Pydantic dal sottomodulo MODELS
from src.models.modelli_output import AnalisiLetteraria

# ==============================================================================
# 1. INIZIALIZZAZIONE DELL'AMBIENTE E CONFIGURAZIONE CLIENT
# ==============================================================================
# Leggiamo il file .env (Non cerchiamo più GEMINI_API_KEY, perché Ollama aziendale
# gestisce l'accesso internamente sulla rete locale VPN).
load_dotenv()


# ==============================================================================
# 2. IL CERVELLO DELL'AGENTE (Funzione Asincrona)
# ==============================================================================
# 'async def' trasforma questa funzione in una Coroutina. Significa che può
# essere sospesa e ripresa senza bloccare il thread principale (es. la UI).
async def esegui_agente_rag(indice_rag, domanda_utente):
    print("\n" + "=" * 60)
    print("[AGENTE AI]: Avvio protocollo di analisi asincrona (Server Ollama)...")
    print("=" * 60)

    # --------------------------------------------------------------------------
    # FASE A: IL RETRIEVAL (Interrogazione del Database Vettoriale)
    # --------------------------------------------------------------------------
    print("[AGENTE AI]: Interrogazione dell'archivio storico (Qdrant) in corso...")

    # ==========================================================================
    # MODIFICA ARCHITETTURALE:
    # Invochiamo la nuova funzione 'esegui_ricerca_ibrida' (ora importata da src.rag)
    # in modo che il database su SSD (Qdrant) non faccia bloccare l'Event Loop di Python.
    # ==========================================================================
    dati_estratti = esegui_ricerca_ibrida(indice_rag, domanda_utente, top_k=7)

    # PRINCIPIO FAIL-FAST: Prevenzione assoluta delle allucinazioni a monte.
    # Se il motore di ricerca non trova nessun frammento rilevante, fermiamo
    # immediatamente l'agente restituendo None. L'LLM non viene nemmeno interpellato.
    if not dati_estratti:
        print("[AGENTE AI]: Nessun documento storico trovato. Interrompo la generazione.")
        return None

    # Estraiamo il testo puro e i metadati dal dizionario restituito da Qdrant
    testo_recuperato = dati_estratti['testo']
    metadati_recuperati = dati_estratti['metadati']

    # --------------------------------------------------------------------------
    # FASE B: PROMPT ENGINEERING (Costruzione del Contesto Semantico Anti-Allucinazione)
    # --------------------------------------------------------------------------
    print("[AGENTE AI]: Preparazione del Prompt Semantico Anti-Allucinazione...")

    # ==========================================================================
    # GABBIA SEMANTICA E ANTI-ALLUCINAZIONE ASSOLUTA
    # Per prevenire il fenomeno per cui il LLM inventa collegamenti fittizi
    # (es. attribuire l'Innominato a un romanzo sbagliato o ignorare i Metadati storici),
    # il prompt impone una rigida clausola di Valutazione della Pertinenza.
    # ==========================================================================
    prompt_di_sistema = f"""
    Sei un assistente esperto in analisi letteraria ad altissima precisione.
    Il tuo compito è rispondere alla domanda dell'utente basandoti ESCLUSIVAMENTE 
    sui seguenti 7 frammenti storici recuperati dal nostro archivio e sui loro METADATI.

    --- INIZIO FRAMMENTI STORICI ---
    {testo_recuperato}
    --- FINE FRAMMENTI STORICI ---

    METADATI DEI DOCUMENTI SORGENTE (Titolo, Autore, Anno):
    {metadati_recuperati}

    REGOLE FONDAMENTALI ANTI-ALLUCINAZIONE E FORMATTAZIONE (RISPETTALE RIGOROSAMENTE):
    1. VALUTAZIONE PRELIMINARE: Prima di rispondere, confronta la domanda dell'utente con i TESTI e con i METADATI (Titolo e Autore).
       Se i testi recuperati parlano di un argomento completamente slegato dalla domanda,
       oppure appartengono a un libro/autore chiaramente diverso da quello cercato, 
       DEVI DICHIARARE L'ASSENZA DELL'INFORMAZIONE. Non forzare collegamenti inventati.
    2. DIVIETO DI INVENZIONE: È SEVERAMENTE VIETATO usare le tue conoscenze generali o 
       inventare informazioni (allucinazioni) non presenti esplicitamente nel testo fornito.
    3. RISPOSTA POSITIVA: Se i frammenti contengono la risposta coerente, la tua 'sintesi_storica' 
       DEVE essere molto lunga, ricca di dettagli, discorsiva e argomentata. Analizza e collega i dettagli. 
       Non scrivere MAI una sola frase telegrafica.
    4. RISPOSTA NEGATIVA (SCUDO ATTIVO): Se il frammento non contiene la risposta o riguarda un romanzo errato:
       - Imposta "risposta_trovata" su false.
       - Imposta "sintesi_storica" su "Informazione non presente o non pertinente nei frammenti storici estratti. I documenti recuperati appartengono a contesti differenti."
       - Imposta "concetti_chiave" come lista vuota [].
       - Imposta "affidabilita_risposta" su 0.
    5. Rispondi ESCLUSIVAMENTE con un oggetto JSON valido. Non includere alcuna parola,
       spiegazione, commento o testo prima della parentesi graffa iniziale {{ o dopo quella finale }}.

    SCHEMA JSON OBBLIGATORIO:
    {{
        "risposta_trovata": true,
        "sintesi_storica": "La tua risposta molto dettagliata, ricca e argomentata basata esclusivamente sul testo.",
        "concetti_chiave": ["concetto1", "concetto2"],
        "affidabilita_risposta": 9
    }}

    DOMANDA DELL'UTENTE: {domanda_utente}
    """

    # --------------------------------------------------------------------------
    # FASE C: LA GENERAZIONE ASINCRONA (Chiamata API verso OLLAMA)
    # --------------------------------------------------------------------------
    print("[AGENTE AI]: Connessione asincrona al server aziendale Ollama in corso...")

    # Avvolgiamo la chiamata di rete in un blocco try/except.
    try:
        # Recuperiamo il motore LLM (Ollama) che abbiamo configurato precedentemente
        # nel file archivio_rag.py e salvato globalmente in Settings.
        llm_aziendale = Settings.llm

        if not llm_aziendale:
            raise ValueError("Errore: Motore LLM Ollama non trovato in Settings.")

        # ==================================================================
        # MODALITÀ PRODUZIONE: Interazione reale con Ollama
        # ==================================================================
        # 'await' sospende l'esecuzione liberando il processore finché
        # il server aziendale non ha finito di generare la risposta.
        # Usiamo .acomplete() (Asynchronous Complete) nativo di LlamaIndex.
        risposta_api = await llm_aziendale.acomplete(prompt_di_sistema)

        # 4. ESTRAZIONE E ISOLAMENTO DEL PAYLOAD JSON
        # La risposta dell'LLM arriva come oggetto: estraiamo la stringa testuale.
        risposta_testuale = risposta_api.text.strip()

        # ESTRAZIONE ROBUSTA (Resilienza ai preamboli dell'LLM):
        # Cerchiamo la prima graffa aperta '{' e l'ultima chiusa '}' nel testo.
        # Questo elimina automaticamente blocchi markdown ```json, tag di pensiero
        # o eventuali frasi di cortesia inserite dal modello.
        inizio_json = risposta_testuale.find("{")
        fine_json = risposta_testuale.rfind("}")

        if inizio_json != -1 and fine_json != -1 and fine_json > inizio_json:
            json_grezzo_llm = risposta_testuale[inizio_json:fine_json + 1]
        else:
            # Fallback alla stringa originale se le parentesi non sono chiaramente identificate
            json_grezzo_llm = risposta_testuale

        print("✅ [AGENTE AI]: Risposta JSON ricevuta dal Server. Avvio validazione...")

        # ----------------------------------------------------------------------
        # FASE D: VALIDAZIONE STRUTTURALE (Lo Scudo Pydantic)
        # ----------------------------------------------------------------------
        # Diamo in pasto la stringa JSON generata dall'IA al nostro validatore Pydantic.
        # model_validate_json() si assicura che ci siano tutti i campi richiesti e
        # che i tipi di dato (bool, Optional[str], list, Optional[int]) siano corretti.
        report_strutturato = AnalisiLetteraria.model_validate_json(json_grezzo_llm)

        # Restituiamo l'oggetto validato (AnalisiLetteraria) al main.py
        return report_strutturato

    except Exception as e:
        # Intercettiamo qualsiasi errore (es. JSON malformato da Ollama, VPN down)
        print(f"\n❌ [SECURITY BLOCK]: Errore durante la generazione o validazione LLM:\n{e}")
        # In caso di errore stampa anche la risposta grezza per capire cosa ha sbagliato Ollama
        if 'json_grezzo_llm' in locals():
            print(f"--- DUMP OUTPUT GREZZO ---\n{json_grezzo_llm}\n--------------------------")
        elif 'risposta_testuale' in locals():
            print(f"--- DUMP RISPOSTA INTEGRALE ---\n{risposta_testuale}\n--------------------------")
        return None