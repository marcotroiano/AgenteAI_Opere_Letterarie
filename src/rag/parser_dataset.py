# ==============================================================================
# File: src/rag/parser_dataset.py
# (SPOSTATO NELLA CARTELLA 'src/rag/' PER ARCHITETTURA MODULARE)
# ==============================================================================

"""
    ================================================================================
    DOCUMENTAZIONE ARCHITETTURALE: PARSER DINAMICO (DATA INGESTION)
    ================================================================================
    RIFERIMENTI AL MANUALE:
    - Modulo 4: Data Ingestion e pulizia dei dati.
    - Appendice A: Defensive Programming applicata al parsing JSON.

    1. RUOLO E COMPITI TECNICI (COSA FA)
    Questo modulo funge da estrattore e normalizzatore di dati (Data Engineer):
    - Scansione: Cerca e legge tutti i file JSON all'interno di una directory.
    - Pulizia (Data Cleansing): Estrae dinamicamente campi eterogenei e li
      normalizza (es. estrae l'anno da una data completa, formatta la lingua).
    - Fabbricazione: Usa i dati puliti per istanziare oggetti 'LibroStorico' e
      applica la Composizione iniettando il testo sotto forma di frammenti.

    2. SCELTE INGEGNERISTICHE E PATTERN (PERCHÉ È SCRITTA COSÌ)
    - SINGLE RESPONSIBILITY PRINCIPLE (SRP): Isolare questa logica dal main.py
      garantisce che l'orchestrazione AI non sia inquinata dalle operazioni di I/O
      (lettura dischi) o dalla gestione delle eccezioni sui file.
    - DEFENSIVE PROGRAMMING: I dataset reali sono sempre "sporchi". L'uso massiccio
      di 'isinstance', metodi '.get()' con fallback e blocchi 'try/except' garantisce
      che un file malformato venga ignorato silenziosamente senza far crashare
      l'intera pipeline RAG.
    ================================================================================
"""

import json
import os
from typing import List

# ==============================================================================
# MODIFICA ARCHITETTURALE: AGGIORNAMENTO PATH DEGLI IMPORT
# Importiamo lo schema dei dati usando il percorso assoluto dal package models
# ==============================================================================
from src.models.libro_storico import LibroStorico


# ==============================================================================
# PARSER DINAMICO (DATA INGESTION PIPELINE)
# ==============================================================================
# MODIFICA: Aggiunti Type Hints per indicare che la funzione restituisce una
# lista di oggetti LibroStorico.
def carica_dataset_json(percorso_cartella: str) -> List[LibroStorico]:
    """
    Parser per automatizzare l'Ingestion del dataset letterario.
    Legge tutti i file JSON in una cartella, ne estrae i metadati essenziali
    e il contenuto testuale, restituendo una lista di oggetti LibroStorico.
    """
    libri_caricati = []

    # 1. CONTROLLO DI SICUREZZA (Defensive Programming)
    # Verifichiamo che il percorso fornito esista per evitare crash a runtime
    if not os.path.exists(percorso_cartella):
        print(f" ERRORE: La cartella '{percorso_cartella}' non è stata trovata.")
        return libri_caricati

    # 2. SCANSIONE DELLA DIRECTORY
    for nome_file in os.listdir(percorso_cartella):
        if nome_file.endswith(".json"):
            percorso_completo = os.path.join(percorso_cartella, nome_file)

            # 3. APERTURA E LETTURA DEL FILE
            with open(percorso_completo, 'r', encoding='utf-8') as file_json:
                try:
                    # Carica il file JSON convertendolo in un dizionario Python
                    dati = json.load(file_json)

                    # ==================================================================
                    # 4. ESTRAZIONE DINAMICA DEI DATI E GESTIONE DEI "DIRTY DATA"
                    # I dataset reali presentano spesso formattazioni eterogenee.
                    # Applichiamo la "Defensive Programming" verificando dinamicamente
                    # il tipo di dato (con isinstance) prima di elaborarlo.
                    # ==================================================================

                    # --- ESTRAZIONE TITOLO ---
                    # Estrae il dato crudo. Se non esiste, assegna un valore di fallback.
                    titolo_raw = dati.get("title", "Titolo Sconosciuto")
                    # Se il titolo è stato salvato come una lista (es. ["I Promessi Sposi"]),
                    # prende solo il primo elemento [0]. Altrimenti, lo legge come stringa.
                    titolo = str(titolo_raw[0]) if isinstance(titolo_raw, list) else str(titolo_raw)

                    # --- ESTRAZIONE AUTORE ---
                    # L'autore nei JSON è spesso un nodo annidato (un dizionario dentro il dizionario).
                    dati_autore = dati.get("author", {})
                    # A volte potrebbe essere una lista di dizionari. Se lo è, prendiamo il primo.
                    if isinstance(dati_autore, list):
                        dati_autore = dati_autore[0] if dati_autore else {}
                    # Infine estraiamo il valore associato alla chiave "name".
                    autore = dati_autore.get("name", "Autore Ignoto") if isinstance(dati_autore,
                                                                                    dict) else "Autore Ignoto"

                    # --- ESTRAZIONE ANNO ---
                    anno_raw = dati.get("date", "0")
                    anno_str = str(anno_raw[0]) if isinstance(anno_raw, list) and anno_raw else str(anno_raw)
                    try:
                        # Molti file JSON salvano le date in formato completo (es. "1321-09-14").
                        # Usiamo lo slicing [:4] per estrarre forzatamente solo i primi 4 caratteri (l'anno),
                        # ed eseguiamo un casting (conversione) forzato in numero intero (int).
                        anno = int(anno_str[:4])
                    except ValueError:
                        # Se il dato è incomprensibile (es. "Sconosciuto"), il programma non va in crash
                        # ma assegna silenziosamente l'anno 0 di default.
                        anno = 0

                    # --- ESTRAZIONE LINGUA ---
                    lingua_raw = dati.get("language", "sconosciuta")
                    # Estraiamo la stringa, che sia in una lista o meno
                    lingua_str = str(lingua_raw[0]) if isinstance(lingua_raw, list) and lingua_raw else str(lingua_raw)
                    # Applichiamo il .lower() direttamente in fase di pulizia
                    lingua = lingua_str.strip().lower()

                    # ==================================================================
                    # --- NOVITÀ FASE 4: ESTRAZIONE METADATI FISICI E BIBLIOTECONOMICI ---
                    # Estraiamo i campi fisici per permettere all'Agente AI di indicare
                    # all'utente dove trovare i testi e come sono fatti. Usiamo la solita
                    # logica difensiva per evitare crash su campi mancanti o salvati in liste.
                    # ==================================================================

                    loc_raw = dati.get("location", "Posizione ignota")
                    location = str(loc_raw[0]) if isinstance(loc_raw, list) else str(loc_raw)

                    bind_raw = dati.get("binding", "N/D")
                    binding = str(bind_raw[0]) if isinstance(bind_raw, list) else str(bind_raw)

                    weight_raw = dati.get("weight", "N/D")
                    weight = str(weight_raw[0]) if isinstance(weight_raw, list) else str(weight_raw)

                    lines_raw = dati.get("lines", "N/D")
                    lines = str(lines_raw[0]) if isinstance(lines_raw, list) else str(lines_raw)

                    req_raw = dati.get("requests", "N/D")
                    requests = str(req_raw[0]) if isinstance(req_raw, list) else str(req_raw)

                    # --- ESTRAZIONE TESTO ---
                    testo_raw = dati.get("contenuto", "")
                    # Il testo potrebbe essere salvato come una lista di paragrafi separati.
                    # Se è una lista, uniamo (join) tutti i paragrafi in un'unica grande stringa
                    # separandoli con uno spazio " ". Altrimenti, lo castiamo direttamente a stringa.
                    testo_completo = " ".join([str(t) for t in testo_raw]) if isinstance(testo_raw, list) else str(
                        testo_raw)

                    # ==================================================================
                    # 5. ISTANZIAZIONE DELL'OGGETTO OOP (LibroStorico)
                    # Ora che i dati sono ripuliti e uniformati, popoliamo la nostra
                    # classe strutturale.
                    # ==================================================================
                    libro = LibroStorico(
                        titolo=titolo,
                        autore=autore,
                        anno=anno,
                        lingua=lingua
                    )

                    # ==================================================================
                    # 5.5 INIEZIONE DINAMICA DEGLI ATTRIBUTI FISICI (NOVITÀ FASE 4)
                    # Sfruttiamo la flessibilità di Python: aggiungiamo nuovi attributi
                    # all'oggetto appena creato "al volo", senza dover alterare la classe
                    # base LibroStorico. Così ArchivioRAG potrà trovarli con getattr()!
                    # ==================================================================
                    libro.location = location.strip()
                    libro.binding = binding.strip()
                    libro.weight = weight.strip()
                    libro.lines = lines.strip()
                    libro.requests = requests.strip()

                    # ==================================================================
                    # 6. COMPOSIZIONE DEI FRAMMENTI
                    # ==================================================================
                    # Verifichiamo che il testo non sia vuoto o composto solo da spazi bianchi (strip)
                    if testo_completo.strip():
                        # Usiamo la composizione: passiamo l'intero testo al libro.
                        # Sarà la classe LibroStorico, dietro le quinte, a creare gli
                        # oggetti 'Frammento' e ad assorbirli.
                        libro.aggiungi_frammento(testo_completo, 1)

                    # Aggiungiamo il libro appena "fabbricato" alla lista finale da restituire
                    libri_caricati.append(libro)

                except json.JSONDecodeError:
                    # Se il file fisicamente non è un JSON valido, lo saltiamo evitando il crash.
                    print(f" ATTENZIONE: Il file {nome_file} è corrotto o malformato.")

    return libri_caricati