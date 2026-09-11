# ==============================================================================
# File: src/rag/motore_ricerca.py
# (SPOSTATO NELLA CARTELLA 'src/rag/' PER ARCHITETTURA MODULARE)
# OBIETTIVO: Gestire la logica di interrogazione ibrida nativa (Retrieval).
# ==============================================================================

"""
    ================================================================================
    DOCUMENTAZIONE ARCHITETTURALE: MOTORE DI RICERCA IBRIDO (RETRIEVER)
    ================================================================================
    RIFERIMENTI AL MANUALE:
    - Modulo 2 e 5: Retrieval e Similarità Cosenica.
    - Modulo 7: Ricerca Ibrida Nativa (Sparse + Dense Vectors) e Storage su File.
    - Appendice B.2: La Matematica della Pertinenza e il Reciprocal Rank Fusion (RRF).
    - Appendice B.6: La 'R' di RAG (Differenza tra Embedding e LLM).

    1. RUOLO E COMPITI TECNICI (COSA FA)
    Questo modulo isola la logica di interrogazione del database vettoriale.
    Riceve un indice vettoriale già costruito e una domanda in linguaggio naturale.
    Il suo compito è configurare il "Retriever", eseguire la ricerca e formattare
    i risultati estratti.

    2. SCELTE INGEGNERISTICHE E PATTERN (PERCHÉ È SCRITTA COSÌ)
    - SEPARATION OF CONCERNS E TYPE HINTING: Spostando questa logica fuori dal main.py,
      possiamo modificare l'algoritmo di ricerca senza toccare l'orchestratore.
      Abbiamo introdotto i Type Hints per dichiarare chiaramente input e output.

    - LA SOLUZIONE IBRIDA NATIVA (HYBRID SEARCH SU DATABASE):
      Abbiamo superato i limiti della ricerca puramente semantica (che non riusciva
      a identificare correttamente Nomi Propri o Parole Chiave specifiche) attivando
      la modalità "hybrid". In questa modalità avanzata, diciamo a LlamaIndex di far
      fare tutto il lavoro al database Qdrant. Qdrant cercherà autonomamente i
      vettori Densi (significato semantico) e i vettori Sparsi (parole chiave esatte),
      fondendo i punteggi in un'unica classifica in modo nativo ed estremamente veloce.

    - GESTIONE DEL THREADING (IL LUCCHETTO DI QDRANT):
      Qdrant Locale applica un 'LockFile' (lucchetto) ferreo al database per evitare
      corruzioni nei dati. Poiché lavoriamo su Disco (SSD) e non tramite Docker, le
      interrogazioni (query) al database devono essere strettamente SINCRONE per evitare
      che due thread provino a forzare il lucchetto contemporaneamente (Errore AlreadyLocked).
      L'asincronia verrà gestita esclusivamente sulle chiamate di rete remote (LLM).
    ================================================================================
"""

# Importiamo Dict, Optional e Any per il Type Hinting formale
from typing import Dict, Optional, Any

# ==============================================================================
# MODIFICA ARCHITETTURALE: IL RITORNO AL SINCRONISMO PER IL FILE-LOCK
# Questa funzione DEVE essere sincrona ('def' invece di 'async def').
# L'interrogazione asincrona ('aretrieve') su un database locale gestito da Portalocker
# andrebbe in crash per accesso concorrente.
# ==============================================================================

# MODIFICA: Aggiunti i Type Hints alla funzione (indice_rag di tipo Any perché dinamico)
def esegui_ricerca_ibrida(indice_rag: Any, domanda: str, top_k: int = 7) -> Optional[Dict[str, str]]:
    """
    Esegue un'interrogazione ibrida (Semantica + Lessicale) sul database vettoriale
    Qdrant in modalità sincrona e restituisce i risultati formattati.

    :param indice_rag: L'oggetto VectorStoreIndex contenente i dati vettorializzati.
    :param domanda: La stringa con la query dell'utente.
    :param top_k: Il numero di risultati massimi da restituire (default 7).
    :return: Un dizionario contenente 'testo' e 'metadati', oppure None se vuoto.
    """
    # --------------------------------------------------------------------------
    # FASE 6: IL RETRIEVAL (RICERCA IBRIDA NATIVA: SEMANTICA + KEYWORD)
    # --------------------------------------------------------------------------
    print("\n FASE 6: Esecuzione di una Ricerca Ibrida Nativa (Qdrant Dense + Sparse)...")

    # ==========================================================================
    # 1. IMPOSTAZIONE DEL FILTRO E DEL MODELLO
    # Creiamo il motore. Dicendo similarity_top_k=top_k (che vale 7),
    # gli diciamo: "Preparati a restituire una LISTA con i 7 risultati migliori".
    #
    # MAGIA ARCHITETTURALE: Aggiungendo vector_store_query_mode="hybrid", inviamo
    # a Qdrant l'ordine di usare simultaneamente i vettori semantici e l'algoritmo
    # BM25 (vettori sparsi), fondendo poi i punteggi (RRF).
    # ==========================================================================
    try:
        motore_di_ricerca = indice_rag.as_retriever(
            similarity_top_k=top_k,
            vector_store_query_mode="hybrid"
        )

        print(f"Domanda dell'utente: '{domanda}'")

        # ==========================================================================
        # 2. IL RECUPERO DEI DATI (SINCRONO PER SICUREZZA)
        # Usiamo '.retrieve()' in modalità sincrona. Qdrant leggerà i dati
        # dall'SSD ad una velocità spaventosa e ce li restituirà, senza mai forzare
        # o rompere il lucchetto del database locale.
        # Esempio visivo di cosa c'è ora dentro 'risultati': [Nodo_A, Nodo_B, Nodo_C]
        # ==========================================================================
        risultati = motore_di_ricerca.retrieve(domanda)

        # 3. VERIFICA DI SICUREZZA
        # Controlliamo che lo scatolone 'risultati' non sia vuoto (len > 0).
        if len(risultati) > 0:
            print(f"\n 🎯 TROVATI {len(risultati)} RISULTATI PERTINENTI:")

            # 4. PREPARAZIONE DEI CONTENITORI VUOTI
            # Siccome il LLM vuole ricevere in input una sola grande stringa di testo,
            # creiamo una variabile testuale vuota ("") che riempiremo man mano.
            testi_concatenati = ""
            # I metadati invece li raccogliamo in una lista vuota ([]).
            metadati_concatenati = []

            # ======================================================================
            # 5. IL CICLO FOR (L'Iterazione sui risultati fusi)
            # ======================================================================
            # 'enumerate' prende la lista [Nodo_A, Nodo_B, Nodo_C] e restituisce:
            # 1. 'i': Un numero contatore (da 1 a N).
            # 2. 'frammento': Il singolo oggetto estratto.
            for i, frammento in enumerate(risultati, start=1):
                # Poiché Qdrant usa la modalità Ibrida, .score conterrà il
                # punteggio ricalcolato della Reciprocal Rank Fusion.
                score = frammento.score if frammento.score is not None else 0.0

                """
                ========================================================================
                NOTA TEORICA PER LA TESI: L'EVOLUZIONE DELLO SCORE (DA COSENO A RRF)
                ========================================================================
                Nella versione semantica pura, questo numero rappresentava esclusivamente
                la Distanza Cosenica nello spazio multidimensionale.

                LA SIMILARITÀ COSENICA E IL RETRIEVAL:
                1. La query dell'utente viene convertita in un vettore a 384 dimensioni.
                2. Il motore di ricerca confronta il vettore della query con tutti
                   i vettori dei frammenti storici salvati in memoria.
                3. La pertinenza semantica viene stabilita misurando l'angolo tra i vettori.
                4. Più l'angolo è stretto, più l'affinità semantica è alta.

                IL NUOVO SCORE IBRIDO NATIVO (RRF):
                Attivando l'opzione Hybrid, lo score visualizzato è ora il risultato 
                dell'algoritmo interno di Reciprocal Rank Fusion (RRF) applicato dal database.
                Il sistema calcola un rank per l'affinità vettoriale e un rank per l'esattezza
                lessicale (BM25 Sparse Vectors), sommandoli e normalizzandoli.
                Questo penalizza i "falsi amici" semantici e premia i documenti che 
                comprendono il senso logico E contengono le parole chiave esatte.
                ========================================================================
                """

                print(f"\n--- Documento {i} (Score Ibrido DB: {score:.4f}) ---")

                # Stampiamo i primi 150 caratteri per mantenere pulito il terminale
                testo_anteprima = frammento.text[:150].replace('\n', ' ')
                print(f"Testo: {testo_anteprima}...")
                print(f"Metadati: {frammento.metadata}")

                # COSTRUZIONE DELLA STRINGA GIGANTE PER L'LLM
                # Passo A: Etichetta d'inizio
                testi_concatenati += f"\n--- [INIZIO DOCUMENTO {i}] ---\n"
                # Passo B: Testo
                testi_concatenati += f"{frammento.text}\n"
                # Passo C: Etichetta di fine
                testi_concatenati += f"--- [FINE DOCUMENTO {i}] ---\n"

                # SALVATAGGIO DEI METADATI IN CODA
                metadati_concatenati.append(frammento.metadata)

            # ======================================================================
            # AGGIORNAMENTO MODULO 6: RESTITUZIONE DEL DATO (IL TOOL AI)
            # Invece di far morire il dato in una stampa a schermo, lo impacchettiamo
            # in un dizionario strutturato e lo restituiamo. Ora l'Agente LLM potrà
            # leggerlo in memoria e usarlo come "Contesto" per la sua risposta.
            # ======================================================================
            return {
                "testo": testi_concatenati,
                "metadati": str(metadati_concatenati)
            }
        else:
            print("Nessun risultato pertinente trovato nel database.")
            return None

    except Exception as e:
        print(f"\n❌ Errore critico nel motore di ricerca ibrido: {e}")
        return None