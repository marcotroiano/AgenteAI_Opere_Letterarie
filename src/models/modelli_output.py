# ==============================================================================
# File: src/models/modelli_output.py
# (SPOSTATO NELLA CARTELLA 'src/models/' PERCHÉ DEFINISCE GLI SCHEMI DATI)
# OBIETTIVO: Definire i contratti di output dell'LLM tramite Pydantic.
# ==============================================================================

"""
================================================================================
DOCUMENTAZIONE ARCHITETTURALE: MODELLI DI OUTPUT (PYDANTIC)
================================================================================
RIFERIMENTI AL MANUALE:
- Modulo 6: Pipeline RAG e Generazione Risposte.
- Appendice B.6: Il RAG spiegato (La 'G' di Generazione per output strutturati).

1. RUOLO E COMPITI TECNICI (COSA FA)
Questo modulo definisce la "gabbia" strutturale per le risposte del Large
Language Model. Invece di permettere all'IA di generare testo libero e
imprevedibile, la costringiamo a restituire un oggetto JSON che rispetti
esattamente i campi e i tipi di dato definiti in queste classi.

2. SCELTE INGEGNERISTICHE E PATTERN (PERCHÉ È SCRITTA COSÌ)
- TYPE HINTING E VALIDAZIONE A RUNTIME: Pydantic funge da scudo. Se l'LLM
  allucina e restituisce una stringa al posto di un booleano per il campo
  'risposta_trovata', Pydantic solleva un'eccezione, bloccando l'errore.
- STRUCTURED OUTPUT (Data Engineering): Garantisce che il nostro Agente
  Asincrono riceva sempre un formato standardizzato, facilitando l'estrazione
  delle informazioni e azzerando il rischio di risposte fuori contesto.
- TOLLERANZA AGLI ERRORI (Fault Tolerance): L'uso di 'Optional' e di valori di
  default garantisce che il programma non vada in crash qualora l'LLM (Ollama),
  non trovando la risposta, restituisca legittimamente valori nulli ('null' o 0).
================================================================================
"""

# Importiamo BaseModel: è la classe madre di Pydantic. Tutte le nostre "gabbie"
# dovranno ereditare da questa classe per ottenere automaticamente i poteri di validazione.
# Importiamo Field: serve per aggiungere metadati (come le descrizioni) ai nostri campi.
# Queste descrizioni verranno lette dall'LLM per capire cosa deve scrivere!
from pydantic import BaseModel, Field

# Importiamo List e Optional dal modulo typing nativo di Python.
# List ci serve per dire a Pydantic che un determinato campo è un array (una lista) di stringhe.
# Optional ci serve per indicare che un campo può anche essere 'None' (nullo),
# salvando il sistema dai crash se l'LLM restituisce 'null'.
from typing import List, Optional


# Creiamo la nostra classe "contratto". Ereditando da BaseModel, la trasformiamo
# in un validatore inflessibile (ma ora tollerante agli edge-cases).
class AnalisiLetteraria(BaseModel):
    """
    Il contratto rigoroso che l'LLM deve rispettare quando risponde
    a una domanda basandosi sui frammenti storici estratti dal Retriever.
    """

    # ==========================================================================
    # CAMPO 1: IL DISINNESCO DELLE ALLUCINAZIONI (Tipo: Booleano)
    # ==========================================================================
    # Definiamo 'risposta_trovata' come un 'bool' (solo True o False).
    # Field(description=...) è puro Prompt Engineering nascosto nel codice.
    # L'LLM leggerà questa descrizione e capirà che, prima di generare qualsiasi
    # testo, deve onestamente valutare se il frammento fornito da Qdrant
    # contiene davvero la risposta alla domanda dell'utente.
    risposta_trovata: bool = Field(
        description="True se il testo fornito contiene le informazioni per rispondere alla domanda, False se il testo non c'entra nulla."
    )

    # ==========================================================================
    # CAMPO 2: LA RISPOSTA GENERATIVA (Tipo: Stringa o Nulla)
    # ==========================================================================
    # Questo è il campo dove l'IA scriverà la sua risposta discorsiva (la 'G' del RAG).
    # La descrizione forza l'IA a un comportamento di fallback sicuro.
    # Usiamo Optional[str] e default="". In questo modo, se l'LLM aziendale
    # restituisce 'null' (perché non ha trovato la risposta), Pydantic accetta
    # il dato pacificamente e inserisce una stringa vuota al suo posto.
    sintesi_storica: Optional[str] = Field(
        default="",
        description="La risposta discorsiva alla domanda dell'utente. Se risposta_trovata è False, scrivi 'Non ci sono informazioni sufficienti nei documenti per rispondere'."
    )

    # ==========================================================================
    # CAMPO 3: ESTRAZIONE DATI STRUTTURATI (Tipo: Lista di Stringhe)
    # ==========================================================================
    # Chiediamo all'IA di estrarre dei concetti chiave, ma imponiamo un tipo 'List[str]'.
    # Questo è fondamentale per l'interfaccia utente o per le analisi successive:
    # avremo sempre la certezza di ricevere un array iterabile.
    # Aggiungiamo default_factory=list per evitare che un JSON con
    # "concetti_chiave": null provochi un'eccezione bloccante.
    concetti_chiave: List[str] = Field(
        default_factory=list,
        description="Una lista di massimo 3 concetti o parole chiave fondamentali estratti dal testo."
    )

    # ==========================================================================
    # CAMPO 4: VERIFICA MATEMATICA E LOGICA (Tipo: Decimale o Nullo)
    # ==========================================================================
    # Definiamo un punteggio di affidabilità come numero decimale (float).
    # Abbiamo modificato il tipo da int a float perché modelli LLM come Gemma
    # a volte restituiscono voti decimali (es. 9.8 o 9.5).
    # Usiamo i parametri 'ge' (Greater or Equal - >= 0.0) e 'le' (Less or Equal - <= 10.0).
    # Se l'LLM non trova la risposta e decide di auto-valutarsi con affidabilità 0
    # oppure omette il campo (restituendo null), il sistema non andrà in crash grazie all'Optional.
    affidabilita_risposta: Optional[float] = Field(
        default=0.0,
        description="Un punteggio da 1 a 10 che indica quanto la risposta è direttamente supportata dal testo (es. 9.5). Se risposta_trovata è False, restituisci 0.",
        ge=0.0,
        le=10.0
    )