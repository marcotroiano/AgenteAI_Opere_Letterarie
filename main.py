# ==============================================================================
# File: main.py - LA PLANCIA DI COMANDO (ENTRY POINT) E MANIFESTO
# ==============================================================================

"""
    ================================================================================
    MANIFESTO ARCHITETTONICO E TEORICO (NOTE PER LA TESI)
    ================================================================================
    DIARIO DI BORDO / DOCUMENTAZIONE ARCHITETTURALE:
    Questo file rappresenta l'"Entry Point" (il punto di ingresso) dell'applicazione.
    In un'architettura software ben progettata, il main.py non deve contenere logica
    complessa, ma deve limitarsi a fare da "direttore d'orchestra" tra i vari moduli.

    L'intera architettura del progetto poggia su tre grandi pilastri teorici:

    PILASTRO 1: INGEGNERIA DEL SOFTWARE (OOP E MODULARITÀ)
    1. Entry Point Pattern: Il main.py siede fuori dal codice sorgente ('src/'). È l'unico
       file ad avere visione d'insieme, importando e assemblando i pezzi del puzzle.
    2. Incapsulamento: In questo file NON c'è alcun import diretto alle librerie di IA
       (come llama_index o qdrant). Il main non sa come funzionano i vettori, sa solo
       che chiamando 'mio_archivio.genera_vettori_qdrant()' il lavoro verrà fatto.
    3. Composizione (Has-a): I libri "fabbricano" e possiedono i Frammenti.
    4. Separation of Concerns: Il parser legge, l'archivio valida, il motore cerca,
       l'agente genera e Pydantic struttura. Ogni file ha una singola responsabilità.

    PILASTRO 2: INTELLIGENZA ARTIFICIALE (PARADIGMA RAG ON-PREMISE HYBRID)
    1. Architettura RAG Distribuita e Ibrida: Questo sistema copre la fase di Retrieval
       tramite la funzionalità "Hybrid Search" nativa di Qdrant su SSD (che gestisce
       internamente sia vettori semantici che Keyword Sparse Vectors senza appesantire la RAM).
    2. Elaborazione Asincrona Continua: L'agente comunica con il modello generativo
       remoto (sulla VPN) tramite un Event Loop persistente, liberando il thread
       principale del client ed evitando la chiusura anomala dei socket di rete.

    PILASTRO 3: METADATA FILTERING E CONTEXT ENRICHMENT (NOVITÀ FASE 4)
    1. Filtraggio Rigido: I campi deterministici (Anno, Autore, Titolo) sono trattati come
       Payload matematici per restringere il campo di ricerca ed evitare allucinazioni.
    2. Fusione del Contesto Fisico: I dettagli biblioteconomici (Location, Scaffale, Rilegatura)
       sono iniettati discorsivamente nel testo, permettendo all'LLM di comprendere e
       guidare l'utente fisicamente all'interno dell'archivio.
    ================================================================================
"""

# ==============================================================================
# IMPORTAZIONI DI SISTEMA E DI MODULI
# ==============================================================================

# Importiamo asyncio: è la libreria fondamentale che ci permette di fare da "ponte"
# tra questo file e l'agente asincrono, permettendo di gestire le attese di rete
# in modo cooperativo.
import asyncio

# Aggiungiamo textwrap per l'incolonnamento estetico delle risposte lunghe a terminale
import textwrap

# ==============================================================================
# IMPORT DEI MODULI CUSTOM (Architettura basata su 'src')
# Qui si vede la pulizia della nuova architettura. Il main.py importa in modo
# esplicito le singole funzionalità attingendo ai pacchetti (cartelle) specializzati.
# ==============================================================================
from src.rag.archivio_rag import ArchivioRAG
from src.rag.parser_dataset import carica_dataset_json
from src.agent.agente_asincrono import esegui_agente_rag


# NOTA ARCHITETTURALE: Trasformiamo l'intera pipeline in una Coroutina (async def).
# Questo ci permette di avviare l'infrastruttura di rete una sola volta all'avvio.
async def esegui_pipeline() -> None:
    """
    Esegue l'intera pipeline aziendale in ambiente asincrono nativo:
    Ingestion locale, Embedding remoto (Ollama), Retrieval Ibrido locale (Qdrant)
    e Generation remota continua senza distruzione dell'Event Loop.
    """

    # --------------------------------------------------------------------------
    # FASE 1: AVVIO INFRASTRUTTURA
    # Il main delega ad ArchivioRAG l'accensione del database Qdrant (su disco fisso)
    # e il setup del client per il server LLM aziendale (Ollama).
    # Crea l'oggetto mio_archivio di classe ArchivioRAG.
    # --------------------------------------------------------------------------
    print(" FASE 1: Avvio del sistema RAG On-Premise (Modalità Ibrida Nativa)...")
    mio_archivio = ArchivioRAG()

    # --------------------------------------------------------------------------
    # FASE 2: DATA INGESTION (Estrazione Dati)
    # Il main chiama il "Data Engineer" (parser_dataset) per leggere i file JSON
    # dal disco rigido e trasformarli in veri e propri oggetti OOP (LibroStorico).
    # --------------------------------------------------------------------------
    print("\n FASE 2: Lettura dinamica del dataset e creazione degli oggetti (OOP)...")

    # MODIFICA ARCHITETTURALE: I PERCORSI RELATIVI
    # Poiché 'main.py' viene eseguito dalla radice del progetto, il percorso "./dataset"
    # funzionerà perfettamente, cercando la cartella al suo stesso livello. Lo stesso vale
    # per la creazione di "./mio_database_vettoriale" fatta in archivio_rag.py.
    cartella_dataset = "./dataset"
    lista_libri_reali = carica_dataset_json(cartella_dataset)
    print(f" Trovati e costruiti {len(lista_libri_reali)} libri letterari dal dataset.")

    # --------------------------------------------------------------------------
    # FASE 3: IL GATEKEEPER (Validazione)
    # Cicliamo tutti gli oggetti creati e li facciamo esaminare dal buttafuori.
    # Grazie al Duck Typing, l'archivio accetterà solo documenti sicuri.
    # --------------------------------------------------------------------------
    print("\n FASE 3: Ingestione e Validazione (Il Gatekeeper entra in azione)...")
    for libro in lista_libri_reali:
        mio_archivio.ingurgita_documento(libro)
    print(" Tutti i documenti hanno superato i controlli del Gatekeeper.")

    # --------------------------------------------------------------------------
    # FASE 4: EMBEDDING (Traduzione in Vettori Ibridi)
    # Deleghiamo al server aziendale il calcolo matematico. I testi vengono convertiti
    # in coordinate e salvati permanentemente nel Qdrant locale.
    # --------------------------------------------------------------------------
    print("\n FASE 4: Generazione Vettoriale (Delegata al server Ollama e Qdrant)...")

    # ==========================================================================
    # MODIFICA ARCHITETTURALE (ASINCRONIA PURA):
    # La chiamata è ora ASINCRONA (await). Attendiamo nativamente la risposta del
    # DB vettoriale senza interrompere l'Event Loop.
    # ==========================================================================
    await mio_archivio.genera_vettori_qdrant()

    # --------------------------------------------------------------------------
    # FASE 5: CHECK DI SICUREZZA
    # Un classico approccio "Fail-Fast". Se per qualche motivo Qdrant è fallito
    # e l'indice è None, fermiamo tutto (return) per evitare che la chat esploda.
    # --------------------------------------------------------------------------
    print("\n FASE 5: Verifica finale...")
    if mio_archivio.indice_rag is not None:
        print(" SUCCESSO TOTALE: L'indice vettoriale Ibrido è attivo e pronto per essere interrogato!")
    else:
        print(" ERRORE: L'indice vettoriale non è stato generato.")
        return

    # --------------------------------------------------------------------------
    # FASE 6: CHAT INTERATTIVA (Retrieval + LLM Generation)
    # Qui entriamo nel vivo del Modulo 6. Inizia l'interazione uomo-macchina.
    # --------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(" FASE 6: L'Agente Asincrono entra in azione (Interazione Utente)")
    print("=" * 60)

    # Questo ciclo 'while True' mantiene l'infrastruttura attiva all'interno
    # dello stesso Event Loop asincrono. Ciò garantisce che i socket di rete
    # non vengano chiusi ("Event loop is closed") tra una domanda e l'altra.
    while True:

        # 1. Input dell'utente: Mettiamo il programma in pausa aspettando che
        #    l'utente digiti qualcosa sulla tastiera. L'input in Python è bloccante,
        #    ma va bene perché aspettiamo l'azione umana.
        #
        # MODIFICA: Aggiorniamo il prompt per far capire all'utente che il sistema
        # ora conosce anche le posizioni fisiche e i dettagli biblioteconomici.
        domanda = input(
            "\nFai una domanda letteraria (o digita 'esci' per terminare):\n> ")

        # 2. Graceful Exit (Uscita Elegante): Controlliamo se l'utente vuole uscire.
        #    Usiamo .lower().strip() per catturare anche " ESCI " o "Esci".
        if domanda.lower().strip() in ['esci', 'exit', 'quit']:
            print("Chiusura del sistema. Arrivederci!")
            break  # Interrompe il ciclo while e fa terminare il programma

        # 3. L'INVOCAZIONE ASINCRONA CONTINUA (Fix per "Event loop is closed")
        # Questa è la riga cruciale modificata. Invece di usare `asyncio.run()`
        # (che aprirebbe e chiuderebbe brutalmente il loop di rete ad ogni ciclo),
        # usiamo `await`. Questo dice al sistema: "Attendi pacificamente la
        # risposta del server Ollama senza distruggere la connessione".

        # ==============================================================================
        # CHIAMATA ALL'AGENTE RAG ASINCRONO
        # 1. 'mio_archivio.indice_rag': Estraiamo l'attributo dal Qdrant vivo.
        # 2. Passiamo questo indice e la domanda all'Agente (esegui_agente_rag).
        # 3. L'Agente invierà l'indice al Motore di Ricerca per estrarre i testi.
        # 4. 'await': Essendo l'Agente una coroutina (async def), lo eseguiamo nativamente.
        # ==============================================================================
        report_finale = await esegui_agente_rag(mio_archivio.indice_rag, domanda)

        # 4. RENDERIZZAZIONE DELL'OUTPUT STRUTTURATO
        # Se l'agente ha restituito qualcosa (e non un None per errore di rete)
        if report_finale:
            print("\n" + "★" * 70)
            print("📄 RISPOSTA STRUTTURATA DELL'AGENTE (JSON Validato da Pydantic)")
            print("★" * 70)

            # Qui godiamo dei frutti dello "Structured Output" (Pydantic).
            # Invece di dover fare il parsing di una stringa grezza rischiando crash,
            # accediamo ai dati in modo sicuro e tipizzato (Type Hinting).
            # Usiamo un if-inline per trasformare il booleano (True/False) in ✅ o ❌.
            print(f"La risposta era nei testi?: {'Sì ✅' if report_finale.risposta_trovata else 'No ❌'}")

            # Leggiamo il campo numerico (l'intero da 1 a 10)
            print(f"Affidabilità RAG: {report_finale.affidabilita_risposta}/10")

            # report_finale.concetti_chiave è garantito essere una Lista di stringhe.
            # Usiamo ', '.join(...) per trasformare la lista in una singola frase.
            print(f"Concetti Chiave estratti: {', '.join(report_finale.concetti_chiave)}")

            print("-" * 70)

            # ==================================================================
            # INCOLONNAMENTO DEL TESTO (Text Wrapping)
            # Siccome abbiamo forzato il modello generativo a produrre risposte
            # lunghe e corpose (tramite top_k=7 e tuning del prompt), usiamo textwrap
            # per mandare a capo automaticamente le frasi. Questo garantisce
            # un'eccellente leggibilità a terminale senza fastidiosi scroll orizzontali.
            # ==================================================================
            print("Sintesi Storica:\n")

            # wrap() restituisce una lista di righe spezzate alla lunghezza desiderata (es. 75 caratteri).
            # Gestiamo separatamente gli "a capo" (\n) originali generati dal modello.
            if report_finale.sintesi_storica:
                paragrafi = report_finale.sintesi_storica.split('\n')
                for paragrafo in paragrafi:
                    if paragrafo.strip():  # Se il paragrafo non è vuoto
                        righe_incolonnate = textwrap.wrap(paragrafo, width=75)
                        for riga in righe_incolonnate:
                            print(riga)
                    else:
                        print()  # Stampa una riga vuota per mantenere la spaziatura tra paragrafi
            else:
                print("Nessuna sintesi disponibile.")

            print("\n" + "★" * 70 + "\n")


# ==============================================================================
# BLOCCO DI ESECUZIONE SCRIPT (DUNDER MAIN)
# ==============================================================================
# Variabile '__name__' assume il valore "__main__" SOLO se l'utente esegue
# direttamente questo file.
if __name__ == "__main__":
    # Avviamo un unico Event Loop globale che racchiuderà tutta la sessione interattiva,
    # dalla Fase 1 (creazione database) fino alla fine del ciclo 'while True'.
    asyncio.run(esegui_pipeline())