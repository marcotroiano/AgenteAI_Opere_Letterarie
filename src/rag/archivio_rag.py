# ==============================================================================
# File: src/rag/archivio_rag.py
# (SPOSTATO DALLA ROOT ALLA CARTELLA 'src/rag/' PER ARCHITETTURA MODULARE)
# ==============================================================================

"""
    ================================================================================
    DOCUMENTAZIONE ARCHITETTURALE: CLASSE ArchivioRAG
    ================================================================================

    RIFERIMENTI AL MANUALE:
    - Modulo 1: Duck Typing (Introspezione) ed Ereditarietà.
    - Modulo 3: LlamaIndex (Gestione di 'Document' base).
    - Modulo 4: Ingestion, Embeddings e Connessione Python-Qdrant.
    - Modulo 7: Ricerca Ibrida Nativa (Sparse Vectors e Database su SSD).

    1. RUOLO E COMPITI TECNICI (COSA FA)
    Questa classe funge da "ponte" tra il mondo della programmazione classica
    (oggetti OOP) e l'ecosistema dell'Intelligenza Artificiale. Svolge 3 compiti:

    - Inizializzazione: Si connette al server aziendale OLLAMA remoto per l'elaborazione
      pesante (LLM ed Embedding) e accende il database vettoriale Qdrant in LOCALE su SSD
      per garantire la persistenza dei dati senza doverli ricaricare ad ogni avvio.
    - Gatekeeper (Buttafuori): Riceve i documenti in ingresso e li analizza per
      assicurarsi che rispettino i requisiti. Respinge i file difettosi.
    - Traduzione e Archiviazione: Estrae i testi dai documenti approvati, li
      impacchetta per LlamaIndex, delega la traduzione in vettori densi al server
      Ollama e la traduzione in vettori sparsi (BM25) a FastEmbed, per poi
      salvarli definitivamente nell'indice ibrido Qdrant su SSD.

    2. SCELTE INGEGNERISTICHE E PATTERN (PERCHÉ È SCRITTA COSÌ)
    - INCAPSULAMENTO E DATA HIDING: La lista '__documenti_da_processare' e il
      database '__client' sono strettamente privati (doppio underscore). Questo
      impedisce a moduli esterni di manomettere il DB o eludere i controlli.

    - INTROSPEZIONE E DUCK TYPING: Il metodo '__valida_contratto' non usa la funzione
      'isinstance' (che creerebbe un vincolo rigido alle classi), ma sfrutta 'hasattr'
      e 'callable'. Verifica dinamicamente a runtime se l'oggetto possiede le abilità
      richieste. Questo garantisce massima scalabilità: l'archivio potrà accettare
      future classi (es. 'AudioTrascritto') senza modificare una riga di questo codice.

    - ARCHITETTURA IBRIDA NATIVA (FULL-ASYNC + QDRANT HYBRID):
      In accordo con le best practice di ingegneria dei dati, abbiamo implementato
      la ricerca ibrida direttamente a livello di database. Abilitando i "Vettori Sparsi"
      (enable_hybrid=True), l'architettura azzera i consumi di memoria RAM che
      sarebbero richiesti da un BM25 manuale.

    - GESTIONE DEL THREADING E DEL FILE LOCK:
      Qdrant in versione locale (su disco) usa 'portalocker' per impedire che due
      processi scrivano contemporaneamente corrompendo il database. Per evitare
      l'errore 'AlreadyLocked', usiamo UN SOLO client sincrono. Sarà poi LlamaIndex
      ad avvolgerlo automaticamente in un Thread asincrono durante la chat.

    - PRINCIPIO 'FAIL-FAST': La validazione avviene PRIMA dell'inserimento in memoria.
      In caso di anomalia, il sistema "fallisce velocemente" bloccando l'esecuzione
      e sollevando un'eccezione. Previene l'inquinamento del database vettoriale.
    ================================================================================
"""

# DIARIO DI BORDO / SPIEGAZIONE DELLE MODIFICHE DELL'ARCHITETTURA:
# --------------------------------------------------------------------------------
# Fase 1 (Architettura Tradizionale):
# La classe ArchivioRAG nasceva come un semplice "raccoglitore". Prendeva i documenti,
# li validava tramite Duck Typing, li metteva in una lista protetta e infine
# incollava tutti i testi in un'unica stringa (metodo 'genera_report_vettoriale').
#
# Fase 2 (Evoluzione in Motore di Intelligenza Artificiale - RAG):
# Abbiamo trasformato il raccoglitore in un vero "motore vettoriale".
# 1. All'avvio (nel costruttore __init__), accende un database vettoriale in RAM (Qdrant).
# 2. Carica un "cervello matematico" locale (HuggingFace) per tradurre le parole in numeri.
# 3. Il vecchio metodo di concatenazione è stato sostituito da 'genera_vettori_qdrant()'.
#    Questo metodo sfrutta ancora il Polimorfismo per leggere i documenti, ma ora
#    li impacchetta per l'IA (LlamaIndex), li trasforma in coordinate matematiche
#    (embedding) e li salva definitivamente nel database.
#
# Fase 3 (Architettura Enterprise Ibrida - Attuale):
# 1. Il "cervello matematico" non è più locale (HuggingFace), ma remoto (Ollama su VPN).
# 2. Qdrant non gira più nella RAM provvisoria, ma salva i vettori direttamente sul
#    disco SSD del computer, permettendo riavvii senza perdere l'indicizzazione.
# 3. Parametrizzazione tramite file .env: letture dinamiche di OLLAMA_URL, OLLAMA_LLM_MODEL
#    (risolvendo il 404 del tag non trovato) e QDRANT_STORAGE_PATH.
# 4. Implementazione del Caching su Disco: Il sistema controlla se i vettori esistono
#    già prima di chiamare il server, abbattendo i tempi di caricamento da minuti a decimi di secondo.
# --------------------------------------------------------------------------------

import os
import re # IMPORTANTE: Aggiunto per le Espressioni Regolari (Spacchettamento Metadati)
from dotenv import load_dotenv

# ==============================================================================
# MODIFICA ARCHITETTURALE: AGGIORNAMENTO PATH DEGLI IMPORT
# Il file documento_base.py ora risiede nel package 'src.models'.
# ==============================================================================
# Importiamo il "contratto" (la classe astratta).
# NOTA: Con il Duck Typing non ci serve più per bloccare gli oggetti, ma è
# comunque buona pratica averlo nel progetto per definire l'architettura.
from src.models.documento_base import DocumentoBase

# ==============================================================================
# NUOVI IMPORT PER L'INTEGRAZIONE AI (LLAMAINDEX, QDRANT E OLLAMA)
# ==============================================================================
# Qdrant è il nostro "Database Vettoriale". Invece di tabelle, salva coordinate.
import qdrant_client
# ABBIAMO RIMOSSO L'AsyncQdrantClient per evitare conflitti sul disco fisso locale!

# Componenti core di LlamaIndex:
# - Document: Il formato standard in cui impacchettare testi e metadati per l'IA.
# - VectorStoreIndex: Il motore che converte i Document in vettori.
# - Settings: Il pannello di controllo globale (es. per impostare i modelli offline).
from llama_index.core import Document, VectorStoreIndex, Settings

# Connettori di storage:
# Dicono a LlamaIndex di non salvare i vettori in memoria provvisoria, ma dentro Qdrant.
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.storage.storage_context import StorageContext

# Nuovi Connettori per Ollama (Sostituiscono HuggingFace locale)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# Carichiamo immediatamente le variabili sicure dal file .env (es. l'URL di Ollama)
load_dotenv()


class ArchivioRAG:
    """
    CLASSE GATEKEEPER E MOTORE VETTORIALE (NATIVE HYBRID EDITION):
    Questa classe funge da sala d'attesa sicura. Il suo scopo è raccogliere
    documenti, verificarli, delegare la traduzione in embedding matematici al
    server Ollama e salvarli in modo permanente nel database Qdrant su SSD.
    """

    # ==========================================================================
    # 1. IL COSTRUTTORE (Incapsulamento e Avvio Motore AI)
    # ==========================================================================
    def __init__(self):
        # NOTA: Quando nasce, l'archivio è vuoto. Poiché non riceve dati dall'esterno
        # direttamente nel costruttore, NON servono validatori in questa fase!

        # Creiamo una lista STRETTAMENTE PRIVATA (doppio underscore).
        # Questo impedisce a programmatori esterni di fare: archivio.__documenti_da_processare.append(virus)
        # L'unico modo per entrare in questa lista è passare dal buttafuori (ingurgita_documento).
        self.__documenti_da_processare = []

        # --- NOVITÀ FASE 3: ACCENSIONE DEL MOTORE AI TRAMITE OLLAMA REMOTO ---
        print("Inizializzazione ArchivioRAG: Connessione al server aziendale OLLAMA...")

        # =====================================================================================
        # IL PROBLEMA LINGUISTICO DELLE MACCHINE E LA VETTORIZZAZIONE
        # Un computer non sa chi sia Dante, non sa cosa sia un "assedio" e non capisce
        # il vocabolario italiano. Per far sì che la macchina possa "capire" i concetti,
        # dobbiamo tradurre le parole in un linguaggio che sa elaborare: la matematica.
        #
        # L'EMBEDDING VIA OLLAMA SERVER
        # In precedenza questa traduzione avveniva usando la CPU locale. Ora, per scalare
        # il sistema, preleviamo l'URL del server aziendale (dove gira Ollama) dal file .env
        # e istruiamo LlamaIndex a spedire i testi lì per la vettorializzazione.
        # =====================================================================================

        # Recuperiamo l'indirizzo del server (chiave OLLAMA_URL configurata nel .env)
        ollama_url = os.getenv("OLLAMA_URL") or os.getenv("OLLAMA_BASE_URL")

        if not ollama_url:
            raise ValueError("❌ Errore: Variabile OLLAMA_URL non trovata nel file .env!")

        # 1. Impostiamo il modello di traduzione vettoriale puntando al server aziendale.
        # 'nomic-embed-text' è il modello verificato e attivo sul server aziendale.
        # embed_batch_size=16 riduce l'overhead di rete accorpando le chiamate API
        Settings.embed_model = OllamaEmbedding(
            model_name="nomic-embed-text:latest",
            base_url=ollama_url,
            embed_batch_size=16,
            request_timeout=120.0 # Se Ollama non risponde entro 120s, lancia errore invece di congelarsi
        )

        # 2. Impostiamo il cervello (LLM) che genererà le risposte, prelevandolo dinamicamente
        # dal file .env (OLLAMA_LLM_MODEL="gemma3:4b" verificato tra i modelli aziendali).
        modello_llm = os.getenv("OLLAMA_LLM_MODEL", "gemma3:4b")

        Settings.llm = Ollama(
            model=modello_llm,
            base_url=ollama_url,
            request_timeout=180.0,  # Timeout esteso per gestire l'inferenza di modelli da 8B via VPN

            # MODIFICA TUTOR: Alziamo num_predict a 1000 per dare al modello
            # abbastanza token per scrivere le risposte lunghe e dettagliate richieste.
            # Abbassiamo leggermente la temperature a 0.1 per mantenerlo ancorato ai fatti.
            additional_kwargs={"options": {"num_predict": 1000, "temperature": 0.1}}
        )

        print("Inizializzazione ArchivioRAG: Avvio database Qdrant su Disco (SSD) in modalità HYBRID...")

        # ======================================================================
        # MODIFICA STORAGE E PERCORSO RELATIVO:
        # Poiché il programma viene eseguito dal file 'main.py' che si trova nella
        # root del progetto, "./mio_database_vettoriale" verrà creato correttamente
        # nella cartella principale (dove sarà poi ignorato da Git grazie al .gitignore),
        # e NON dentro questa cartella src/rag. L'architettura è salva!
        # ======================================================================
        qdrant_path = os.getenv("QDRANT_STORAGE_PATH", "./mio_database_vettoriale")

        self.__client = qdrant_client.QdrantClient(path=qdrant_path)

        # Salvo il nome della collection in una variabile di classe per usarla nel metodo successivo
        self.nome_cassetto = "archivio_storico_tesi"

        # ======================================================================
        # MAGIA ARCHITETTURALE: ABILITAZIONE DELLA RICERCA IBRIDA NATIVA
        # Impostando 'enable_hybrid=True', ordiniamo a LlamaIndex di generare sia
        # i Vettori Densi (Ollama) sia i Vettori Sparsi (FastEmbed), archiviando
        # l'equivalente di un BM25 direttamente su Qdrant SSD in locale.
        # Passiamo solo il client sincrono!
        # ======================================================================
        self.__vector_store = QdrantVectorStore(
            client=self.__client,
            collection_name=self.nome_cassetto,
            enable_hybrid=True
        )
        self.__storage_context = StorageContext.from_defaults(vector_store=self.__vector_store)

        # Variabile pubblica che, alla fine del processo, ospiterà l'indice ricercabile.
        self.indice_rag = None

    # ==========================================================================
    # 2. IL VALIDATORE PRIVATO DI TIPO (Duck Typing / Introspezione)
    # ==========================================================================
    def __valida_contratto(self, documento):

        """
        ================================================================================
        L'EVOLUZIONE VERSO IL DUCK TYPING (Introspezione)
        ================================================================================
        Inizialmente, usavamo 'isinstance(documento, DocumentoBase)'. Quello era un
        approccio tipico dei linguaggi statici (Java, C++), che controlla il pedigree
        della classe (Ereditarietà). Ma bloccava gli "intrusi" come l'ArticoloGiornale!

        Ora usiamo l'Introspezione (hasattr e callable). Il buttafuori non guarda
        più la carta d'identità, ma fa un "provino pratico":
        "Sai eseguire mostra_metadati()? Sai eseguire leggi_tutto()?".
        Se l'oggetto sa fare queste due cose (fa quack come un'anatra), per
        l'archivio è autorizzato a entrare, indipendentemente dalla sua famiglia.

        Spiegazione pratica: Qui c'è l'introspezione. Il codice non chiede if type(documento) == LibroStorico.
        Chiede solo: "L'oggetto che mi hai passato possiede il metodo leggi_tutto?". Se sì, passa.
        È un pattern avanzato che rende il tuo sistema pronto per il futuro.

        ================================================================================
        """
        # Verifichiamo se l'oggetto possiede i metodi richiesti e se sono "eseguibili" (callable)
        ha_metadati = hasattr(documento, 'mostra_metadati') and callable(getattr(documento, 'mostra_metadati'))
        ha_testo = hasattr(documento, 'leggi_tutto') and callable(getattr(documento, 'leggi_tutto'))

        # Se manca anche solo una delle due abilità, lo respingiamo sollevando un'eccezione!
        if not (ha_metadati and ha_testo):
            raise TypeError("❌ Errore di sicurezza: L'oggetto non possiede i metodi richiesti (Duck Typing fallito)!")

        # Se i controlli sono passati, restituiamo l'oggetto validato
        return documento

    # ==========================================================================
    # 3. METODO DI INGESTIONE SICURA (Fail-Fast e Type Checking)
    # ==========================================================================
    def ingurgita_documento(self, documento_in_ingresso):
        # 1. Passiamo l'oggetto al validatore privato (Fail-Fast).
        # Se non va bene, si blocca subito restituendo un errore.
        documento_sicuro = self.__valida_contratto(documento_in_ingresso)

        # 2. Se arriva a questa riga, significa che non ci sono stati errori.
        # Modifichiamo in sicurezza la lista privata.
        self.__documenti_da_processare.append(documento_sicuro)
        print(
            f"✅ Successo: Documento autorizzato e inserito nell'archivio (Totale: {len(self.__documenti_da_processare)}).")

    # ==========================================================================
    # 4. TRASFORMAZIONE E SALVATAGGIO VETTORIALE (Sfruttamento dell'Astrazione)
    # ==========================================================================
    async def genera_vettori_qdrant(self):
        """
        NOVITÀ FASE 3: Verifica l'esistenza del database su SSD per evitare di
        ricalcolare i vettori. Se la collection è vuota o inesistente, delega il
        calcolo vettoriale al server Ollama e salva i vettori di ritorno nel database Qdrant.
        """

        # ======================================================================
        # CONTROLLO PERSISTENZA
        # ======================================================================
        ha_dati_persistenti = False
        try:
            # FIX: Chiamata SINCRONA al DB
            info_db = self.__client.get_collection(collection_name=self.nome_cassetto)

            # Se ci sono vettori ("punti") salvati, non serve ricalcolarli
            if info_db.points_count > 0:
                ha_dati_persistenti = True
                print(
                    f"⚡ [CACHE HIT]: Rilevati {info_db.points_count} vettori Ibridi (Dense+Sparse) persistenti su SSD!")
        except Exception:
            # Se la collection non esiste (primo avvio), Qdrant lancia un'eccezione
            ha_dati_persistenti = False

        if ha_dati_persistenti:
            print("⚡ Caricamento istantaneo dell'indice Ibrido da SSD (Nessun calcolo remoto richiesto)...")

            # Invece di 'from_documents', usiamo 'from_vector_store' per bypassare l'embedding
            # Manteniamo 'use_async=True' per garantire l'accesso parallelo e asincrono durante l'uso in chat.
            self.indice_rag = VectorStoreIndex.from_vector_store(
                vector_store=self.__vector_store,
                storage_context=self.__storage_context,
                use_async=True
            )
            print("✅ Indice Ibrido caricato con successo in frazioni di secondo!")
            return

        # ======================================================================
        # INGESTION E CALCOLO VETTORIALE (Se il database è vuoto o mancano i Vettori Sparsi)
        # ======================================================================
        # Inizializziamo un array vuoto per contenere i documenti nel formato LlamaIndex
        documenti_llama = []

        # Il ciclo itera IN SICUREZZA sulla lista privata dell'oggetto stesso
        for documento in self.__documenti_da_processare:
            # ------------------------------------------------------------------
            # Qui godiamo dei frutti dell'Astrazione e del Polimorfismo.
            # Non ci serve scrivere: "Se è un libro fai X, se è un articolo fai Y".
            # Sappiamo con certezza matematica (grazie al validatore) che qualsiasi
            # oggetto sia arrivato qui dentro possiede i metodi 'mostra_metadati()'
            # e 'leggi_tutto()'. Li chiamiamo "alla cieca".
            # ------------------------------------------------------------------

            info_metadati = documento.mostra_metadati()
            testo_puro = documento.leggi_tutto()

            # LlamaIndex richiede che i metadati siano forniti come Dizionario Python.
            # Manteniamo la stringa originale intera per retrocompatibilità
            metadati_dict = {"informazioni_storiche": info_metadati}

            # ==============================================================
            # METADATA EXTRACTION (Filtraggio Avanzato per Qdrant)
            # Estraiamo dinamicamente Anno, Titolo e Autore dalla stringa
            # usando le Espressioni Regolari (Regex). Questi diventano i
            # campi matematici (Payload) che Qdrant usa per i filtri rigidi.
            # ==============================================================
            match = re.search(r"\[(\d+)\]\s+(.*?)\s+-\s+di\s+(.*?)\s+\(Lingua:\s+(.*?)\)", info_metadati)
            if match:
                metadati_dict["anno"] = int(match.group(1))  # Trasformato in VERO NUMERO per Qdrant
                metadati_dict["titolo"] = match.group(2).strip()
                metadati_dict["autore"] = match.group(3).strip()
                metadati_dict["lingua"] = match.group(4).strip()
                # Ora Qdrant costruirà autonomamente dei 'Payload Index' invisibili su queste chiavi!

            # ==============================================================
            # NOVITÀ FASE 4: FUSIONE TESTUALE METADATI FISICI (Best of Both Worlds)
            # Per evitare di inquinare il Payload Index di Qdrant con campi minori
            # come 'location', 'peso' o 'rilegatura', estraiamo questi dati dagli
            # attributi dell'oggetto Python (usando getattr in modo sicuro) e li
            # iniettiamo direttamente ALL'INIZIO DEL TESTO.
            # In questo modo:
            # 1. Il BM25 indicizzerà le parole "Sala C" e "Scaffale".
            # 2. L'LLM leggerà questa stringa discorsiva e saprà rispondere alla perfezione.
            # (NOTA: Affinché getattr funzioni, assicurati che la classe che legge il
            # JSON salvi effettivamente queste chiavi come attributi dell'oggetto,
            # ad es. self.location = dati_json.get('location')).
            # ==============================================================
            location = getattr(documento, 'location', 'Posizione ignota')
            binding = getattr(documento, 'binding', 'N/D')
            weight = getattr(documento, 'weight', 'N/D')
            lines = getattr(documento, 'lines', 'N/D')
            requests = getattr(documento, 'requests', 'N/D')

            # Creiamo il blocco discorsivo che fa da "prefazione" al frammento
            info_biblioteca = (
                f"[DETTAGLI FISICI E ARCHIVISTICI DEL DOCUMENTO]\n"
                f"- Posizione in Biblioteca: {location}\n"
                f"- Rilegatura: {binding}\n"
                f"- Peso: {weight}\n"
                f"- Impaginazione: {lines}\n"
                f"- Consultazione: {requests}\n"
                f"----------------------------------------------\n"
            )

            # Fondiamo i metadati fisici con il testo dell'opera per l'LLM
            testo_arricchito = f"{info_biblioteca}\n[TESTO OPERA]\n{testo_puro}"

            # Impacchettiamo il tutto nell'oggetto 'Document' standard di LlamaIndex
            # NOTA: Passiamo il 'testo_arricchito' invece del 'testo_puro'
            doc_llama = Document(text=testo_arricchito, metadata=metadati_dict)

            # Aggiungiamo alla lista da inviare al motore vettoriale
            documenti_llama.append(doc_llama)

        print(
            f"\nRichiesta di calcolo vettoriale Ibrido per {len(documenti_llama)} documenti (Semantica su Ollama + Lessicale Locale su FastEmbed)...")

        # MOTORE IN AZIONE: VectorStoreIndex prende i documenti impacchettati,
        # chiama il server remoto (Ollama) per farsi dare i numeri vettoriali,
        # usa FastEmbed per creare l'indice BM25 locale e infine inietta TUTTO
        # dentro il database Qdrant (sul disco SSD) tramite lo storage_context.
        self.indice_rag = VectorStoreIndex.from_documents(
            documenti_llama,
            storage_context=self.__storage_context,
            show_progress=True,  # Mostra una barra di avanzamento a terminale
            use_async=False, # FIX ARCHITETTURALE: Lavora in sincrono per non bloccare la VPN!
            insert_batch_size = 512  # <-- NUOVA RIGA: Impone a LlamaIndex di fare pause più frequenti
        )

        print("✅ Vettori Ibridi generati e salvati definitivamente in Qdrant (SSD locale) con successo!")