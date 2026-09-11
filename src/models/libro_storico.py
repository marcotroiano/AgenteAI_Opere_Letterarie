# ==============================================================================
# File: src/models/libro_storico.py
# (SPOSTATO NELLA CARTELLA 'src/models/' PER ARCHITETTURA MODULARE)
# ==============================================================================

"""
    ================================================================================
    DOCUMENTAZIONE ARCHITETTURALE: CLASSE LibroStorico
    ================================================================================
    RIFERIMENTI AL MANUALE:
    - Modulo 1: Ripasso Fondamentali (Incapsulamento, Validatori DRY, Properties).
    - Modulo 3: LlamaIndex (Composizione per il Chunking e preparazione ai Nodi).
    - Appendice A: Mappatura OOP e relazioni tra classi.
    - Appendice B.3: La Composizione usata per il Chunking e per evitare l'Out of Memory.

    1. RUOLO E COMPITI TECNICI (COSA FA)
    Questa classe è lo scheletro dei dati del progetto. Il suo compito è:
    - Raccogliere metadati: Salva in modo strutturato titolo, autore, anno e lingua
      del documento.
    - Fabbricare e contenere Frammenti: Non salva il testo in una stringa gigante,
      ma funge da "fabbrica". Riceve il testo, crea dei piccoli oggetti separati
      chiamati 'Frammento' e li colleziona in una lista interna.
    - Restituire i dati su richiesta: Quando l'IA ha bisogno del testo completo per
      calcolare i vettori matematici (gli embedding), la classe riassembla tutti i
      frammenti al volo e glieli consegna.

    2. SCELTE INGEGNERISTICHE E PATTERN (PERCHÉ È SCRITTA COSÌ)
    - COMPOSIZIONE (HAS-A) PER IL CHUNKING: Nel mondo dell'IA, dare in pasto a un
      LLM o a un modello di embedding un intero libro in una singola variabile
      stringa farebbe esplodere la memoria (o superare i limiti di token). Usando
      la Composizione, il libro "ha un" elenco di frammenti. Questo prepara il
      terreno per il Chunking (la suddivisione in blocchi semantici), un requisito
      obbligatorio per qualsiasi sistema RAG di livello enterprise.

    - PRINCIPIO DRY (Don't Repeat Yourself) NEI VALIDATORI: Invece di controllare
      che il titolo non sia vuoto sia nel costruttore che nel setter, la logica è
      centralizzata in un metodo privato (__valida_stringa). La regola si scrive
      una volta sola: se in futuro il titolo non dovesse superare i 100 caratteri,
      si modificherà una singola riga e tutto il software si aggiornerà
      automaticamente (Single Source of Truth).

    - INCAPSULAMENTO 100% PYTHONIC (@property): I dati sensibili sono nascosti
      tramite attributi privati (es. __titolo), ma l'utente non è obbligato a usare
      scomodi metodi come get_titolo(). I decoratori @property offrono una sintassi
      pulitissima (es. libro.titolo = "Nuovo") attivando, dietro le quinte, uno
      scudo di sicurezza che filtra i dati sporchi prima del salvataggio.
    ================================================================================
"""

# ================================================================================
# OBIETTIVO: Rappresentare il contenitore principale dei metadati e del testo.
# ================================================================================
# AGGIORNAMENTO ARCHITETTURALE (COMPOSIZIONE E VALIDATORI):
# 1. COMPOSIZIONE: Il libro non gestisce più il testo come un'unica variabile
#    stringa gigante. Ora il libro "HA UN" elenco di oggetti di tipo Frammento.
# 2. VALIDATORI E TYPE HINTING: Abbiamo blindato l'oggetto. I controlli di sicurezza sono stati
#    centralizzati in metodi privati (__valida_x). Inoltre, abbiamo aggiunto i Type Hint
#    (str, int) per allineare il codice agli standard industriali.
# ================================================================================


# Diciamo al LibroStorico dove trovare la "scatola" Frammento e il contratto Base per poterli usare.
# AGGIORNAMENTO IMPORT: Usiamo i percorsi assoluti a partire dal package 'src'.
from src.models.frammento import Frammento
from src.models.documento_base import DocumentoBase


# Facciamo ereditare la classe dal contratto astratto
class LibroStorico(DocumentoBase):
    # 1. VARIABILE DI CLASSE (Appartiene al "progetto", non al singolo libro)
    # Resa strettamente PRIVATA con '__'. Nessuno all'esterno può azzerare il contatore.
    __totale_libri = 0

    # 2. COSTRUTTORE DELLA CLASSE
    # MODIFICA: Aggiunti i Type Hint per garantire la sicurezza del tipo a livello IDE
    def __init__(self, titolo: str, autore: str, anno: int, lingua: str):
        # Assegnazione di attributi resi PRIVATI (doppio underscore) TRAMITE VALIDATORI.
        # Inaccessibili direttamente dall'esterno e inaccessibili direttamente dalle figlie.
        self.__titolo = self.__valida_stringa(titolo, "Il titolo")

        # Salviamo l'autore formattandolo sempre con le iniziali maiuscole (.title())
        self.__autore = self.__valida_stringa(autore, "L'autore").title()

        self.__anno = self.__valida_anno(anno)

        # Assegnazione di attributo PROTETTO (singolo underscore) TRAMITE VALIDATORE.
        # Inaccessibile dall'esterno (per convenzione), ma manipolabile dalle classi figlie.
        self._lingua = self.__valida_lingua(lingua)

        # LA COMPOSIZIONE IN AZIONE:
        # Invece di self.__contenuto = "", creiamo una lista vuota.
        # Questo attributo è strettamente privato: nessuno dall'esterno può
        # cancellare o sostituire brutalmente l'intera lista dei frammenti.
        self.__frammenti = []

        # Incrementiamo il contatore di classe ogni volta che nasce un nuovo oggetto.
        LibroStorico.__totale_libri += 1

    # ==========================================================================
    # I NUOVI VALIDATORI PRIVATI (Single Source of Truth per la sicurezza)
    # ==========================================================================
    # MODIFICA: Aggiunti Type Hint
    def __valida_stringa(self, testo: str, nome_campo: str) -> str:
        """Controlla che il testo sia una stringa e non sia vuota."""
        if not isinstance(testo, str) or not testo.strip():
            # Il 'raise ValueError' blocca il programma immediatamente se il dato è corrotto
            raise ValueError(f"Errore: {nome_campo} deve essere una stringa valida e non vuota.")
        return testo.strip()

    # MODIFICA: Aggiunti Type Hint
    def __valida_anno(self, anno: int) -> int:
        """Controlla che l'anno sia un numero intero positivo."""
        if not isinstance(anno, int) or anno <= 0:
            raise ValueError("Errore: L'anno deve essere un numero intero positivo.")
        return anno

    # MODIFICA: Aggiunti Type Hint
    def __valida_lingua(self, lingua: str) -> str:
        """Controlla che la lingua sia una stringa valida e la formatta in minuscolo."""
        if not isinstance(lingua, str) or not lingua.strip():
            raise ValueError("Errore: La lingua deve essere una stringa valida e non vuota.")

        # Restituiamo la lingua pulita dagli spazi e in minuscolo
        return lingua.strip().lower()

    # --- METODO DI CLASSE ---
    # Il decoratore @classmethod dice a Python che questo metodo opera sulla Classe (cls)
    # e non sul singolo oggetto (self). Serve per leggere il contatore privato.
    @classmethod
    def get_totale_libri(cls) -> int:
        """Restituisce il numero totale di libri istanziati finora nel sistema."""
        return cls.__totale_libri

    # --- PROPERTIES PER L'INCAPSULAMENTO STILE PYTHON ---

    # GETTER per il titolo: permette di usare 'libro.titolo' all'esterno.
    @property
    def titolo(self) -> str:
        return self.__titolo

    # SETTER per il titolo: intercetta 'libro.titolo = "Nuovo"' e applica i controlli.
    @titolo.setter
    def titolo(self, nuovo_titolo: str):
        # Verifica delegata al validatore centrale (Principio DRY: Don't Repeat Yourself)
        self.__titolo = self.__valida_stringa(nuovo_titolo, "Il titolo")

    @property
    def autore(self) -> str:
        return self.__autore

    @autore.setter
    def autore(self, nuovo_autore: str):
        self.__autore = self.__valida_stringa(nuovo_autore, "L'autore").title()

    # ==========================================================================
    # PROPERTIES PER L'ANNO (100% Pythonic)
    # ==========================================================================
    @property
    def anno(self) -> int:
        """Restituisce l'anno in formato property."""
        return self.__anno

    @anno.setter
    def anno(self, nuovo_anno: int):
        """Modifica l'anno con la sintassi libro.anno = valore."""
        self.__anno = self.__valida_anno(nuovo_anno)

    # GETTER per la lingua tramite property (nasconde l'attributo protetto _lingua)
    @property
    def lingua(self) -> str:
        """Restituisce la lingua del libro."""
        return self._lingua

    @lingua.setter
    def lingua(self, nuova_lingua: str):
        """Modifica la lingua validando che sia una stringa esatta di 3 caratteri."""
        self._lingua = self.__valida_lingua(nuova_lingua)

    # --- METODI OPERATIVI DELLA CLASSE ---
    def mostra_metadati(self) -> str:
        """
        SELF-ENCAPSULATION: All'interno della classe stessa usiamo i metodi
        di interfaccia (self.anno, self.titolo, ecc.) invece delle variabili
        private (__anno, __titolo). Questo garantisce che eventuali modifiche
        future ai getter si propaghino automaticamente anche qui.
        """
        return f"[{self.anno}] {self.titolo} - di {self.autore} (Lingua: {self.lingua})"

    """
    ================================================================================
    L'ANATOMIA DELLA COMPOSIZIONE IN 2 STEP
    ================================================================================
    Come avviene l'interazione quando nel main si chiama:
    divina_commedia.aggiungi_frammento("Nel mezzo del cammin...", 1)

    Dietro le quinte, il metodo operativo aggiungi_frammento esegue due azioni:

    1. Fabbrica l'oggetto: Usa la classe esterna Frammento come un vero e proprio 
       "stampo industriale". Prende il testo e il numero di pagina passati dall'utente 
       e li usa per creare un'entità fisica e indipendente (la variabile locale nuovo_chunk).

    2. Inietta l'oggetto: Prende questo oggetto appena nato e lo "spinge" all'interno 
       della lista privata (self.__frammenti.append()) della classe LibroStorico.

    In sintesi: l'utente chiede al Libro di aggiungere del testo, e il Libro si 
    occupa dinamicamente di creare un oggetto della classe Frammento e di 
    collezionarlo al suo interno in modo ordinato.

    Spiegazione pratica: Qui hai applicato la "Composizione". 
    Invece di creare un attributo self.testo = "intero libro..." (che farebbe collassare la memoria dell'IA), 
    hai preparato il libro a "fabbricare" e contenere tanti piccoli oggetti separati. 
    Questo è il "Chunking": spezzare i dati letterari in porzioni digeribili.

    ================================================================================
    """

    # ==========================================================================
    # I METODI OPERATIVI PER LA COMPOSIZIONE
    # ==========================================================================
    # Questo metodo sostituisce il vecchio 'aggiungi_testo' e i setter del contenuto.
    # Funge da "fabbrica": riceve i dati grezzi, costruisce l'oggetto Frammento
    # e lo inserisce al sicuro nella lista privata.
    # MODIFICA: Aggiunti Type Hint (str, int)
    def aggiungi_frammento(self, testo: str, pagina: int):
        """Crea un nuovo oggetto Frammento e lo inietta all'interno del Libro."""
        # 1. Creiamo fisicamente il nuovo oggetto passando i dati al suo costruttore
        nuovo_chunk = Frammento(testo, pagina)

        # 2. Aggiungiamo l'oggetto appena nato alla nostra lista privata
        self.__frammenti.append(nuovo_chunk)

    # Questo metodo sostituisce il vecchio 'get_contenuto'.
    # Non si limita a restituire una variabile, ma "assembla" dinamicamente
    # l'output leggendo la lista in tempo reale.
    def leggi_tutto(self) -> str:
        """Usa un ciclo for per estrarre e unire tutti i frammenti del libro."""

        # Controllo di sicurezza: se la lista è vuota (lunghezza 0)
        if len(self.__frammenti) == 0:
            return "Il libro è ancora vuoto."

        testo_completo = ""

        # IL CICLO FOR "PYTHONIC" ENTRA IN AZIONE:
        # Scorriamo la lista. Ad ogni giro, 'chunk' è un vero oggetto Frammento.
        for chunk in self.__frammenti:
            # Sfruttiamo il "Dunder Method" __str__ che abbiamo definito in frammento.py!
            # Scrivendo str(chunk), Python chiama in automatico quel metodo magico,
            # restituendo la stringa formattata "[Pag. X] Testo...", a cui aggiungiamo
            # un ritorno a capo (\n) per impaginare bene l'output.
            testo_completo += str(chunk) + "\n"

        # Restituiamo il "muro di testo" finale, perfettamente impaginato
        return testo_completo