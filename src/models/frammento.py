# ==============================================================================
# File: src/models/frammento.py
# (SPOSTATO NELLA CARTELLA 'src/models/' PER ARCHITETTURA MODULARE)
# ==============================================================================

"""
    ================================================================================
    DOCUMENTAZIONE ARCHITETTURALE: CLASSE Frammento
    ================================================================================
    RIFERIMENTI AL MANUALE:
    - Modulo 1: Composizione (Has-A).
    - Modulo 3: LlamaIndex (Concetto di Nodo / Chunking).
    - Appendice B.3: Il Chunking per prevenire Out of Memory e "Minestroni Semantici".

    1. RUOLO E COMPITI TECNICI (COSA FA)
    Questa classe rappresenta l'unità atomica di testo (il singolo Chunk/Nodo):
    - Validazione preventiva: Filtra testo e numero di pagina direttamente nel
      costruttore (__init__), bloccando l'istanza di oggetti corrotti (Fail-Fast).
    - Immutabilità: Espone solo i getter (@property) senza setter, impedendo
      qualsiasi modifica dello stato dell'oggetto dopo la sua nascita.
    - Formattazione: Implementa il metodo magico '__str__' per restituire un
      rappresentazione testuale pulita [Pag. X] Testo.

    2. SCELTE INGEGNERISTICHE E PATTERN (PERCHÉ È SCRITTA COSÌ)
    - IMMUTABILITÀ DEI NODI: Previene alterazioni accidentali o manomissioni del
      testo durante il passaggio attraverso le varie fasi della pipeline RAG.
    - PREPARAZIONE AL CHUNKING: L'IA e i modelli di embedding faticano su testi
      troppo lunghi. Questa classe permette di dividere il libro in frammenti
      densi di significato spaziale, ottimizzati per la Cosine Similarity.
    - SELF-ENCAPSULATION: Utilizza le property interne anche dentro i metodi della
      classe stessa (es. in __str__), garantendo manutenibilità.
    ================================================================================
"""

# ================================================================================
# FILE: frammento.py
# OBIETTIVO: Rappresentare la più piccola unità di testo indipendente (Chunk).
# ================================================================================
# Questa classe è progettata per essere "IMMUTABILE". Nel mondo dell'architettura
# software, un oggetto immutabile è un oggetto il cui stato non può essere
# modificato dopo la sua creazione. Questo previene alterazioni accidentali
# quando l'oggetto viene passato tra decine di funzioni diverse.
#
# AGGIORNAMENTO (VALIDATORI):
# Essendo immutabile, non abbiamo i setter. Ma dobbiamo comunque impedire che
# l'oggetto "nasca" corrotto. Per questo, i validatori privati intervengono
# direttamente e unicamente nel Costruttore. Se i dati sono errati, l'oggetto
# non viene nemmeno creato (Fail-Fast).
# ================================================================================

# DIARIO DI BORDO / SPIEGAZIONE DELLE MODIFICHE DELL'ARCHITETTURA:
# --------------------------------------------------------------------------------
# Aggiunta del Type Hinting (Tipizzazione Statica):
# Per allinearci agli standard enterprise e a librerie come Pydantic e LlamaIndex,
# abbiamo introdotto le "Type Hints" (es. testo: str, pagina: int -> str).
# Questo non altera la logica a runtime (Python rimane dinamico), ma aiuta enormemente
# l'IDE (PyCharm) a suggerire autocompletamenti e a rilevare errori di tipo in fase di
# scrittura del codice.
# --------------------------------------------------------------------------------


# ================================================================================
# L'ANATOMIA DELLA COMPOSIZIONE IN 2 STEP
# ================================================================================
# Come avviene l'interazione quando nel main si chiama:
# divina_commedia.aggiungi_frammento("Nel mezzo del cammin...", 1)
#
# Dietro le quinte, il metodo operativo aggiungi_frammento esegue due azioni:
#
# 1. Fabbrica l'oggetto: Usa la classe esterna Frammento come un vero e proprio
#    "stampo industriale". Prende il testo e il numero di pagina passati dall'utente
#    e li usa per creare un'entità fisica e indipendente (la variabile locale nuovo_chunk).
#
# 2. Inietta l'oggetto: Prende questo oggetto appena nato e lo "spinge" all'interno
#    della lista privata (self.__frammenti.append()) della classe LibroStorico.
#
# In sintesi: l'utente chiede al Libro di aggiungere del testo, e il Libro si
# occupa dinamicamente di creare un oggetto della classe Frammento e di
# collezionarlo al suo interno in modo ordinato.
# ================================================================================


class Frammento:

    # ==========================================================================
    # 1. IL COSTRUTTORE E I VALIDATORI (Sicurezza alla nascita)
    # ==========================================================================
    # MODIFICA: Aggiunti i Type Hint (str, int) ai parametri.
    def __init__(self, testo: str, pagina: int):
        # I dati vengono salvati in attributi STRETTAMENTE PRIVATI (__x).
        # Prima di essere salvati, passano attraverso i filtri dei validatori.
        # Questo garantisce che nessuno, dall'esterno, possa fare qualcosa come:
        # mio_frammento.testo = "Testo manomesso" (bloccato dall'assenza di setter)
        # o creare Frammento("", -5) (bloccato dai validatori qui sotto).
        self.__testo = self.__valida_testo(testo)
        self.__pagina = self.__valida_pagina(pagina)

    # ==========================================================================
    # VALIDATORI PRIVATI (La dogana del Costruttore)
    # ==========================================================================
    # MODIFICA: Aggiunti Type Hint (input: str, output: str)
    def __valida_testo(self, testo: str) -> str:
        """Controlla che il testo non sia vuoto e sia una stringa."""
        if not isinstance(testo, str) or not testo.strip():
            raise ValueError("Errore: Il testo del frammento deve essere valido e non vuoto.")
        return testo.strip()

    # MODIFICA: Aggiunti Type Hint (input: int, output: int)
    def __valida_pagina(self, pagina: int) -> int:
        """Controlla che il numero di pagina sia un intero positivo."""
        if not isinstance(pagina, int) or pagina <= 0:
            raise ValueError("Errore: Il numero di pagina deve essere un intero positivo maggiore di zero.")
        return pagina

    # ==========================================================================
    # 2. INTERFACCIA DI SOLA LETTURA (I Getter)
    # ==========================================================================
    # Usiamo le @property per permettere all'utente di leggere i dati in modo
    # pulito (es. print(frammento.testo)), ma omettiamo volutamente i @setter.
    # Senza setter, Python bloccherà qualsiasi tentativo di scrittura,
    # garantendo la totale immutabilità dell'oggetto.

    @property
    def testo(self) -> str:
        """Restituisce il contenuto testuale del frammento."""
        return self.__testo

    @property
    def pagina(self) -> int:
        """Restituisce il numero di pagina a cui appartiene il frammento."""
        return self.__pagina

    # ==========================================================================
    # 3. IL METODO MAGICO PER LA STAMPA (Dunder Method)
    # ==========================================================================
    # In Python, i metodi circondati dal doppio underscore (come __init__ o __str__)
    # si chiamano "Dunder Methods" (Double UNDERscore) o metodi magici.
    # Non vanno quasi mai chiamati manualmente, ci pensa Python a usarli
    # dietro le quinte quando serve.

    def __str__(self) -> str:
        """
        Insegna a Python come convertire questo oggetto in una stringa leggibile.
        Invece di creare un metodo "stampa_frammento()", ridefiniamo __str__.
        In questo modo, quando nel codice principale faremo semplicemente:
        print(mio_frammento)
        Python eseguirà automaticamente questo metodo, formattando l'output.
        """
        # Nota come, anche all'interno della classe stessa, usiamo self.pagina
        # e self.testo (che chiamano le @property) invece dei campi privati
        # __pagina e __testo. È l'applicazione pura del Self-Encapsulation!
        return f"[Pag. {self.pagina}] {self.testo}"