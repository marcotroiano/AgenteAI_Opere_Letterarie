# ==============================================================================
# File: src/models/documento_base.py
# (SPOSTATO NELLA CARTELLA 'src/models/' IN QUANTO SCHEMA FONDAMENTALE DEI DATI)
# ==============================================================================

"""
    ================================================================================
    DOCUMENTAZIONE ARCHITETTURALE: CLASSE ASTRATTA DocumentoBase
    ================================================================================
    RIFERIMENTI AL MANUALE:
    - Modulo 1: OOP, Ereditarietà, Astrazione e Interfacce.
    - Appendice A: Mappatura OOP e contratti del software.

    1. RUOLO E COMPITI TECNICI (COSA FA)
    Questa classe rappresenta l'Interfaccia Astratta del sistema:
    - Impedisce l'istanza diretta: Ereditando da 'ABC', blocca la creazione di oggetti
      generici 'DocumentoBase()' che non avrebbero significato logico.
    - Dichiara i vincoli contrattuali: Tramite '@abstractmethod', impone alle classi
      figlie di implementare i metodi 'mostra_metadati()' e 'leggi_tutto()'.

    2. SCELTE INGEGNERISTICHE E PATTERN (PERCHÉ È SCRITTA COSÌ E DOVE SI TROVA)
    - POSIZIONAMENTO NELLA CARTELLA 'MODELS': Come best practice architetturale,
      le interfacce base e le definizioni dei dati risiedono nel layer dei modelli.
      L'orchestratore ('agent') e il database ('rag') dipenderanno da questo schema.
    - DESIGN BY CONTRACT: Definisce uno standard rigoroso per le classi concrete
      (es. LibroStorico). Garantisce che qualsiasi sorgente dati esponga la stessa
      interfaccia verso l'esterno.
    - PREVEDIBILITÀ DELLA PIPELINE RAG: Assicura che l'orchestratore e i motori di
      ingestion possano invocare l'estrazione del testo e dei metadati in totale
      sicurezza, senza dover gestire eccezioni per metodi mancanti o difformi.
    ================================================================================
"""

# ================================================================================
# IL PILASTRO DELL'ASTRAZIONE: LE CLASSI ASTRATTE E IL "CONTRATTO"
# ================================================================================
#
# Nel design del software orientato agli oggetti (OOP), una CLASSE ASTRATTA è un'entità
# puramente concettuale. Non nasce per essere trasformata in un oggetto reale (istanza),
# ma per fungere da PLANIMETRIA o da CONTRATTO LEGALE per tutte le classi figlie.
#
# Pensala così:
# - Classe Concreta (es. LibroStorico): È un edificio fisico in cui puoi entrare.
# - Classe Astratta (es. DocumentoBase): È il regolamento edilizio. Non ci puoi abitare dentro,
#   ma stabilisce le regole tassative che ogni edificio vero dovrà rispettare.
#
# ================================================================================
# A COSA SERVE E COSA GESTISCE?
# ================================================================================
#
# 1. GESTISCE L'IMPOSSIBILITÀ DI ESISTENZA (Sicurezza Logica)
#    Impedisce a chiunque di scrivere "doc = DocumentoBase()". Un documento generico
#    non ha senso che esista nel sistema. Bloccare l'istanza diretta evita di avere
#    oggetti "fantasma" in memoria privi di logica reale.
#
# 2. GESTISCE IL "CONTRATTO" (Standardizzazione dei Metodi)
#    Attraverso il meccanismo dei metodi astratti (@abstractmethod), la classe padre
#    dichiara solo il NOME e la FIRMA di un'azione, senza scriverne il codice (usa 'pass').
#    Questo firma un accordo: qualsiasi classe figlia che erediti dal padre si impegna
#    solennemente a implementare quel metodo. Se non lo fa, il programma si blocca.
#
# 3. GESTISCE LA PREVEDIBILITÀ (Blindatura del Duck Typing)
#    In un sistema complesso come una pipeline RAG per testi storici, avrai un ciclo
#    che elabora centinaia di sorgenti diverse (Libri, Manoscritti, Articoli, Papiri).
#    Grazie alla classe astratta, hai la CERTEZZA MATEMATICA che ogni singolo elemento
#    avrà il metodo .leggi_tutto(). Non servono più mille controlli "if/else" nel main.
#    Chiami il metodo in totale sicurezza: se l'oggetto esiste, il metodo c'è.
#
# ================================================================================
# COME FUNZIONA IN PYTHON? (Il Modulo 'abc')
# ================================================================================
# Python non ha una parola chiave nativa come "abstract" (presente in Java o C++).
# Usa un modulo della libreria standard chiamato 'abc' (Abstract Base Classes).
#
# - Una classe diventa astratta se eredita da 'ABC' (es. class DocumentoBase(ABC):).
# - Un metodo diventa un vincolo contrattuale se preceduto da '@abstractmethod'.
# - La trappola scatta all'istanza: se la figlia non ha implementato TUTTI i metodi
#   astratti, al momento del "m = MiaClasseFiglia()" Python solleva un TypeError
#   e blocca l'esecuzione prima ancora che il programma possa fare danni.
# ================================================================================

from abc import ABC, abstractmethod

class DocumentoBase(ABC):
    """
    CLASSE ASTRATTA (IL CONTRATTO)
    Qualsiasi documento voglia essere inserito nella nostra pipeline di estrazione
    deve obbligatoriamente implementare questi due metodi.
    """

    @abstractmethod
    def mostra_metadati(self):
        """Obbliga la classe figlia a restituire una stringa con i metadati."""
        pass

    @abstractmethod
    def leggi_tutto(self):
        """Obbliga la classe figlia a restituire l'intero testo estratto."""
        pass