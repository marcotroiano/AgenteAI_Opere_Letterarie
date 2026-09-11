import requests
import socket
import warnings

# Disabilitiamo i noiosi avvisi di sicurezza (InsecureRequestWarning)
# quando facciamo richieste HTTPS senza certificati validi (tipico in reti aziendali chiuse)
warnings.filterwarnings('ignore', message='Unverified HTTPS request')

# ==============================================================================
# CONFIGURAZIONE TARGET
# ==============================================================================
# Inserisci qui l'IP aziendale fornito
ip_azienda = "IP_AZIENDALE"
# Inserisci la porta specifica su cui vuoi fare il test HTTP (es. quella di Ollama)
porta_http = "PORTA"

# ==============================================================================
# FASE 1: TEST HTTP/HTTPS SULLA PORTA SPECIFICA (Test Ollama/Web Server)
# ==============================================================================
print(f"\n==================================================")
print(f" 🔍 FASE 1: TEST PROTOCOLLO WEB (IP: {ip_azienda}, Porta: {porta_http})")
print(f"==================================================\n")

url_http = f"http://{ip_azienda}:{porta_http}/"
url_https = f"https://{ip_azienda}:{porta_http}/"

print("--- TENTATIVO HTTP ---")
try:
    # Aumentato il timeout a 10s per reti VPN lente
    risposta = requests.get(url_http, timeout=10)
    print(f"🟢 [SUCCESSO] Codice: {risposta.status_code}")
    print(f"Risposta Server: {risposta.text.strip()}")
except Exception as e:
    print(f"🔴 [ERRORE] Nessuna risposta HTTP: {e}")

print("\n--- TENTATIVO HTTPS ---")
try:
    risposta = requests.get(url_https, verify=False, timeout=10)
    print(f"🟢 [SUCCESSO] Codice: {risposta.status_code}")
    print(f"Risposta Server: {risposta.text.strip()}")
except Exception as e:
    print(f"🔴 [ERRORE] Nessuna risposta HTTPS: {e}")

# ==============================================================================
# FASE 1.1: ISPEZIONE METADATI OLLAMA (VERSIONE E MODELLI DISPONIBILI)
# ==============================================================================
print(f"\n==================================================")
print(f" 🔍 FASE 1.1: QUERY API ENDPOINTS OLLAMA")
print(f"==================================================\n")

# Endpoint standard di Ollama per ottenere la versione del server
url_version = f"http://{ip_azienda}:{porta_http}/api/version"
print("--- CONTROLLO VERSIONE OLLAMA ---")
try:
    resp_ver = requests.get(url_version, timeout=5)
    if resp_ver.status_code == 200:
        dati_ver = resp_ver.json()
        print(f"🟢 Versione rilevata: {dati_ver.get('version', 'N/D')}")
    else:
        print(f"⚠️ Endpoint /api/version ha risposto con codice: {resp_ver.status_code}")
except Exception as e:
    print(f"🔴 Impossibile recuperare la versione: {e}")

# Endpoint standard di Ollama per elencare tutti i modelli installati localmente sul server
url_tags = f"http://{ip_azienda}:{porta_http}/api/tags"
print("\n--- CONTROLLO MODELLI INSTALLATI (TAGS) ---")
try:
    resp_tags = requests.get(url_tags, timeout=10)
    if resp_tags.status_code == 200:
        dati_tags = resp_tags.json()
        modelli = dati_tags.get("models", [])
        if modelli:
            print(f"🟢 Trovati {len(modelli)} modelli disponibili:")
            for m in modelli:
                nome = m.get("name", "Sconosciuto")
                size_gb = m.get("size", 0) / (1024 ** 3)
                print(f"  • {nome:<30} (Dimensione: {size_gb:.2f} GB)")
        else:
            print("⚠️ Nessun modello presente nel catalogo del server.")
    else:
        print(f"⚠️ Endpoint /api/tags ha risposto con codice: {resp_tags.status_code}")
except Exception as e:
    print(f"🔴 Impossibile recuperare l'elenco dei modelli: {e}")

# ==============================================================================
# FASE 2: PORT SCANNER MIRATO (Ricerca di Qdrant o altri servizi)
# ==============================================================================
print(f"\n==================================================")
print(f" 🔍 FASE 2: PORT SCANNING MIRATO (Target: {ip_azienda})")
print(f"==================================================\n")

# Lista delle porte più comuni in architetture Enterprise AI:
porte_da_testare = [
    6333,  # Qdrant REST API
    6334,  # Qdrant gRPC
    11434,  # Ollama API (Standard)
    80,  # HTTP (Reverse Proxy Nginx/Apache)
    443,  # HTTPS (Reverse Proxy Nginx/Apache Secure)
    8000,  # FastAPI/Uvicorn (Standard Python Backend)
    8080,  # Alternative Web Server (Tomcat, etc.)
    9000,  # MinIO o servizi simili
]

for porta in porte_da_testare:
    # Creiamo un socket TCP (IPv4)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Timeout basso: se non risponde in 2 secondi, andiamo avanti
    sock.settimeout(2.0)

    # connect_ex tenta la connessione in modo silenzioso.
    # Restituisce 0 (successo) o un codice di errore (es. WinError 10061)
    risultato = sock.connect_ex((ip_azienda, porta))

    if risultato == 0:
        print(f"🟢 [APERTA]  La porta {porta} è in ascolto! (Servizio attivo)")
    else:
        print(f"🔴 [CHIUSA]  La porta {porta} non risponde (Codice errore sistema: {risultato})")

    # Chiudiamo sempre la connessione per pulizia
    sock.close()

print("\n🏁 Diagnostica di rete completata.")
