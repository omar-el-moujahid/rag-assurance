"""Configuration centrale du projet : chemins, modèles et paramètres.
Changer de modèle ou de taille de chunk = changer une ligne ici."""
from pathlib import Path
import os
from dotenv import load_dotenv
# Chemins (calculés depuis l'emplacement de ce fichier, donc indépendants du dossier courant)
RACINE = Path(__file__).resolve().parent   # config.py est à la racine du projet
load_dotenv(RACINE / ".env") 

DOSSIER_CG = RACINE / "data" / "cg"
CATALOGUE = DOSSIER_CG / "catalogue.json"
DOSSIER_CP = RACINE / "data" / "cp"
QUESTIONS = RACINE / "data" / "eval" / "questions.json"
DOSSIER_RESULTATS = RACINE / "results"


# Base vectorielle
CHROMA_DIR = RACINE / "chroma_db"
COLLECTION = "cg_habitation"

# Modèles (Ollama, en local)
LLM_MODEL = "qwen2.5:7b"
EMBED_MODEL = "bge-m3"
GEN_MODEL = "qwen3:8b"     # génération des réponses ; LLM_MODEL (qwen2.5:7b) reste pour l'extraction

# Découpage (en caractères)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Recherche
TOP_K = 5


load_dotenv()  # charge le fichier .env dans les variables d'environnement
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")