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
RERANK = False                              # False = recherche vectorielle seule (baseline v1)
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
N_CANDIDATS = 20                           # candidats donnés au reranker

PROMPT = """Tu es un assistant qui aide un assuré à comprendre SON contrat d'assurance habitation.

Règles :
1. Réponds UNIQUEMENT à partir des documents ci-dessous. N'utilise pas tes connaissances générales.
2. Les Conditions Particulières (CP) sont propres au client et PRÉVALENT sur les Conditions Générales (CG).
3. Cite tes sources après chaque information : [CP] pour les conditions particulières, [1], [2]... pour les extraits des CG.
4. Si les documents ne parlent PAS DU TOUT du sujet, réponds : « Je ne trouve pas cette information dans vos documents. »
   S'ils y répondent EN PARTIE, donne ce qu'ils disent avec les sources, puis précise ce qui n'y figure pas.
5. Ne conclus « oui » que si un extrait le dit clairement. Sinon, réponds « oui sous conditions » ou « à vérifier ».
6. Réponds en français, en suivant la forme de cet exemple (sur une autre question, ne recopie pas son contenu) :
Question : « Suis-je couvert si ma vitre est cassée ? »
Réponse : oui sous conditions.
Votre contrat : la garantie bris de glace est souscrite, avec une franchise de 100 € [CP].
Conditions et exclusions : les vitres du logement sont couvertes [2], mais pas les simples rayures [4].

=== CONDITIONS PARTICULIÈRES DU CLIENT ===
{cp}

=== EXTRAITS DES CONDITIONS GÉNÉRALES ===
{passages}

=== QUESTION ===
{question}
"""

load_dotenv()  # charge le fichier .env dans les variables d'environnement
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")