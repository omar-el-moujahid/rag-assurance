"""Étape 2 : indexation des CG/IPID.
PDF -> texte par page (nettoyé) -> chunks -> embeddings (bge-m3) -> Chroma, avec métadonnées.
Lancer depuis la racine du projet :  python -m etape2_indexation.indexing
"""
import json
import re
import shutil
import time

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from config import (CATALOGUE, DOSSIER_CG, CHROMA_DIR, COLLECTION,
                    EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP)


def nettoyer(texte: str) -> str:
    """Normalise le texte extrait du PDF."""
    # Apostrophes typographiques (ʼ ’) -> apostrophe simple : utile pour la recherche par mots-clés
    texte = texte.replace("ʼ", "'").replace("’", "'")
    texte = texte.replace(" ", " ")            # espaces insécables
    texte = re.sub(r"[ \t]+", " ", texte)           # espaces multiples
    texte = re.sub(r"\n\s*\n+", "\n\n", texte)      # lignes vides multiples
    return texte.strip()


def metadonnees_pour_chroma(entree: dict) -> dict:
    """Chroma n'accepte que str/int/float/bool : on convertit les listes et les None."""
    meta = {}
    for cle, valeur in entree.items():
        if valeur is None:
            meta[cle] = ""
        elif isinstance(valeur, list):
            meta[cle] = ", ".join(valeur)
        else:
            meta[cle] = valeur
    return meta


def charger_pages(entree: dict) -> list[Document]:
    """Un Document LangChain par page, avec le numéro de page dans les métadonnées."""
    reader = PdfReader(DOSSIER_CG / entree["fichier"])
    base = metadonnees_pour_chroma(entree)
    pages = []
    for numero, page in enumerate(reader.pages, start=1):
        texte = nettoyer(page.extract_text() or "")
        if len(texte) < 50:          # page vide ou quasi vide (couverture, page blanche)
            continue
        pages.append(Document(page_content=texte, metadata={**base, "page": numero}))
    return pages


def construire_chunks() -> list[Document]:
    catalogue = json.loads(CATALOGUE.read_text(encoding="utf-8"))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],   # on coupe d'abord entre paragraphes
    )
    tous = []
    for entree in catalogue:
        pages = charger_pages(entree)
        chunks = splitter.split_documents(pages)   # découpe page par page : la page reste exacte
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = f"{entree['fichier']}#{i}"
        print(f"{entree['fichier']:42} {len(pages):4} pages -> {len(chunks):4} chunks")
        tous.extend(chunks)
    return tous


def indexer(chunks: list[Document]) -> Chroma:
    # On repart de zéro à chaque indexation : résultat reproductible
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    store = Chroma(
        collection_name=COLLECTION,
        embedding_function=OllamaEmbeddings(model=EMBED_MODEL),
        persist_directory=str(CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},   # distance cosinus
    )
    debut = time.time()
    taille_lot = 64
    for i in range(0, len(chunks), taille_lot):
        lot = chunks[i:i + taille_lot]
        store.add_documents(lot, ids=[c.metadata["chunk_id"] for c in lot])
        print(f"  embeddings : {min(i + taille_lot, len(chunks))}/{len(chunks)}", end="\r")
    print(f"\nIndexation terminée en {time.time() - debut:.0f} s")
    return store


def test_recherche(store: Chroma):
    """Vérification rapide : la recherche filtrée sur un contrat renvoie-t-elle des passages pertinents ?"""
    question = "vol de vélo dans la cave"
    resultats = store.similarity_search_with_score(
        question, k=3, filter={"contrat_id": "luko_hce_2504"})
    print(f"\nTest : « {question} » (filtre luko_hce_2504)")
    for doc, distance in resultats:
        m = doc.metadata
        print(f"- {m['fichier']} p.{m['page']} | distance={distance:.3f}")
        print("  ", doc.page_content[:200].replace("\n", " "), "...")


if __name__ == "__main__":
    chunks = construire_chunks()
    longueurs = [len(c.page_content) for c in chunks]
    print(f"\nTotal : {len(chunks)} chunks, longueur moyenne {sum(longueurs) / len(longueurs):.0f} caractères")
    store = indexer(chunks)
    test_recherche(store)