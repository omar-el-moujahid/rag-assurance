"""Recherche : les passages des CG les plus proches d'une question, filtrés sur le contrat.
Test :  python -m src.retrieval.retrieval data/cp/cp_01_luko_etendue.pdf "Suis-je couvert si on me vole mon vélo dans ma cave ?"
"""
import sys
from pathlib import Path
from langsmith import traceable
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from config import CHROMA_DIR, COLLECTION, EMBED_MODEL, TOP_K


def ouvrir_store() -> Chroma:
    """Rouvre la base construite à l'étape 2 (même modèle d'embeddings qu'à l'indexation)."""
    return Chroma(collection_name=COLLECTION,
                  embedding_function=OllamaEmbeddings(model=EMBED_MODEL),
                  persist_directory=str(CHROMA_DIR))


@traceable(name="rechercher")
def rechercher(store, question, contrat_id, k=TOP_K):
    """Les k passages les plus proches, uniquement dans les CG du contrat. Pas de contrat -> aucune recherche."""
    if contrat_id is None:
        return []
    return store.similarity_search(question, k=k, filter={"contrat_id": contrat_id})


if __name__ == "__main__":
    from src.extraction_cp.extraction import analyser_cp

    fichier, question = Path(sys.argv[1]), sys.argv[2]
    texte_cp, fiche, contrat_id = analyser_cp(fichier)
    print("Contrat :", contrat_id)

    passages = rechercher(ouvrir_store(), question, contrat_id)
    print(f"\n{len(passages)} passages trouvés :")
    for i, p in enumerate(passages, start=1):
        m = p.metadata
        print(f"[{i}] {m['fichier']} p.{m['page']}")
        print("    ", p.page_content[:200].replace("\n", " "), "...")