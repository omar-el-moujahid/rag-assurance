"""Recherche : passages des CG filtrés sur le contrat, puis reranking (cross-encoder).
Test :  python -m src.retrieval.retrieval data/cp/cp_01_luko_etendue.pdf "Suis-je couvert si on me vole mon vélo dans ma cave ?"
"""
import sys
from pathlib import Path

from langsmith import traceable
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from src.tracing.tracing import tracer

from config import CHROMA_DIR, COLLECTION, EMBED_MODEL, TOP_K, RERANK, RERANK_MODEL, N_CANDIDATS


def ouvrir_store() -> Chroma:
    """Rouvre la base construite à l'étape 2 (même modèle d'embeddings qu'à l'indexation)."""
    return Chroma(collection_name=COLLECTION,
                  embedding_function=OllamaEmbeddings(model=EMBED_MODEL),
                  persist_directory=str(CHROMA_DIR))


_reranker = None

def charger_reranker():
    """Charge le cross-encoder au premier besoin seulement, puis le réutilise."""
    global _reranker
    if _reranker is None:
        import torch
        from sentence_transformers import CrossEncoder
        appareil = "cuda" if torch.cuda.is_available() else "cpu"
        _reranker = CrossEncoder(RERANK_MODEL, max_length=512, device=appareil)
    return _reranker


@tracer("reranker")
def reclasser(question, documents, k):
    """Le cross-encoder note chaque (question, passage), puis on garde les k meilleurs, une seule fois par page."""
    scores = charger_reranker().predict([(question, d.page_content) for d in documents])
    classes = [d for _, d in sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)]

    resultat, pages_vues = [], set()
    for d in classes:
        page = (d.metadata["fichier"], d.metadata["page"])
        if page not in pages_vues:
            pages_vues.add(page)
            resultat.append(d)
        if len(resultat) == k:
            break
    return resultat


@tracer("rechercher", "retriever")
def rechercher(store, question, contrat_id, k=TOP_K, rerank=RERANK):
    """Pas de contrat -> aucune recherche. Sinon : vecteurs filtrés, puis reranking si activé."""
    if contrat_id is None:
        return []
    filtre = {"contrat_id": contrat_id}
    if not rerank:
        return store.similarity_search(question, k=k, filter=filtre)
    candidats = store.similarity_search(question, k=N_CANDIDATS, filter=filtre)
    return reclasser(question, candidats, k)


if __name__ == "__main__":
    from src.extraction_cp.extraction import analyser_cp

    fichier, question = Path(sys.argv[1]), sys.argv[2]
    texte_cp, fiche, contrat_id = analyser_cp(fichier)
    print("Contrat :", contrat_id)

    passages = rechercher(ouvrir_store(), question, contrat_id)
    print(f"\n{len(passages)} passages trouvés (rerank={RERANK}) :")
    for i, p in enumerate(passages, start=1):
        print(f"[{i}] {p.metadata['fichier']} p.{p.metadata['page']}")
        print("    ", p.page_content[:200].replace("\n", " "), "...")