import time
from langchain_ollama import ChatOllama
from config import GEN_MODEL
from src.retrieval.retrieval import rechercher
from langsmith import traceable
from src.tracing.tracing import callbacks
from src.tracing.tracing import tracer

MESSAGE_HORS_BASE = ("Les Conditions Générales de votre contrat ne sont pas dans notre base. "
                     "Je ne peux pas répondre de façon fiable : déposez vos CG ou contactez votre assureur.")

import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent.parent))

from config import DOSSIER_CP , PROMPT
from src.extraction_cp.extraction import analyser_cp
from src.retrieval.retrieval import ouvrir_store, rechercher

from langchain_core.tracers.langchain import wait_for_all_tracers

store = ouvrir_store()
texte_cp, fiche, contrat_id = analyser_cp(DOSSIER_CP / "cp_01_luko_etendue.pdf")
question = "Suis-je couvert si on me vole mon vélo dans ma cave ?"
passages = rechercher(store, question, contrat_id)
# print(contrat_id, "|", len(passages), "passages")

llm_gen = ChatOllama(model=GEN_MODEL, temperature=0, num_ctx=8192, num_predict=600, reasoning=False)


def formater_passages(passages):
    """Numérote chaque passage et indique sa source, pour que le LLM puisse la citer."""
    blocs = []
    for i, p in enumerate(passages, start=1):
        m = p.metadata
        blocs.append(f"[{i}] (CG {m['assureur']}, {m['fichier']}, page {m['page']})\n{p.page_content}")
    return "\n\n".join(blocs)

import re

def ajouter_sources(texte_reponse, passages):
    """Remplace les numéros cités par le LLM par les vraies sources (fichier + page)."""
    numeros = sorted({int(n) for n in re.findall(r"\[(\d+)\]", texte_reponse)})
    lignes = []
    if "[CP]" in texte_reponse:
        lignes.append("[CP] Vos Conditions Particulières")
    for n in numeros:
        if 1 <= n <= len(passages):
            m = passages[n - 1].metadata
            lignes.append(f"[{n}] {m['fichier']}, page {m['page']}")
        else:
            lignes.append(f"[{n}] ⚠️ citation inexistante")
    return texte_reponse + ("\n\nSources :\n" + "\n".join(lignes) if lignes else "")

@tracer("repondre")
def repondre(store, question, texte_cp, contrat_id):
    """Stratégie stricte -> recherche filtrée -> prompt -> LLM -> sources ajoutées par le code."""
    if contrat_id is None:                                   # pas de contrat : ni recherche ni LLM
        return {"reponse": MESSAGE_HORS_BASE, "passages": [], "temps": 0.0}

    debut = time.time()
    passages = rechercher(store, question, contrat_id)
    prompt = PROMPT.format(cp=texte_cp, passages=formater_passages(passages), question=question)
    texte = llm_gen.invoke(prompt, config={"callbacks": callbacks()}).content
    return {"reponse": ajouter_sources(texte, passages),
            "passages": passages,
            "temps": time.time() - debut}

# if __name__ == "__main__":
#     fichier, question = Path(sys.argv[1]), sys.argv[2]
#     debut = time.time()
#     texte_cp, fiche, contrat_id = analyser_cp(fichier)
#     print("Contrat :", contrat_id)

#     resultat = repondre(store, question, texte_cp, contrat_id)
#     print(f"\nRéponse en {resultat['temps']:.1f} s :\n{resultat['reponse']}")
#     print(f"temps de recherche : {resultat['temps']:.1f} s")
#     wait_for_all_tracers()

@tracer("demande_complete", "chain")
def demande_complete(chemin_cp, question):
    store = ouvrir_store()
    texte_cp, fiche, contrat_id = analyser_cp(chemin_cp)
    return repondre(store, question, texte_cp, contrat_id)


if __name__ == "__main__":
    resultat = demande_complete(Path(sys.argv[1]), sys.argv[2])
    print(resultat["reponse"])
    wait_for_all_tracers()