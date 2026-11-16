import time
from langchain_ollama import ChatOllama
from config import GEN_MODEL
from src.retrieval.retrieval import rechercher
from langsmith import traceable

MESSAGE_HORS_BASE = ("Les Conditions Générales de votre contrat ne sont pas dans notre base. "
                     "Je ne peux pas répondre de façon fiable : déposez vos CG ou contactez votre assureur.")

import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent.parent))

from config import DOSSIER_CP
from src.extraction_cp.extraction import analyser_cp
from src.retrieval.retrieval import ouvrir_store, rechercher

from langchain_core.tracers.langchain import wait_for_all_tracers

store = ouvrir_store()
texte_cp, fiche, contrat_id = analyser_cp(DOSSIER_CP / "cp_01_luko_etendue.pdf")
question = "Suis-je couvert si on me vole mon vélo dans ma cave ?"
passages = rechercher(store, question, contrat_id)
# print(contrat_id, "|", len(passages), "passages")

llm_gen = ChatOllama(model=GEN_MODEL, temperature=0, num_ctx=8192, num_predict=600, reasoning=False)
PROMPT = """Tu es un assistant qui aide un assuré à comprendre SON contrat d'assurance habitation.

Règles :
1. Réponds UNIQUEMENT à partir des documents ci-dessous. N'utilise pas tes connaissances générales.
2. Les Conditions Particulières (CP) sont propres au client et PRÉVALENT sur les Conditions Générales (CG).
3. Cite tes sources après chaque information : [CP] pour les conditions particulières, [1], [2]... pour les extraits des CG.
4. Si les documents ne parlent PAS DU TOUT du sujet, réponds : « Je ne trouve pas cette information dans vos documents. »
   S'ils y répondent EN PARTIE, donne ce qu'ils disent avec les sources, puis précise ce qui n'y figure pas.
5. Ne conclus « oui » que si un extrait le dit clairement. Sinon, réponds « oui sous conditions » ou « à vérifier ».
6. Réponds en français, avec exactement cette structure :
Réponse : oui / non / oui sous conditions / à vérifier / je ne sais pas, en une phrase.
Votre contrat : ce que prévoient les CP (garantie souscrite ou non, franchise, plafonds) [CP].
Conditions et exclusions : ce que disent les CG, avec leurs numéros [1], [2]...

=== CONDITIONS PARTICULIÈRES DU CLIENT ===
{cp}

=== EXTRAITS DES CONDITIONS GÉNÉRALES ===
{passages}

=== QUESTION ===
{question}
"""

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

@traceable(name="repondre", run_type="llm")
def repondre(store, question, texte_cp, contrat_id):
    """Stratégie stricte -> recherche filtrée -> prompt -> LLM -> sources ajoutées par le code."""
    if contrat_id is None:                                   # pas de contrat : ni recherche ni LLM
        return {"reponse": MESSAGE_HORS_BASE, "passages": [], "temps": 0.0}

    debut = time.time()
    passages = rechercher(store, question, contrat_id)
    prompt = PROMPT.format(cp=texte_cp, passages=formater_passages(passages), question=question)
    texte = llm_gen.invoke(prompt).content
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

@traceable(name="demande_complete")
def demande_complete(chemin_cp, question):
    store = ouvrir_store()
    texte_cp, fiche, contrat_id = analyser_cp(chemin_cp)
    return repondre(store, question, texte_cp, contrat_id)


if __name__ == "__main__":
    resultat = demande_complete(Path(sys.argv[1]), sys.argv[2])
    print(resultat["reponse"])
    wait_for_all_tracers()